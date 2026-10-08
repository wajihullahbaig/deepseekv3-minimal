"""Decoding for the minimal DeepSeek-V3 model: greedy / sampling, beam search and speculative
decoding that uses the MTP head as a draft model and verifies every draft token."""
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GenerationConfig:
    """Configuration for text generation."""

    def __init__(self, max_length: int = 128, temperature: float = 1.2, top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.2, no_repeat_ngram_size: int = 3,
                 eos_token_id: Optional[int] = None, pad_token_id: Optional[int] = None,
                 do_sample: bool = True, use_mtp: bool = True, mtp_speculation_mode: bool = True,
                 num_beams: int = 1, length_penalty: float = 1.5, early_stopping: bool = False):
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.do_sample = do_sample
        self.use_mtp = use_mtp
        self.mtp_speculation_mode = mtp_speculation_mode
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping


class TextGenerator:
    """Text generator for the DeepSeek model with several decoding strategies."""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    # ------------------------------------------------------------------ public API
    def generate(self, prompts: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        single = isinstance(prompts, str)
        prompts = [prompts] if single else prompts
        if config.num_beams > 1:
            texts = self.generate_with_beam_search(prompts, config)
        elif config.use_mtp and config.mtp_speculation_mode:
            texts = self.generate_with_speculation(prompts, config)
        else:
            texts = self.generate_standard(prompts, config)
        return texts[0] if single else texts

    # ------------------------------------------------------------------ helpers
    def _encode_prompt(self, prompt: str, config: GenerationConfig) -> torch.Tensor:
        """Prompt ids WITHOUT a trailing EOS: the model is asked to continue the text, not to start over."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=False, truncation=True,
                                    max_length=max(1, config.max_length // 2), return_tensors="pt")
        if ids.numel() == 0:
            ids = torch.tensor([[self.tokenizer.pad_token_id]])
        return ids.to(self.device)

    def _forward(self, generated: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.model(generated, attention_mask=torch.ones_like(generated))
        if isinstance(outputs, tuple):
            return outputs[0], outputs[1]
        return outputs, None

    def _filter_logits(self, logits: torch.Tensor, context: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        """Temperature, repetition penalty, n-gram ban, top-k and top-p on one [1, V] row."""
        logits = logits.clone().float()
        if config.temperature > 0 and config.temperature != 1.0:
            logits = logits / config.temperature
        if config.repetition_penalty != 1.0 and context.numel():
            seen = context[0].unique()
            vals = logits[0, seen]
            logits[0, seen] = torch.where(vals > 0, vals / config.repetition_penalty, vals * config.repetition_penalty)
        n = config.no_repeat_ngram_size
        if n > 0 and context.size(1) > n:
            banned = self._get_banned_tokens(context[0], self._get_ngrams(context[0], n), n)
            if banned:
                logits[0, banned] = -float("inf")
        if config.top_k > 0:
            logits = self._top_k_filtering(logits, min(config.top_k, logits.size(-1)))
        if config.top_p < 1.0:
            logits = self._top_p_filtering(logits, config.top_p)
        return logits

    def _pick(self, probs: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        if config.do_sample:
            return torch.multinomial(probs, num_samples=1)
        return probs.argmax(dim=-1, keepdim=True)

    def _next_probs(self, logits_row, context, config):
        return F.softmax(self._filter_logits(logits_row, context, config), dim=-1)

    # ------------------------------------------------------------------ strategies
    def generate_standard(self, prompts: List[str], config: GenerationConfig) -> List[str]:
        """Plain auto-regressive decoding (greedy or sampled)."""
        texts = []
        for prompt_idx, prompt in enumerate(prompts):
            generated = self._encode_prompt(prompt, config)
            max_new = max(0, config.max_length - generated.size(1))
            with torch.no_grad(), tqdm(total=max_new, desc=f"Generating text {prompt_idx + 1}/{len(prompts)}", leave=False) as pbar:
                for _ in range(max_new):
                    logits, _ = self._forward(generated)
                    next_token = self._pick(self._next_probs(logits[:, -1], generated, config), config)
                    generated = torch.cat([generated, next_token], dim=1)
                    pbar.update(1)
                    if next_token.item() == config.eos_token_id:
                        break
            texts.append(self.tokenizer.decode(generated[0], skip_special_tokens=True))
        return texts

    def generate_with_speculation(self, prompts: List[str], config: GenerationConfig) -> List[str]:
        """Speculative decoding with the MTP head as the draft model.

        Each forward yields the main head's next token *and* an MTP draft for the token after
        it.  Both are appended; the next forward verifies the draft against the main head's
        distribution at that position (exact match for greedy, acceptance probability
        min(1, p/q) for sampling) and replaces it on rejection.  Accepted drafts give two
        tokens per forward; greedy decoding stays deterministic.
        """
        texts = []
        for prompt_idx, prompt in enumerate(prompts):
            generated = self._encode_prompt(prompt, config)
            drafts = accepted = 0
            draft_q = None                                   # draft distribution for generated[:, -1], if any
            max_new = max(0, config.max_length - generated.size(1))
            with torch.no_grad(), tqdm(total=max_new, desc=f"Generating text {prompt_idx + 1}/{len(prompts)}", leave=False) as pbar:
                while generated.size(1) < config.max_length:
                    main_logits, mtp_logits = self._forward(generated)
                    if draft_q is not None:
                        draft = generated[0, -1].item()
                        p = self._next_probs(main_logits[:, -2], generated[:, :-1], config)
                        if config.do_sample:
                            ok = torch.rand(1).item() < min(1.0, p[0, draft].item() / max(draft_q[0, draft].item(), 1e-12))
                        else:
                            ok = p.argmax().item() == draft
                        if ok:
                            accepted += 1
                        else:
                            if config.do_sample:
                                residual = torch.clamp(p - draft_q, min=0)
                                p = residual / residual.sum() if residual.sum() > 0 else p
                            generated[0, -1] = self._pick(p, config).item()
                            main_logits, mtp_logits = self._forward(generated)   # logits of the last position changed
                        draft_q = None
                        if generated[0, -1].item() == config.eos_token_id or generated.size(1) >= config.max_length:
                            break
                    next_token = self._pick(self._next_probs(main_logits[:, -1], generated, config), config)
                    generated = torch.cat([generated, next_token], dim=1)
                    pbar.update(1)
                    if next_token.item() == config.eos_token_id or generated.size(1) >= config.max_length:
                        break
                    if config.use_mtp and mtp_logits is not None:
                        draft_q = self._next_probs(mtp_logits[:, 0, -1], generated, config)   # predicts the token after next_token
                        generated = torch.cat([generated, self._pick(draft_q, config)], dim=1)
                        drafts += 1
                        pbar.update(1)
            if drafts:
                logger.info(f"Speculation: {accepted}/{drafts} MTP drafts accepted ({100 * accepted / drafts:.0f}%)")
            texts.append(self.tokenizer.decode(generated[0], skip_special_tokens=True))
        return texts

    def _length_normalised(self, score: float, length: int, config: GenerationConfig) -> float:
        return score / (((5.0 + length) / 6.0) ** config.length_penalty)

    def generate_with_beam_search(self, prompts: List[str], config: GenerationConfig) -> List[str]:
        texts = []
        for prompt_idx, prompt in enumerate(prompts):
            input_ids = self._encode_prompt(prompt, config)
            beams = [(input_ids, 0.0)]
            finished = []
            max_new = max(0, config.max_length - input_ids.size(1))
            with torch.no_grad(), tqdm(total=max_new, desc=f"Beam search {prompt_idx + 1}/{len(prompts)}", leave=False) as pbar:
                for _ in range(max_new):
                    candidates = []
                    for seq, score in beams:
                        logits, _ = self._forward(seq)
                        log_probs = F.log_softmax(self._filter_logits(logits[:, -1], seq, config), dim=-1)[0]
                        top_lp, top_tok = torch.topk(log_probs, k=min(config.num_beams, log_probs.numel()))
                        for lp, tok in zip(top_lp.tolist(), top_tok.tolist()):
                            if lp > -float("inf"):
                                candidates.append((torch.cat([seq, torch.tensor([[tok]], device=self.device)], dim=1), score + lp))
                    if not candidates:
                        break
                    candidates.sort(key=lambda c: self._length_normalised(c[1], c[0].size(1), config), reverse=True)
                    beams = []
                    for seq, score in candidates[:config.num_beams]:
                        (finished if seq[0, -1].item() == config.eos_token_id else beams).append((seq, score))
                    pbar.update(1)
                    if not beams or (config.early_stopping and len(finished) >= config.num_beams):
                        break
            pool = finished or beams
            best = max(pool, key=lambda c: self._length_normalised(c[1], c[0].size(1), config))[0] if pool else input_ids
            texts.append(self.tokenizer.decode(best[0], skip_special_tokens=True))
        return texts

    # ------------------------------------------------------------------ filters
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        filtered = logits.clone()
        filtered[indices_to_remove] = -float("inf")
        return filtered

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()       # keep the first token above the threshold
        remove[..., 0] = False
        indices_to_remove = remove.scatter(dim=1, index=sorted_indices, src=remove)
        filtered = logits.clone()
        filtered[indices_to_remove] = -float("inf")
        return filtered

    def _get_ngrams(self, token_ids: torch.Tensor, n: int) -> List[Tuple[int, ...]]:
        return [tuple(token_ids[i:i + n].tolist()) for i in range(len(token_ids) - n + 1)]

    def _get_banned_tokens(self, token_ids: torch.Tensor, ngrams: List[Tuple[int, ...]], n: int) -> List[int]:
        """Tokens that would complete an n-gram that already occurred."""
        if len(token_ids) < n - 1:
            return []
        prefix = tuple(token_ids[-(n - 1):].tolist())
        return [ngram[-1] for ngram in ngrams if ngram[:-1] == prefix]
