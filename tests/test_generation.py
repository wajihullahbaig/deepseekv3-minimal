import pytest
import torch

from conftest import freeze_router
from text_generation import GenerationConfig, TextGenerator


@pytest.fixture
def gen(tiny_model, stub_tokenizer):
    freeze_router(tiny_model)
    return TextGenerator(tiny_model, stub_tokenizer, device="cpu")


def _cfg(tok, **kw):
    base = dict(max_length=24, eos_token_id=None, pad_token_id=tok.pad_token_id,
                use_mtp=False, mtp_speculation_mode=False)
    base.update(kw)
    return GenerationConfig(**base)


PROMPT = "the quick brown fox"


def test_greedy_is_deterministic_and_bounded(gen, stub_tokenizer):
    cfg = _cfg(stub_tokenizer, do_sample=False)
    a, b = gen.generate(PROMPT, cfg), gen.generate(PROMPT, cfg)
    assert isinstance(a, str) and a == b
    assert len(a.split()) <= cfg.max_length


def test_sampling_is_reproducible_under_a_seed(gen, stub_tokenizer):
    cfg = _cfg(stub_tokenizer, do_sample=True, temperature=1.0, top_k=10, top_p=0.9)
    torch.manual_seed(1)
    a = gen.generate(PROMPT, cfg)
    torch.manual_seed(1)
    b = gen.generate(PROMPT, cfg)
    assert a == b and len(a.split()) <= cfg.max_length


def test_beam_search_returns_text(gen, stub_tokenizer):
    out = gen.generate(PROMPT, _cfg(stub_tokenizer, do_sample=False, num_beams=3, length_penalty=1.0))
    assert isinstance(out, str) and len(out.split()) <= 24


def test_mtp_speculation_path_runs(gen, stub_tokenizer):
    torch.manual_seed(0)
    out = gen.generate(PROMPT, _cfg(stub_tokenizer, do_sample=True, use_mtp=True, mtp_speculation_mode=True))
    assert isinstance(out, str) and len(out.split()) <= 24


def test_list_of_prompts_returns_list(gen, stub_tokenizer):
    out = gen.generate([PROMPT, "hello world"], _cfg(stub_tokenizer, do_sample=False))
    assert isinstance(out, list) and len(out) == 2


def test_top_k_and_top_p_filtering(gen):
    logits = torch.randn(1, 64)
    k = gen._top_k_filtering(logits, 5)
    assert torch.isfinite(k).sum() == 5
    p = gen._top_p_filtering(logits, 0.5)
    assert torch.isfinite(p).sum() >= 1 and torch.isfinite(p[0, logits.argmax()])


def test_banned_tokens_prevent_repeated_ngrams(gen):
    seq = torch.tensor([1, 2, 3, 1, 2])
    ngrams = gen._get_ngrams(seq, 3)
    assert ngrams == [(1, 2, 3), (2, 3, 1), (3, 1, 2)]
    assert gen._get_banned_tokens(seq, ngrams, 3) == [3]


def test_greedy_mtp_speculation_is_deterministic(gen, stub_tokenizer):
    cfg = _cfg(stub_tokenizer, do_sample=False, use_mtp=True, mtp_speculation_mode=True)
    outs = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        outs.append(gen.generate(PROMPT, cfg))
    assert outs[0] == outs[1] == outs[2]


def test_beam_search_generates_past_the_prompt(gen, stub_tokenizer):
    cfg = _cfg(stub_tokenizer, do_sample=False, num_beams=3, eos_token_id=stub_tokenizer.eos_token_id)
    out = gen.generate(PROMPT, cfg)
    assert len(out.split()) > len(PROMPT.split())


def test_prompt_is_encoded_without_eos(gen, stub_tokenizer):
    ids = gen._encode_prompt(PROMPT, _cfg(stub_tokenizer))
    assert ids.shape == (1, 4) and stub_tokenizer.eos_token_id not in ids[0].tolist()


def test_repetition_penalty_lowers_seen_tokens_whatever_their_sign(gen, stub_tokenizer):
    cfg = _cfg(stub_tokenizer, repetition_penalty=2.0, top_k=0, top_p=1.0, temperature=1.0, no_repeat_ngram_size=0)
    logits = torch.zeros(1, 64)
    logits[0, 5], logits[0, 6] = 2.0, -2.0
    out = gen._filter_logits(logits, torch.tensor([[5, 6]]), cfg)
    assert out[0, 5] == 1.0 and out[0, 6] == -4.0


def test_speculation_accepts_drafts_for_a_confident_model(gen, stub_tokenizer, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="text_generation")
    cfg = _cfg(stub_tokenizer, do_sample=False, use_mtp=True, mtp_speculation_mode=True, max_length=40)
    out = gen.generate(PROMPT, cfg)
    assert len(out.split()) <= 40
    assert any("MTP drafts accepted" in r.message for r in caplog.records)
