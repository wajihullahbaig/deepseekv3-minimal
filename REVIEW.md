# Code review

Review of commit `7250964` (2026-10-08). Every row has a regression test in `tests/` or a number in [METRICS.md](METRICS.md). All rows marked fixed were fixed in the same change set.

## Why the model never learned (old log: val accuracy 0.1453 flat for 40 epochs)

| # | Where | Flaw | Status |
|---|-------|------|--------|
| 1 | `DeepSeekMoE` | Routing-bias update had the wrong sign, so overloaded experts got *more* bias; routing collapsed onto 2 of 4 experts. | Fixed: DeepSeek-V3 rule `bias -= s * sign(load - mean)` |
| 2 | `DeepSeekMoE` | Bias was a trainable parameter added to the gate weights. | Fixed: non-trainable buffer, used only to pick the top-k |
| 3 | `DeepSeekMoE` | Bias and load mutated in eval mode. | Fixed: `self.training` guard |
| 4 | chunkers | Texts shorter than `max_length` produced no data: 5.4 % of comments used. | Fixed: `data/chunking.py` keeps short texts whole |
| 5 | `get_scheduler` | Cosine schedulers sized in steps but stepped per epoch; LR never changed. | Fixed: all schedulers step per optimizer step |
| 6 | split | 75 %-overlapping windows split after chunking: 85 % of val chunks came from train comments. | Fixed: split by source text |
| 7 | `main.py` | mt5 tokenizer (250k vocab) made vocab layers 96 % of parameters and 86 % of forward time. | Fixed: small SentencePiece tokenizer trained on the corpus (`tokenizer: mt5` still available) |

## Wrong numbers

| # | Where | Flaw | Status |
|---|-------|------|--------|
| 8 | `train` | Epoch average divided by `len(train_loader)` after an early-stop break (loss 8.22 reported as 4.96). | Fixed |
| 9 | `train` | Early stopping fired on third-decimal noise. | Fixed: `early_stop_min_delta` |
| 10 | `get_optimizer` | `no_decay` never matched LayerNorm params; yaml `weight_decay` ignored. | Fixed |
| 11 | datasets | Last target of every chunk replaced by EOS mid-text. | Fixed: chunks hold `max_length + 1` ids |
| 12 | datasets | Augmentation also applied to val/test, with un-shifted targets. | Fixed: train only, re-tokenised and shifted |
| 13 | metrics | Val accuracy equalled the majority-token share (14.7 %). | Majority share is now 2.9 %, so accuracy is informative |

## Generation

| # | Where | Flaw | Status |
|---|-------|------|--------|
| 14 | decoders | Prompts ended with EOS: beam search returned the prompt unchanged. | Fixed: `add_special_tokens=False` |
| 15 | speculation | MTP drafts accepted by an 80 % coin flip, never verified. | Fixed: exact match (greedy) or `min(1, p/q)` rejection sampling |
| 16 | repetition penalty | Dividing negative logits raised their probability. | Fixed |
| 17 | `text_generation.py` | Imported `main` for one helper; per-token JSON traces; unused helpers. | Removed |
| 18 | decoders | No KV cache. | Open, by design |

## Housekeeping

| # | Where | Flaw | Status |
|---|-------|------|--------|
| 19 | trainer, generation script | Log file opened at import time inside a git-ignored folder: fresh clone crashed. | Fixed |
| 20 | `load_checkpoint` | Vocab mismatch rebuilt only the embedding. | Fixed; checkpoints embed model config and tokenizer |
| 21 | `clean_wikipedia_text` | Citation regex ran before the wiki-link regex, leaving `]`. | Fixed |
| 22 | `main.py`, generation script | Hard-coded CSV path; `--prompt` ignored. | Fixed |
| 23 | `requirements.txt` | Duplicate PyYAML; pandas and sentencepiece missing. | Fixed |
| 24 | repo | Stale plot folders, logs, caches, dead plotting helpers, stale launch configs. | Removed (`corpus/wikipedia_sample.txt` is unused but left in place) |
| 25 | model | `attn_input - mean + 1.0`, 0.1 shared-expert scale, per-head RoPE key from compressed KV. | Open: deviations from the paper, not bugs |
