# Testing

## Environment
Use the conda env `gpt2` (`C:\Users\Precision\anaconda3\envs\gpt2`): Python 3.10, torch 2.5.1 + CUDA 11.8, and every pin in `requirements.txt`. The `deepseek` env has CPU-only torch and lacks transformers, datasets, pandas and sentencepiece; to fix it:

```
conda activate deepseek
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Unit tests
```
python -m pytest tests                 # 41 passed, ~17 s, CPU only
python -m pytest tests -m "not slow"   # skips the corpus-coverage test (needs the CSV + cached mt5 tokenizer)
```
The suite runs from a temp directory, so nothing lands in the repo.

| file | covers |
|------|--------|
| `test_model.py` | shapes, causal and padding masks, RoPE, MoE routing (top-k, gradients, bias moves against the load, eval leaves the model unchanged), backward |
| `test_data.py` | next-token shift and padding, short texts kept, chunk overlap, leak-free split by source text, augmentation only on train, wiki cleaning, corpus coverage (slow) |
| `test_tokenizer.py` | SentencePiece wrapper round trip, caching, tokenizer and model config embedded in checkpoints |
| `test_trainer.py` | accuracy, 3-epoch smoke run with checkpoints, plots and logged sentences, schedulers anneal per step, weight-decay groups, epoch average after an early stop |
| `test_generation.py` | greedy, sampling, beam search, MTP speculation (deterministic, drafts accepted), prompt without EOS, repetition penalty, filters, n-gram ban |

## Benchmarks
```
python benchmarks/bench.py --all                       # throughput, MoE balance, data coverage (~2 min)
python benchmarks/bench.py --train-epoch --generation  # one epoch + decoding timings, in benchmarks/out/
```
Results are summarised in [METRICS.md](METRICS.md).
