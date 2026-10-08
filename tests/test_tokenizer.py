import pickle

import torch

from data.tokenizer import SentencePieceTokenizer, get_tokenizer
from models.deepseek_v3 import DeepSeekV3
from training.checkpointing import load_tokenizer_from_checkpoint, save_checkpoint

TEXTS = [f"the quick brown fox jumps over the lazy dog number {i}" for i in range(200)] + ["hello world"] * 50


def test_sentencepiece_tokenizer_roundtrip(tmp_path):
    tok = SentencePieceTokenizer.train(TEXTS, vocab_size=100, model_prefix=str(tmp_path / "spm"))
    assert tok.pad_token_id == 0 and tok.eos_token_id == 1 and 10 < len(tok) <= 100
    ids = tok.encode("the quick brown fox")
    assert ids[-1] == tok.eos_token_id and tok.decode(ids) == "the quick brown fox"
    assert tok.encode("hello", return_tensors="pt").shape[0] == 1
    assert len(tok.encode("the quick brown fox", add_special_tokens=False, truncation=True, max_length=2)) == 2
    clone = pickle.loads(pickle.dumps(tok))
    assert clone.encode("hello world") == tok.encode("hello world")
    assert tok.convert_ids_to_tokens(ids[0]).startswith("▁")


def test_get_tokenizer_trains_once_then_loads_from_cache(tmp_path):
    cfg = {"tokenizer": "sentencepiece", "tokenizer_vocab_size": 100}
    first = get_tokenizer(cfg, TEXTS, cache_dir=str(tmp_path), name="t")
    assert (tmp_path / "spm_t_100.model").exists()
    second = get_tokenizer(cfg, None, cache_dir=str(tmp_path), name="t")      # no texts needed any more
    assert second.encode("the lazy dog") == first.encode("the lazy dog")


def test_checkpoint_embeds_the_tokenizer(tiny_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tok = SentencePieceTokenizer.train(TEXTS, vocab_size=100, model_prefix=str(tmp_path / "spm"))
    tiny_config["vocab_size"] = len(tok)
    model = DeepSeekV3(tiny_config)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(model, opt, epoch=0, batch_id=0, tokenizer=tok, is_best=True)
    ckpt = torch.load(tmp_path / "checkpoints" / "best_model.pt", map_location="cpu", weights_only=False)
    assert ckpt["vocab_size"] == len(tok) and ckpt["model_config"]["hidden_dim"] == tiny_config["hidden_dim"]
    loaded = load_tokenizer_from_checkpoint(ckpt)
    assert loaded.encode("the quick") == tok.encode("the quick")
