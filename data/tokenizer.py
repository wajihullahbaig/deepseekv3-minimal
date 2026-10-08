"""Tokenizer choice: Google's mt5 SentencePiece model (250k vocab, vocab layers dominate the
model) or a small SentencePiece model trained on the training corpus itself.

The small tokenizer keeps mt5's special-id layout (pad=0, eos=1) so the rest of the code
does not care which one is in use.
"""
import os

import torch


class SentencePieceTokenizer:
    """Minimal T5Tokenizer-compatible wrapper around a SentencePiece model (pad=0, eos=1, unk=2)."""

    def __init__(self, model_proto: bytes):
        import sentencepiece as spm
        self.model_proto = model_proto
        self.sp = spm.SentencePieceProcessor(model_proto=model_proto)
        self.pad_token_id = self.sp.pad_id()
        self.eos_token_id = self.sp.eos_id()
        self.unk_token_id = self.sp.unk_id()

    # -- construction -------------------------------------------------------------
    @classmethod
    def train(cls, texts, vocab_size, model_prefix, model_type="unigram"):
        import sentencepiece as spm
        os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)
        corpus = f"{model_prefix}.txt"
        with open(corpus, "w", encoding="utf-8") as f:
            for t in texts:
                t = " ".join(str(t).split())
                if t:
                    f.write(t + "\n")
        spm.SentencePieceTrainer.train(
            input=corpus, model_prefix=model_prefix, vocab_size=vocab_size, model_type=model_type,
            pad_id=0, eos_id=1, unk_id=2, bos_id=-1, character_coverage=0.9995, hard_vocab_limit=False,
            shuffle_input_sentence=True, max_sentence_length=16384, minloglevel=1,
        )
        return cls.load(f"{model_prefix}.model")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(f.read())

    def save(self, path):
        with open(path, "wb") as f:
            f.write(self.model_proto)

    def serialized_model_proto(self):
        return self.model_proto

    def __getstate__(self):
        return {"model_proto": self.model_proto}

    def __setstate__(self, state):
        self.__init__(state["model_proto"])

    # -- T5Tokenizer-like API -------------------------------------------------------
    def __len__(self):
        return self.sp.get_piece_size()

    def encode(self, text, add_special_tokens=True, truncation=False, max_length=None, return_tensors=None):
        ids = self.sp.encode(str(text), out_type=int)
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        if skip_special_tokens:
            ids = [int(i) for i in ids if i not in (self.pad_token_id, self.eos_token_id)]
        return self.sp.decode([int(i) for i in ids])

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.sp.id_to_piece(ids)
        return [self.sp.id_to_piece(int(i)) for i in ids]


def get_tokenizer(config, texts=None, cache_dir="./cache", name="corpus"):
    """``config["tokenizer"]`` is "mt5" or "sentencepiece" (trained once on ``texts`` and cached)."""
    kind = str(config.get("tokenizer", "mt5")).lower()
    if kind == "mt5":
        from transformers import T5Tokenizer
        return T5Tokenizer.from_pretrained("google/mt5-base")
    if kind == "sentencepiece":
        vocab_size = int(config.get("tokenizer_vocab_size", 8000))
        prefix = os.path.join(cache_dir, f"spm_{name}_{vocab_size}")
        if os.path.exists(prefix + ".model"):
            return SentencePieceTokenizer.load(prefix + ".model")
        if texts is None:
            raise ValueError(f"No trained tokenizer at {prefix}.model and no texts to train one on")
        print(f"Training a {vocab_size}-piece SentencePiece tokenizer on {len(texts)} texts...")
        return SentencePieceTokenizer.train(texts, vocab_size, prefix)
    raise ValueError(f"Unknown tokenizer kind: {kind}")
