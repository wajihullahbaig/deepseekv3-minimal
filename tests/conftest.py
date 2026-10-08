"""Shared fixtures for the test suite.

training/trainer.py opens ``logs/training.log`` at *import* time and the trainer,
text generator and data loaders write checkpoints/plots/logs/cache relative to the
working directory.  To keep the repo clean the whole session runs from a scratch
directory; repo files are addressed through REPO_ROOT.
"""
import os
import pathlib
import sys
import tempfile

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_SCRATCH = pathlib.Path(tempfile.mkdtemp(prefix="deepseekv3_tests_"))
(_SCRATCH / "logs").mkdir()
os.chdir(_SCRATCH)

YT_CSV = pathlib.Path(os.environ.get("YT_CSV", "C:/Users/Precision/Onus/Data/YoutubeCommentsDataSet.csv"))

TINY_CONFIG = dict(
    num_layers=2, hidden_dim=32, num_heads=2, head_dim=16, rope_dim=8,
    kv_compression_dim=16, query_compression_dim=16, num_experts=4,
    activated_experts=2, dropout_rate=0.0, vocab_size=64,
    min_seq_len=5, max_seq_len=32, stride=8,
)


class StubTokenizer:
    """Tiny stand-in for T5Tokenizer: pad=0, eos=1, words map deterministically to ids >= 2."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def encode(self, text, add_special_tokens=True, truncation=False, max_length=None, return_tensors=None):
        ids = [2 + sum(map(ord, w)) % (self.vocab_size - 2) for w in text.split()]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if truncation and max_length:
            ids = ids[:max_length]
        if return_tensors == "pt":
            return torch.tensor([ids])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return " ".join(f"t{i}" for i in ids if not (skip_special_tokens and i in (0, 1)))


@pytest.fixture
def tiny_config():
    return dict(TINY_CONFIG)


@pytest.fixture
def tiny_model(tiny_config):
    from models.deepseek_v3 import DeepSeekV3
    torch.manual_seed(0)
    return DeepSeekV3(tiny_config)


@pytest.fixture
def stub_tokenizer():
    return StubTokenizer(TINY_CONFIG["vocab_size"])


@pytest.fixture(scope="session")
def mt5_tokenizer():
    from transformers import T5Tokenizer
    try:
        return T5Tokenizer.from_pretrained("google/mt5-base", local_files_only=True)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"google/mt5-base is not in the local Hugging Face cache: {exc}")


def freeze_router(model):
    """Stop the MoE routing-bias side effect so two forwards on equal inputs are comparable."""
    for layer in model.layers:
        layer["moe"].bias_update_speed = 0.0
