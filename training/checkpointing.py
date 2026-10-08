import json
import logging
import os
import pickle
import shutil
from datetime import datetime

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, batch_id, metrics=None, is_best=False, tokenizer=None):
    """Save model/optimizer state plus metadata. A SentencePiece tokenizer is embedded in the
    checkpoint (``tokenizer_model`` bytes); any other tokenizer is pickled next to it."""
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "batch_id": batch_id,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vocab_size": model.config["vocab_size"],
        "model_config": dict(model.config),
    }
    if metrics:
        checkpoint["metrics"] = metrics
    tokenizer_filename = None
    if tokenizer is not None:
        if hasattr(tokenizer, "serialized_model_proto"):
            checkpoint["tokenizer_model"] = tokenizer.serialized_model_proto()
        else:
            tokenizer_filename = f"checkpoints/tokenizer_epoch_{epoch}_{batch_id}.pkl"
            with open(tokenizer_filename, "wb") as f:
                pickle.dump(tokenizer, f)

    checkpoint_filename = f"checkpoints/checkpoint_epoch_{epoch}_{batch_id}.pt"
    torch.save(checkpoint, checkpoint_filename)
    logger.info(f"Saved checkpoint: {checkpoint_filename}")

    metadata = {k: checkpoint[k] for k in ("epoch", "batch_id", "timestamp", "vocab_size")}
    if metrics:
        metadata["metrics"] = metrics
    with open(f"checkpoints/checkpoint_epoch_{epoch}_{batch_id}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if is_best:
        shutil.copyfile(checkpoint_filename, "checkpoints/best_model.pt")
        if tokenizer_filename:
            shutil.copyfile(tokenizer_filename, "checkpoints/best_model_tokenizer.pkl")
        with open("checkpoints/best_model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved as best model: checkpoints/best_model.pt")


def load_tokenizer_from_checkpoint(checkpoint, checkpoint_path=None):
    """Tokenizer embedded in / saved next to a checkpoint, else the mt5 default."""
    if "tokenizer_model" in checkpoint:
        from data.tokenizer import SentencePieceTokenizer
        return SentencePieceTokenizer(checkpoint["tokenizer_model"])
    if checkpoint_path:
        base = os.path.basename(checkpoint_path)
        candidates = [os.path.join(os.path.dirname(checkpoint_path), base.replace(".pt", "_tokenizer.pkl")),
                      os.path.join(os.path.dirname(checkpoint_path),
                                   "tokenizer_" + base.replace("checkpoint_", "").replace(".pt", ".pkl"))]
        for path in candidates:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return pickle.load(f)
    logger.warning("No tokenizer stored with the checkpoint, using google/mt5-base")
    from transformers import T5Tokenizer
    return T5Tokenizer.from_pretrained("google/mt5-base")


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cuda", load_tokenizer=True):
    """Load model (and optimizer) state. On a vocab mismatch the three vocab-sized layers are rebuilt."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=False)

    vocab_size = checkpoint.get("vocab_size")
    if vocab_size and vocab_size != model.config["vocab_size"]:
        logger.warning(f"Vocabulary size mismatch: checkpoint has {vocab_size}, model has "
                       f"{model.config['vocab_size']}. Rebuilding vocab-sized layers.")
        hidden = model.config["hidden_dim"]
        model.config["vocab_size"] = vocab_size
        model.embedding = torch.nn.Embedding(vocab_size, hidden)
        model.output_head = torch.nn.Linear(hidden, vocab_size)
        model.mtp.output_head = torch.nn.Linear(hidden, vocab_size)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    tokenizer = load_tokenizer_from_checkpoint(checkpoint, checkpoint_path) if load_tokenizer else None
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    return checkpoint, tokenizer
