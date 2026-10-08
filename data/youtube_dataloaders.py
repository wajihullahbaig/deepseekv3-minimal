import os
import pickle

import pandas as pd

from data.chunking import TextChunkDataset, collate_fn, make_loaders, split_by_source, tokenize_and_chunk

# kept for backwards compatibility with older imports
YoutubeCommentsTextDataset = TextChunkDataset
__all__ = ["YoutubeCommentsTextDataset", "collate_fn", "load_yt_texts", "preprocess_and_chunk_dataframe",
           "create_yt_and_loaders"]


def load_yt_texts(csv_path, sample_size=18000, input_column="Comment"):
    """The raw comment strings main.py also uses to train the tokenizer."""
    df = pd.read_csv(csv_path)
    return df[input_column].dropna().astype(str).tolist()[:sample_size]


def preprocess_and_chunk_dataframe(df, tokenizer, max_length, stride, min_length, input_column,
                                   sample_size=5000, cache_file=None):
    """Returns (chunks, owners) with pickle caching."""
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    texts = df[input_column].dropna().astype(str).tolist()[:sample_size]
    result = tokenize_and_chunk(texts, tokenizer, max_length, stride, min_length)
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    return result


def create_yt_and_loaders(csv_path, tokenizer, batch_size=32, min_length=5, max_length=512, stride=128,
                          device="cpu", num_workers=4, drop_last=True, use_augmentations=False,
                          sample_size=18000, cache_dir="./cache", seed=42):
    """Reads the CSV, chunks it (cached), splits by comment and returns train/val/test loaders."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"yt_chunks_v2_{sample_size}_{max_length}_{stride}_{len(tokenizer)}.pkl")
    print(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    chunks, owners = preprocess_and_chunk_dataframe(df, tokenizer, max_length, stride, min_length,
                                                    input_column="Comment", sample_size=sample_size,
                                                    cache_file=cache_file)
    splits = split_by_source(chunks, owners, seed=seed)
    return make_loaders(splits, tokenizer, max_length, batch_size, use_augmentations=use_augmentations,
                        num_workers=num_workers, drop_last=drop_last, pin_memory=(device == "cuda"))
