import os
import pickle

from datasets import load_dataset

from data.chunking import TextChunkDataset, collate_fn, make_loaders, split_by_source, tokenize_and_chunk
from data.preprocessing import clean_wikipedia_text

# kept for backwards compatibility with older imports
WikipediaTextDataset = TextChunkDataset
__all__ = ["WikipediaTextDataset", "collate_fn", "load_wiki_texts", "preprocess_and_chunk_dataset",
           "create_wikipedia_loaders"]


def load_wiki_texts(sample_size=1000):
    """Cleaned article texts; main.py also uses them to train the tokenizer."""
    dataset = load_dataset("wikipedia", "20220301.simple", split=f"train[:{sample_size}]")
    return [clean_wikipedia_text(item["text"]) for item in dataset]


def preprocess_and_chunk_dataset(dataset, tokenizer, max_length, stride, min_length, cache_file=None):
    """``dataset`` is an iterable of {"text": ...}; returns (chunks, owners) with pickle caching."""
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    texts = [clean_wikipedia_text(item["text"]) for item in dataset]
    result = tokenize_and_chunk(texts, tokenizer, max_length, stride, min_length)
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    return result


def create_wikipedia_loaders(tokenizer, batch_size=32, min_length=20, max_length=128, stride=64,
                             device="cuda", num_workers=4, drop_last=True, use_augmentations=False,
                             sample_size=1000, cache_dir="./cache", seed=42):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"wiki_chunks_v2_{sample_size}_{max_length}_{stride}_{len(tokenizer)}.pkl")
    dataset = load_dataset("wikipedia", "20220301.simple", split=f"train[:{sample_size}]")
    chunks, owners = preprocess_and_chunk_dataset(dataset, tokenizer, max_length, stride, min_length, cache_file)
    splits = split_by_source(chunks, owners, seed=seed)
    return make_loaders(splits, tokenizer, max_length, batch_size, use_augmentations=use_augmentations,
                        num_workers=num_workers, drop_last=drop_last, pin_memory=(device == "cuda"))
