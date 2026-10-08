"""Tokenise -> chunk -> split -> Dataset pipeline shared by both corpora.

Every chunk holds ``max_length + 1`` token ids so that the training pair is a true
shift (input = chunk[:-1], target = chunk[1:]).  Texts shorter than that are kept
whole and padded; the split is done on *source texts* so overlapping windows of one
text never end up on both sides of the train / validation boundary.
"""
import random

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.preprocessing import random_deletion, random_shuffle


def chunk_token_ids(token_ids, max_length, stride, min_length):
    """Windows of ``max_length + 1`` ids; short texts are returned whole; texts with
    fewer than ``min_length`` ids (or fewer than 2) are dropped."""
    n = len(token_ids)
    if n < max(min_length, 2):
        return []
    window = max_length + 1
    if n <= window:
        return [list(token_ids)]
    return [list(token_ids[i:i + window]) for i in range(0, n - window + 1, stride)]


def tokenize_and_chunk(texts, tokenizer, max_length, stride, min_length, desc="Processing items"):
    """Returns (chunks, owners): owners[i] is the index of the source text of chunks[i]."""
    chunks, owners = [], []
    for ci, text in enumerate(tqdm(texts, desc=desc, unit="item")):
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        for c in chunk_token_ids(ids, max_length, stride, min_length):
            chunks.append(c)
            owners.append(ci)
    return chunks, owners


def split_by_source(chunks, owners, fractions=(0.7, 0.15, 0.15), seed=42):
    """Train / val / test split on source texts, so no text contributes to two splits."""
    sources = sorted(set(owners))
    g = torch.Generator().manual_seed(seed)
    perm = [sources[i] for i in torch.randperm(len(sources), generator=g).tolist()]
    n_train = int(fractions[0] * len(perm))
    n_val = int(fractions[1] * len(perm))
    bucket = {}
    for s in perm[:n_train]:
        bucket[s] = 0
    for s in perm[n_train:n_train + n_val]:
        bucket[s] = 1
    for s in perm[n_train + n_val:]:
        bucket[s] = 2
    out = ([], [], [])
    for c, o in zip(chunks, owners):
        out[bucket[o]].append(c)
    return out


class TextChunkDataset(Dataset):
    """Next-token-prediction samples: input_ids, attention_mask, output_ids of length max_length."""

    def __init__(self, chunks, tokenizer, max_length, augmentation_prob=0.15, use_augmentations=False):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmentation_prob = augmentation_prob
        self.use_augmentations = use_augmentations
        self.augmentations = [random_shuffle, random_deletion]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = self.chunks[idx]
        if self.use_augmentations and random.random() < self.augmentation_prob:
            text = random.choice(self.augmentations)(self.tokenizer.decode(ids, skip_special_tokens=True))
            aug = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=self.max_length + 1)
            if len(aug) >= 2:
                ids = aug
        ids = torch.tensor(ids[: self.max_length + 1], dtype=torch.long)
        inp, tgt = ids[:-1], ids[1:]
        n = inp.numel()
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((self.max_length,), pad, dtype=torch.long)
        output_ids = torch.full((self.max_length,), pad, dtype=torch.long)   # pad targets are ignored by the loss
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        input_ids[:n] = inp
        output_ids[:n] = tgt
        attention_mask[:n] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "output_ids": output_ids}


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    output_ids = torch.stack([item["output_ids"] for item in batch])
    return input_ids, attention_mask, output_ids


def make_loaders(splits, tokenizer, max_length, batch_size, use_augmentations=False,
                 num_workers=4, drop_last=True, pin_memory=False):
    """splits = (train_chunks, val_chunks, test_chunks); augmentation is applied to train only."""
    loaders = []
    for i, chunks in enumerate(splits):
        ds = TextChunkDataset(chunks, tokenizer, max_length, use_augmentations=use_augmentations and i == 0)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=(i == 0), collate_fn=collate_fn,
                                  num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory))
    return tuple(loaders)
