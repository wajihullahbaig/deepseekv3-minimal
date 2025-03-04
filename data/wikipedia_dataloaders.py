import random
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from tqdm import tqdm
import os
import pickle
from functools import partial
from data.preprocessing import clean_wikipedia_text, random_deletion, random_shuffle

class WikipediaTextDataset(torch.utils.data.Dataset):
    def __init__(self, token_chunks, tokenizer, max_length, augmentation_prob=0.15, use_augmentations=False, device=None):
        self.token_chunks = token_chunks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmentation_prob = augmentation_prob
        self.device = device
        self.augmentations = [
            random_shuffle,
            random_deletion
        ]
        self.use_augmentations = use_augmentations
    
    def __len__(self):
        return len(self.token_chunks)
    
    def __getitem__(self, idx):
        # Get token chunk directly - more efficient
        original_input_ids = self.token_chunks[idx]
        
        # Only decode if augmentation is needed (improves performance)
        if random.random() < self.augmentation_prob and self.use_augmentations:
            original_text = self.tokenizer.decode(original_input_ids, skip_special_tokens=True)
            augmentation = random.choice(self.augmentations)
            augmented_text = augmentation(original_text)
        
            input_encoding = self.tokenizer(
                augmented_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
            input_ids = input_encoding["input_ids"].squeeze()
            attention_mask = input_encoding["attention_mask"].squeeze()

            output_encoding = self.tokenizer(
                original_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
            output_ids = output_encoding["input_ids"].squeeze()
        else:
            # Create padded tensor from token chunk
            input_ids = torch.tensor(original_input_ids)
            
            # Padding handling
            if len(input_ids) < self.max_length:
                padding = torch.full((self.max_length - len(input_ids),), self.tokenizer.pad_token_id, dtype=torch.long)
                input_ids = torch.cat([input_ids, padding])
            else:
                input_ids = input_ids[:self.max_length]
            
            # Create attention mask (1 for tokens, 0 for padding)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            
            # For standard next-token prediction, shift input by one position
            output_ids = input_ids.clone()
            output_ids[:-1] = input_ids[1:]
            output_ids[-1] = self.tokenizer.eos_token_id
        
        # Don't move to device here - let DataLoader handle it with pin_memory=True
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_ids": output_ids
        }

def preprocess_and_chunk_dataset(dataset, tokenizer, max_length, stride, min_length, cache_file=None):
    """Preprocess dataset with caching for faster loading"""
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
            
    tokenized_samples = []
    for item in tqdm(dataset, desc="Processing items", unit="item"):
        text = clean_wikipedia_text(item["text"])
        token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if len(token_ids) < min_length:
            continue
        for i in range(0, len(token_ids) - max_length + 1, stride):
            chunk = token_ids[i:i + max_length]
            tokenized_samples.append(chunk)
    
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(tokenized_samples, f)
    
    return tokenized_samples

def collate_fn(batch):
    """Optimized collate function that creates tensors and stacks them efficiently"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    output_ids = torch.stack([item["output_ids"] for item in batch])
    return input_ids, attention_mask, output_ids

def create_wikipedia_loaders(tokenizer, batch_size=32, min_length=20, max_length=128, stride=64, 
                           device="cuda", num_workers=4, drop_last=True, use_augmentations=False, 
                           sample_size=1000, cache_dir='./cache'):
    """Create dataloaders with improved performance"""
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/wiki_chunks_{sample_size}_{max_length}_{stride}.pkl"
    
    # Load limited dataset for faster processing
    dataset = load_dataset("wikipedia", "20220301.simple", split=f"train[:{sample_size}]")
    
    # Preprocess and chunk the dataset with caching
    token_chunks = preprocess_and_chunk_dataset(
        dataset, tokenizer, max_length, stride, min_length, cache_file
    )
    
    # Create the full dataset
    full_dataset = WikipediaTextDataset(
        token_chunks, tokenizer, max_length, use_augmentations=use_augmentations
    )
    
    # Compute split sizes
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)

    # Create subsets using random_split
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, total_size - train_size - val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True  # Speeds up host to GPU transfers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader