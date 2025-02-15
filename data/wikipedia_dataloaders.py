import random
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch
from data.preprocessing import clean_wikipedia_text

class WikipediaTextDataset(torch.utils.data.Dataset):
    def __init__(self, token_chunks, tokenizer, max_length, augmentation_prob=0.5, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.token_chunks = token_chunks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmentation_prob = augmentation_prob
        self.device = device
        self.augmentations = [
            self.random_shuffle,
            self.random_deletion
        ]
    
    def __len__(self):
        return len(self.token_chunks)
    
    def __getitem__(self, idx):
        original_input_ids = self.token_chunks[idx]
        input_text = self.tokenizer.decode(original_input_ids, skip_special_tokens=True)
        
        augmented_text = input_text
        if random.random() < self.augmentation_prob:
            augmentation = random.choice(self.augmentations)
            augmented_text = augmentation(augmented_text)
        
        encoding = self.tokenizer(
            augmented_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze().to(self.device)
        attention_mask = encoding["attention_mask"].squeeze().to(self.device)
        
        output_ids = input_ids.clone()
        output_ids[:-1] = input_ids[1:]
        output_ids[-1] = self.tokenizer.eos_token_id
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_ids": output_ids
        }

    @staticmethod
    def random_shuffle(text):
        words = text.split()
        random.shuffle(words)
        return " ".join(words)
    
    @staticmethod
    def random_deletion(text):
        words = text.split()
        remaining = [word for word in words if random.random() > 0.1]
        return " ".join(remaining)

def preprocess_and_chunk_dataset(dataset, tokenizer, max_length, stride, min_length):
    tokenized_samples = []
    for item in tqdm(dataset, desc="Processing items", unit="item"):
        text = clean_wikipedia_text(item["text"])
        token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if len(token_ids) < min_length:
            continue
        for i in range(0, len(token_ids) - max_length + 1, stride):
            chunk = token_ids[i:i + max_length]
            tokenized_samples.append(chunk)
    return tokenized_samples

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    output_ids = torch.stack([item["output_ids"] for item in batch])
    return input_ids,attention_mask,output_ids
    

from torch.utils.data import Dataset, DataLoader, random_split

def create_wikipedia_loaders(tokenizer, batch_size=32, min_length=20, max_length=128, stride=64, device="cuda", num_workers=1, drop_last=True):
    dataset = load_dataset("wikipedia", "20220301.simple", split="train[:2000]")
    token_chunks = preprocess_and_chunk_dataset(dataset, tokenizer, max_length, stride, min_length)
    
    full_dataset = WikipediaTextDataset(token_chunks, tokenizer, max_length, device=device)
    
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last
    )
    
    return train_loader, val_loader, test_loader
