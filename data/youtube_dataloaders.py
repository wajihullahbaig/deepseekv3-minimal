import random
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
import torch
from data.preprocessing import random_deletion, random_shuffle


class YoutubeCommentsTextDataset(Dataset):
    def __init__(self, token_chunks, tokenizer, max_length, augmentation_prob=0.5,use_augmentations=True, device=None):
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
        original_input_ids = self.token_chunks[idx]
        original_text = self.tokenizer.decode(original_input_ids, skip_special_tokens=True)
        
        augmented_text = original_text
        if random.random() < self.augmentation_prob and self.use_augmentations:
            augmentation = random.choice(self.augmentations)
            augmented_text = augmentation(augmented_text)
        
            input_encoding = self.tokenizer(
                augmented_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
            input_ids = input_encoding["input_ids"].squeeze().to(self.device)
            attention_mask = input_encoding["attention_mask"].squeeze().to(self.device)            

            output_encoding = self.tokenizer(
                original_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt"
            )
            output_ids = output_encoding["input_ids"].squeeze().to(self.device)            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_ids": output_ids
            }        
        else:
            input_encoding = self.tokenizer(
            original_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
             )
            input_ids = input_encoding["input_ids"].squeeze().to(self.device)
            attention_mask = input_encoding["attention_mask"].squeeze().to(self.device)            
            output_ids = input_ids.clone()
            output_ids[:-1] = input_ids[1:]
            output_ids[-1] = self.tokenizer.eos_token_id                            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "output_ids": output_ids
            }


def preprocess_and_chunk_dataframe(df, tokenizer, max_length, stride, min_length, input_column):
    tokenized_samples = []
    dataset = df[input_column].dropna().tolist()[:5000]    
    for text in tqdm(dataset, desc="Processing items", unit="item"):
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

def create_yt_and_loaders(csv_path, tokenizer, batch_size=32, min_length=5, max_length=512, stride = 128,device='cpu',num_workers=0,drop_last=True,use_augmentations=True):
    """
    Reads the CSV file, preprocesses the data, creates datasets, and returns data loaders.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    token_chunks = preprocess_and_chunk_dataframe(df, tokenizer, max_length, stride, min_length, input_column="Comment")
    
    full_dataset = YoutubeCommentsTextDataset(token_chunks, tokenizer, max_length,use_augmentations, device=device)
    
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
