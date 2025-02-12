import random
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from data.preprocessing import random_deletion, random_shuffle, split_text_into_chunks



class YoutubeCommentsTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, device='cpu'):
        """
        Initializes the TextDataset with texts, tokenizer, maximum sequence length, and device.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.texts = texts
        self.augmentations = [
            random_shuffle,
            random_deletion
        ]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        original_text = self.texts[idx]
                        
        for augmentation in self.augmentations:
            if random.random() < 0.5:
                augmented_text = original_text
                augmented_text = augmentation(augmented_text)
        
                encoding = self.tokenizer(
                    augmented_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].squeeze().to(self.device)
                attention_mask = encoding['attention_mask'].squeeze().to(self.device)
        
                encoding = self.tokenizer(
                    original_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors='pt'
                )
                output_ids = encoding['input_ids'].squeeze().to(self.device)
                return input_ids, attention_mask, output_ids
            else:
                encoding = self.tokenizer(
                    original_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].squeeze().to(self.device)
                attention_mask = encoding['attention_mask'].squeeze().to(self.device)
                output_ids = input_ids.clone()
                output_ids[:-1] = input_ids[1:]  # Shift left
                output_ids[-1] = self.tokenizer.eos_token_id  # Set EOS                
                return input_ids, attention_mask, output_ids

def preprocess_and_chunk_dataset(df, min_length, max_length, tokenizer,input_colum):
    cleaned_chunks = []
    dataset = df[input_colum].dropna().tolist()
    for text in tqdm(dataset, desc="Processing items", unit="item"):
            
        if len(text.split()) < min_length:
            continue
        
        sentences = text.split(".")
        chunks = split_text_into_chunks(
            sentences,
            max_tokens=max_length,
            tokenizer=tokenizer
        )
        cleaned_chunks.extend(chunks)
    
    return cleaned_chunks

def yt_create_datasets_and_loaders(csv_path, tokenizer, batch_size=32, min_length=5, max_length=512, device='cpu'):
    """
    Reads the CSV file, preprocesses the data, creates datasets, and returns data loaders.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Preprocess and chunk the dataset
    cleaned_chunks = preprocess_and_chunk_dataset(df, min_length, max_length, tokenizer,'Comment')
    
    # Create the TextDataset
    full_dataset = YoutubeCommentsTextDataset(
        cleaned_chunks,
        tokenizer,
        max_length=max_length,
        device=device
    )
    
    # Split the dataset into train, validation, and test sets
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

