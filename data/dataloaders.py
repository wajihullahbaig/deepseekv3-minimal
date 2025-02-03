from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from datasets import load_dataset
import os

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=4096, device='cpu', pad_token_id=None, eos_token_id=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        self.device = device
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize the text and add special tokens
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True  # Add EOS token automatically
        )
        input_ids = encoding['input_ids'].squeeze().to(self.device)
        attention_mask = encoding['attention_mask'].squeeze().to(self.device)
        
        # Shift input_ids to create target labels for next-token prediction
        target_ids = input_ids.clone()
        target_ids[:-1] = input_ids[1:]  # Shift tokens to the left
        target_ids[-1] = self.eos_token_id  # Set the last token to EOS token
        
        return input_ids, attention_mask, target_ids

def create_datasets_and_loaders(tokenizer, batch_size=32, max_length=4096, device='cpu'):
    # Download Wikipedia dataset
    dataset = load_dataset("wikipedia", "20220301.simple", split="train")
    
    # Save the dataset to a local file
    os.makedirs("corpus", exist_ok=True)
    with open("corpus/wikipedia_sample.txt", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(item['text'] + "\n")
    
    # Create a single dataset
    full_dataset = TextDataset(
        dataset['text'],
        tokenizer,
        max_length,
        device,
        pad_token_id=tokenizer.pad_token_id,  # Pass pad_token_id
        eos_token_id=tokenizer.eos_token_id   # Pass eos_token_id
    )
    
    # Split the dataset
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

