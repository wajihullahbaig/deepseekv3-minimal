from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import re

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024, device='cpu', pad_token_id=None, eos_token_id=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.texts = texts  # texts are already preprocessed and chunked

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

def clean_wikipedia_text(text):
    # Remove markup (simplified example)
    text = re.sub(r'\[.*?\]', '', text)  # Remove citations like [1], [2], etc.
    text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove templates like {{cite}}
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def split_text_into_chunks(text, max_tokens, tokenizer):
    # First, split the text into sentences or paragraphs
    sentences = text.split('.')
    
    chunks = []
    current_chunk = ""
    current_token_count = 0

    for sentence in sentences:
        # Add the period back (except for the last sentence if it didn't end with a period)
        sentence = sentence.strip() + ('.' if sentence else '')
        
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=True)
        sentence_token_count = len(sentence_tokens)
    
        # If adding this sentence would exceed the limit, start a new chunk
        if current_token_count + sentence_token_count > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_token_count = sentence_token_count
        else:
            current_chunk += ' ' + sentence
            current_token_count += sentence_token_count

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def preprocess_and_chunk_dataset(dataset, min_length, max_length, tokenizer):
    cleaned_chunks = []
    
    for item in tqdm(dataset, desc="Processing items", unit="item"):
        text = item['text']
        
        # Clean the text
        cleaned_text = clean_wikipedia_text(text)
        
        # Filter low-quality text
        if len(cleaned_text) >= min_length:
            # Split the text into chunks
            chunks = split_text_into_chunks(cleaned_text, max_length, tokenizer)
            cleaned_chunks.extend(chunks)
    
    return cleaned_chunks


def create_datasets_and_loaders(tokenizer, batch_size=32,min_length=5, max_length=4096, device='cpu'):
    # Download Wikipedia dataset
    dataset = load_dataset("wikipedia", "20220301.simple", split="train")
    
    # Preprocess and filter the dataset
    cleaned_chunks = preprocess_and_chunk_dataset(dataset, min_length=min_length, max_length = max_length,tokenizer=tokenizer)
    
    # Create a single dataset
    full_dataset = TextDataset(
        cleaned_chunks,
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
