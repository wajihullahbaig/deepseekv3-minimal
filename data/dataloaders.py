import random
import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

nltk.download('wordnet')
nltk.download('punkt')

def clean_wikipedia_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove citations
    text = re.sub(r'\[\[.*?\]\]', '', text)  # Remove wiki links
    text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove templates
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word:
                synonyms.append(synonym)
    return list(set(synonyms))

def synonym_replacement(text, aug_prob=0.1):
    words = word_tokenize(text)
    augmented_words = []
    for word in words:
        if random.random() < aug_prob:
            synonyms = get_synonyms(word)
            if synonyms:
                augmented_words.append(random.choice(synonyms))
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    return TreebankWordDetokenizer().detokenize(augmented_words)

def random_shuffle(text, p_shuffle=0.5):
    sentences = sent_tokenize(text)
    if len(sentences) > 1 and random.random() < p_shuffle:
        random.shuffle(sentences)
    return ' '.join(sentences)

def random_deletion(text, p_deletion=0.1):
    words = word_tokenize(text)
    if len(words) == 1:
        return text
    new_words = []
    for word in words:
        if random.random() > p_deletion:
            new_words.append(word)
    if not new_words:
        return random.choice(words)
    return TreebankWordDetokenizer().detokenize(new_words)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024, device='cpu'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.texts = texts
        self.augmentations = [
            synonym_replacement,
            random_shuffle,
            random_deletion
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        original_text = self.texts[idx]
        
        augmented_text = original_text
        for augmentation in self.augmentations:
            if random.random() < 0.5:
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
        
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift left
        labels[-1] = self.tokenizer.eos_token_id  # Set EOS
        
        return input_ids, attention_mask, labels

def split_text_into_chunks(sentences, max_tokens, tokenizer):
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        tokenized_sentence = tokenizer(sentence, add_special_tokens=False)
        num_tokens = len(tokenized_sentence.input_ids)
        
        if current_tokens + num_tokens > max_tokens and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_tokens = 0
        
        current_chunk.append(sentence)
        current_tokens += num_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def preprocess_and_chunk_dataset(dataset, min_length, max_length, tokenizer):
    cleaned_chunks = []
    
    for item in tqdm(dataset, desc="Processing items", unit="item"):
        text = item['text']
        cleaned_text = clean_wikipedia_text(text)
        
        if len(cleaned_text.split()) < min_length:
            continue
        
        sentences = sent_tokenize(cleaned_text)
        chunks = split_text_into_chunks(
            sentences,
            max_tokens=tokenizer.model_max_length,
            tokenizer=tokenizer
        )
        cleaned_chunks.extend(chunks)
    
    return cleaned_chunks

def create_datasets_and_loaders(tokenizer, batch_size=32, min_length=5, max_length=512, device='cpu'):
    dataset = load_dataset("wikipedia", "20220301.simple", split="train[:5000]")
    cleaned_chunks = preprocess_and_chunk_dataset(dataset, min_length, max_length, tokenizer)
    
    full_dataset = TextDataset(
        cleaned_chunks,
        tokenizer,
        max_length=max_length,
        device=device
    )
    
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

