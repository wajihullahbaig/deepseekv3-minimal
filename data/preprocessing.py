import re
import random



def clean_wikipedia_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # Remove citations
    text = re.sub(r'\[\[.*?\]\]', '', text)  # Remove wiki links
    text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove templates
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

def random_shuffle(text):
    words = text.split()
    random.shuffle(words)
    return " ".join(words)   

def random_deletion(text):
    words = text.split()
    remaining = [word for word in words if random.random() > 0.1]
    return " ".join(remaining)