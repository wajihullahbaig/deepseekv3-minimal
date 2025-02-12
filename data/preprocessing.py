import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random

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

