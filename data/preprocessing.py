import random
import re


def clean_wikipedia_text(text):
    text = re.sub(r"\[\[.*?\]\]", "", text)  # wiki links first, before the single-bracket rule
    text = re.sub(r"\{\{.*?\}\}", "", text)  # templates
    text = re.sub(r"\[.*?\]", "", text)      # citations
    text = re.sub(r"\s+", " ", text)         # normalise whitespace
    return text.strip()


def random_shuffle(text):
    words = text.split()
    random.shuffle(words)
    return " ".join(words)


def random_deletion(text):
    words = text.split()
    remaining = [word for word in words if random.random() > 0.1]
    return " ".join(remaining)
