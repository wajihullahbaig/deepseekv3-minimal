"""Train the minimal DeepSeek-V3 on YouTube comments (default) or Simple-English Wikipedia.

    python main.py            # YouTube comments, settings in config/train_yt.yaml + model_yt.yaml
    python main.py wiki       # Wikipedia,       settings in config/train_wiki.yaml + model_wiki.yaml
"""
import sys

import yaml

from data.tokenizer import get_tokenizer
from data.wikipedia_dataloaders import create_wikipedia_loaders, load_wiki_texts
from data.youtube_dataloaders import create_yt_and_loaders, load_yt_texts
from models.deepseek_v3 import DeepSeekV3
from seeding import set_seed
from trainable_params import print_trainable_parameters
from training import trainer


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(dataset):
    config = load_config("config/base.yaml")
    train_config = load_config(f"config/train_{dataset}.yaml")
    model_config = load_config(f"config/model_{dataset}.yaml")
    set_seed(config["seed"])
    cache_dir = config.get("cache_dir", "./cache")

    print("Preparing data, please wait...")
    if dataset == "yt":
        texts = load_yt_texts(train_config["csv_path"], train_config["sample_size"])
        tokenizer = get_tokenizer(config, texts, cache_dir=cache_dir, name="yt")
        loaders = create_yt_and_loaders(
            train_config["csv_path"], tokenizer, batch_size=train_config["batch_size"],
            min_length=model_config["min_seq_len"], max_length=model_config["max_seq_len"],
            stride=model_config["stride"], use_augmentations=train_config["use_augmentations"],
            sample_size=train_config["sample_size"], device=config["device"],
            num_workers=train_config.get("num_workers", 0), cache_dir=cache_dir, seed=config["seed"])
    else:
        texts = load_wiki_texts(train_config["sample_size"])
        tokenizer = get_tokenizer(config, texts, cache_dir=cache_dir, name="wiki")
        loaders = create_wikipedia_loaders(
            tokenizer, batch_size=train_config["batch_size"], min_length=model_config["min_seq_len"],
            max_length=model_config["max_seq_len"], stride=model_config["stride"],
            use_augmentations=train_config["use_augmentations"], sample_size=train_config["sample_size"],
            device=config["device"], num_workers=train_config.get("num_workers", 0), cache_dir=cache_dir,
            seed=config["seed"])
    train_loader, val_loader, test_loader = loaders

    model_config["vocab_size"] = len(tokenizer)
    config["pad_token_id"] = tokenizer.pad_token_id
    model = DeepSeekV3(model_config)
    print(f"Tokenizer: {type(tokenizer).__name__} with {len(tokenizer)} pieces")
    print(f"Batches - train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")
    print_trainable_parameters(model, unit="M")

    trainer.train(model, train_loader, test_loader, val_loader, {**config, **train_config}, tokenizer=tokenizer)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "yt")
