import yaml
from models.deepseek_v3 import DeepSeekV3
from data.dataloaders import create_datasets_and_loaders
from seeding import set_seed
from trainable_params import print_trainable_parameters
from training.train import train
from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast
from datasets import load_dataset

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config('config/base.yaml')
    set_seed(config["seed"])    
    train_config = load_config('config/train.yaml')
    model_config = load_config('config/model.yaml')

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Add special tokens to the tokenizer
    tokenizer.add_special_tokens({
        "pad_token": "<|pad|>"        
    })
    model_config["vocab_size"] = len(tokenizer) # We added a special token. So Lets update
    model = DeepSeekV3(model_config)    
    train_loader, val_loader, test_loader = create_datasets_and_loaders(tokenizer,batch_size=train_config['batch_size'],max_length=model_config['max_seq_len'],device=config['device'])

    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")
    print_trainable_parameters(model, unit="M")
    model.to(config['device'])
    
    train(model, train_loader, val_loader, {**config, **train_config},tokenizer,tokenizer.pad_token_id)

if __name__ == "__main__":
    main()