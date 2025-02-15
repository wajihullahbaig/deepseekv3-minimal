import yaml
from data.wikipedia_dataloaders import create_wikipedia_loaders
from data.youtube_dataloaders import create_yt_and_loaders
from models.deepseek_v3 import DeepSeekV3
from seeding import set_seed
from trainable_params import print_trainable_parameters
from training.train import train
from transformers import T5Tokenizer

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def wikipedia_main():
    config = load_config('config/base.yaml')
    set_seed(config["seed"])    
    train_config = load_config('config/train.yaml')
    model_config = load_config('config/model.yaml')

    tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')
    model_config["vocab_size"] = len(tokenizer) 
    model = DeepSeekV3(model_config)    
    print("Preparing data, please wait...")
    train_loader, val_loader, test_loader = create_wikipedia_loaders(
        tokenizer,
        batch_size=train_config['batch_size'],
        min_length = model_config["min_seq_len"],
        max_length=model_config['max_seq_len'],
        stride=model_config['stride'],
        device=config['device']
        )
    config["pad_token_id"] = tokenizer.pad_token_id
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")
    print_trainable_parameters(model, unit="M")
    model.to(config['device'])
    
    train(model, train_loader, test_loader,val_loader, {**config, **train_config})

def youtube_comments_main():
    config = load_config('config/base.yaml')
    set_seed(config["seed"])    
    train_config = load_config('config/train.yaml')
    model_config = load_config('config/model.yaml')

    tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')
    model_config["vocab_size"] = len(tokenizer) 
    model = DeepSeekV3(model_config)    
    print("Preparing data, please wait...")
    csv_path = "C:/Users/Precision/Onus/Data/YoutubeCommentsDataSet.csv"
    train_loader, val_loader, test_loader = create_yt_and_loaders(
        csv_path,
        tokenizer,
        batch_size=train_config['batch_size'],
        min_length = model_config["min_seq_len"],
        max_length=model_config['max_seq_len'],
        stride=model_config['stride'],
        device=config['device']
        )
    config["pad_token_id"] = tokenizer.pad_token_id
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")
    print_trainable_parameters(model, unit="M")
    model.to(config['device'])
    
    train(model, train_loader,test_loader, val_loader, {**config, **train_config})
if __name__ == "__main__":
    #wikipedia_main()
    youtube_comments_main()