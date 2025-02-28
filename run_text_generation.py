# Fix encoding issues at the very beginning of the script
import sys
import os
import io

# Fix for Windows console encoding issues
if sys.platform == 'win32':
    # Change console encoding to UTF-8
    os.system('chcp 65001 > NUL')
    
    # Use utf-8 encoding for stdout
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import yaml
import torch
import argparse
import logging
import os
import pickle
from transformers import T5Tokenizer
from models.deepseek_v3 import DeepSeekV3
from text_generation import TextGenerator, GenerationConfig
from seeding import set_seed

# Configure logging with safe encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/generation_test.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model_with_correct_vocab(model_path, device='cuda'):
    """Load the model with the correct vocabulary size from the checkpoint."""
    # Load the checkpoint to get embedding size
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Check if the checkpoint has model_state_dict key
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Get vocab size from embedding weight
    vocab_size = state_dict['embedding.weight'].shape[0]
    logger.info(f"Checkpoint vocabulary size: {vocab_size}")
    
    model_config = load_config('config/model.yaml')
    
    # Set vocab size to match checkpoint
    model_config['vocab_size'] = vocab_size
    logger.info(f"Setting model vocab_size to {vocab_size}")
    
    model = DeepSeekV3(model_config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    # Initialize tokenizer - make sure it matches the vocab size if possible
    tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')
    
    return model, tokenizer

def test_model_generation(model_path, device='cuda'):
    """Test model generation with different strategies."""
    # Load model and tokenizer with correct vocabulary handling
    model, tokenizer = load_model_with_correct_vocab(model_path, device)
    generator = TextGenerator(model, tokenizer, device)
    
    os.makedirs('logs', exist_ok=True)
    
    prompts = [
        "The artificial intelligence revolution is changing how we",
        "Climate change has become one of the most pressing issues because",
        "The future of renewable energy depends on",
        "The number of people living in cities has",
        "Learning a new language requires",
        "The solution to the equation x squared plus 7x plus 12 equals 0 is"
    ]
    
    # Hold all results for comparison
    all_results = {}
    
    # Try different generation methods
    for i, prompt in enumerate(prompts):
        logger.info(f"\nPrompt {i+1}: {prompt}")
        prompt_results = {}
        
        # Standard greedy generation (deterministic)
        config = GenerationConfig(
            max_length=100, 
            temperature=1.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        try:
            result = generator.generate(prompt, config)
            # Handle potential Unicode issues in output
            safe_result = result.encode('ascii', 'ignore').decode('utf-8')
            logger.info(f"\nGreedy decoding:\n{safe_result}")
            prompt_results['greedy'] = safe_result
        except Exception as e:
            logger.error(f"Greedy generation failed: {str(e)}")
            prompt_results['greedy'] = f"Error: {str(e)}"
        
        # Standard sampling
        config = GenerationConfig(
            max_length=100, 
            temperature=1.2,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        try:
            result = generator.generate(prompt, config)
            # Handle potential Unicode issues in output
            safe_result = result.encode('ascii', 'ignore').decode('utf-8')
            logger.info(f"\nStandard sampling:\n{safe_result}")
            prompt_results['sampling'] = safe_result
        except Exception as e:
            logger.error(f"Sampling generation failed: {str(e)}")
            prompt_results['sampling'] = f"Error: {str(e)}"
        
        all_results[prompt] = prompt_results
    
    # Save all results to file with safe encoding
    import json
    with open('logs/test_results.json', 'w', encoding='utf-8') as f:
        # Convert results to strings for JSON serialization
        serializable_results = {str(k): {str(k2): str(v2) for k2, v2 in v.items()} 
                              for k, v in all_results.items()}
        json.dump(serializable_results, f, indent=2, ensure_ascii=True)
    
    logger.info(f"Generation results saved to logs/test_results.json")
    return all_results

def main():
    """Main function to parse arguments and run the test."""
    parser = argparse.ArgumentParser(description='Test DeepSeek model text generation')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    try:
        test_model_generation(args.model_path, args.device)
    except Exception as e:
        logger.error(f"An error occurred during text generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()