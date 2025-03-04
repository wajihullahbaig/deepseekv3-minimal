# Fix encoding issues at the very beginning of the script
import sys
import os
import io
import time
import yaml
import torch
import argparse
import logging
import json
from transformers import T5Tokenizer
from models.deepseek_v3 import DeepSeekV3
from text_generation import TextGenerator, GenerationConfig
from seeding import set_seed

# Fix for Windows console encoding issues
if sys.platform == 'win32':
    # Change console encoding to UTF-8
    os.system('chcp 65001 > NUL')
    
    # Use utf-8 encoding for stdout
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure logging with safe encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/generation_test.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TEST RUNNER")

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model_with_correct_vocab(model_path,model_yaml, device='cuda'):
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
    
    model_config = load_config(model_yaml)
    
    # Set vocab size to match checkpoint
    model_config['vocab_size'] = vocab_size
    logger.info(f"Setting model vocab_size to {vocab_size}")
    
    model = DeepSeekV3(model_config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Initialize tokenizer - make sure it matches the vocab size if possible
    tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')
    
    return model, tokenizer

def run_generation_comparison(model_path,model_yaml, device='cuda', prompt=None):
    """
    Run a side-by-side comparison of all generation methods on a single prompt.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to run generation on
        prompt: Custom prompt to use (if None, a default will be used)
    """
    # Load model and tokenizer
    model, tokenizer = load_model_with_correct_vocab(model_path,model_yaml, device)
    generator = TextGenerator(model, tokenizer, device)
    
    # # Use provided prompt or default
    # if prompt is None:
    #   prompt = "Place your prompt here either for wikipedia or youtube style comments"

    # Wiki
    prompts = [
        "April comes between March and May, making it the  ",
        "The Hubble Space Telescope is an outerspace ",
        "Artificial Intelligence makes computers ",
        "Atoms small particles, with a center made of protons and neutrons, surrounded by "
    ]    
    # Youtube 
    #prompts = [
    #    "extended arm towards it and points out the 27” screen size and suddenly you realize ",
    #    "replacing silicon as semiconductors nearly every industry can benefit from the improvements and cost reductions ",
    #    "managed to puncture my tyre the other day sealant didn’t do the trick nor did a couple of slugs"
    #]    
    
    for c,prompt in enumerate(prompts):
        logger.info(f"Running generation comparison for prompt: {prompt}")
        
        # Define configurations for different generation methods
        configs = {
            "Greedy": GenerationConfig(
                max_length=50,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            ),
            "Sampling (temp=0.7)": GenerationConfig(
                max_length=50,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            ),
            "Sampling (temp=1.2)": GenerationConfig(
                max_length=50,
                temperature=1.2,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            ),
            "Beam Search (beams=4)": GenerationConfig(
                max_length=50,
                do_sample=False,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            ),
            "MTP Speculation": GenerationConfig(
                max_length=50,
                temperature=1.2,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                use_mtp=True,
                mtp_speculation_mode=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        }
        
        # Generate using each method and print side by side
        results = {}
        generation_times = {}
        
        for name, config in configs.items():
            logger.info(f"Generating with {name}...")
            try:
                start_time = time.time()
                result = generator.generate(prompt, config)
                end_time = time.time()
                generation_time = end_time - start_time
                
                results[name] = result
                generation_times[name] = generation_time
                logger.info(f"{name} completed in {generation_time:.2f} seconds")
            except Exception as e:
                logger.error(f"{name} generation failed: {str(e)}")
                results[name] = f"Error: {str(e)}"
                generation_times[name] = None
        
        # Print detailed comparison
        logger.info("\n" + "="*80)
        logger.info(f"PROMPT: {prompt}")
        logger.info("="*80)
        
        for name, result in results.items():
            if generation_times[name] is not None:
                time_str = f"{generation_times[name]:.2f}s"
            else:
                time_str = "N/A"
            logger.info(f"\n{name} (Time: {time_str}):")
            logger.info("-"*80)
            logger.info(result)
            logger.info("-"*80)
        
        # Create directory for logs if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Save results to file in proper JSON format
        comparison_data = {
            "prompt": prompt,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": model_path,
            "device": device,
            "results": {}
        }
        
        # Add each method's results and timing
        for name, result in results.items():
            comparison_data["results"][name] = {
                "text": result,
                "generation_time_seconds": generation_times[name] if generation_times[name] is not None else None,
                "config": {k: str(v) for k, v in configs[name].__dict__.items() if not k.startswith('_')}
            }
        
        # Write to JSON file with proper encoding
        with open(f'logs/prompt_{c}_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Comparison results saved to logs/comparison.json")
    

def main():
    """Main function to parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(description='Compare different text generation methods')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model checkpoint')
    parser.add_argument('--model_yaml', type=str, required=True, 
                        help='Path to the model yaml')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt for generation')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    try:
        run_generation_comparison(args.model_path, args.model_yaml, args.device, args.prompt)
    except Exception as e:
        logger.error(f"An error occurred during text generation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()