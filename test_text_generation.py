import torch
from models.deepseek_v3 import DeepSeekV3
from typing import Dict, List, Optional, Union
import yaml
from transformers import GPT2TokenizerFast
from seeding import set_seed
from trainable_params import print_trainable_parameters
import torch
from models.deepseek_v3 import DeepSeekV3
from typing import Dict, List, Optional, Union
import yaml
from torch.nn import functional as F

import torch
from torch.nn import functional as F

class GenerationConfig:
    def __init__(self, max_length=128, temperature=1.0, top_k=50, top_p=0.9,
                 repetition_penalty=1.2, eos_token_id=None, pad_token_id=None):
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

class TextGenerator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
    def _create_causal_mask(self, bsz: int, seq_len: int) -> torch.Tensor:
        """Create a causal mask to prevent attending to future tokens."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)
        return mask
    
    def generate(self, prompts: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
        
        generated_texts = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            generated = input_ids.clone()
            
            # Early stopping variables
            current_length = generated.size(1)
            eos_reached = False
            
            while current_length < config.max_length and not eos_reached:
                # Create causal mask for current sequence
                causal_mask = self._create_causal_mask(generated.size(0), current_length)
                
                # Model forward pass with MTP
                outputs = self.model(generated, attention_mask=causal_mask)
                
                # Use the main model output for next token prediction
                logits = outputs[:, -1, :] / config.temperature
                
                # Apply repetition penalty
                for i in range(logits.size(0)):
                    for previous_token in generated[i].unique():
                        logits[i, previous_token] /= config.repetition_penalty

                # Top-k and top-p filtering
                filtered_logits = F.softmax(logits, dim=-1)
                if config.top_k is not None:
                    top_k_values, _ = torch.topk(filtered_logits, k=config.top_k)
                    min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                    filtered_logits = filtered_logits.scatter_(-1, (filtered_logits < min_top_k_value).type(torch.int64), 0.0)
                
                if config.top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter sorted mask back to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1,
                        index=sorted_indices,
                        src=sorted_indices_to_remove
                    )
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, 0.0)

                # Sample next token
                probs = filtered_logits / torch.sum(filtered_logits, dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)

                # Check for EOS token
                if next_token.item() == config.eos_token_id:
                    eos_reached = True
                
                # Append token to sequence
                generated = torch.cat([generated, next_token], dim=-1)
                current_length += 1

            # Convert to text and clean up special tokens
            text = self.tokenizer.decode(
                generated[0], 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            generated_texts.append(text)
        
        return generated_texts if len(generated_texts) > 1 else generated_texts[0]

def load_model_and_tokenizer(model_path: str, model_config: dict) -> tuple:
    """Load model and tokenizer with proper error handling."""
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.add_special_tokens({
            "eos_token": "<|endoftext|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>"
        })
        
        model = DeepSeekV3(model_config)  
        checkpoint = torch.load(
            f"{model_path}",
            map_location=torch.device('cpu')
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Error loading model and tokenizer: {e}")

def load_config(config_path: str) -> dict:
    """Load configuration file with error handling."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading config from {config_path}: {e}")

def main():    
    try:
        base_config = load_config('config/base.yaml')
        set_seed(base_config["seed"])    
        model_config = load_config('config/model.yaml')        
        model, tokenizer = load_model_and_tokenizer(
            model_path="checkpoints/checkpoint_epoch_1.pt",
            model_config=model_config
        )
        generator = TextGenerator(model, tokenizer)        
        gen_config = GenerationConfig(
            max_length=50,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        prompts = [
            "World heritage sites in Germany are ",
            "In a galaxy far, far away ",
            "The future of artificial intelligence "
        ]
        
        for prompt in prompts:
            generated_text = generator.generate(prompt, gen_config)
            print(f"Generated text for prompt:\n{prompt}\n{generated_text}\n")
            print("--------------------------------------------------")
            
    except Exception as e:
        print(f"Error in main execution: {e}")

# Example usage
if __name__ == "__main__":
    main()