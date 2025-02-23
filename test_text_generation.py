import torch
from models.deepseek_v3 import DeepSeekV3
import yaml
from transformers import T5Tokenizer
from seeding import set_seed
from models.deepseek_v3 import DeepSeekV3
from typing import List, Union
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
    
    def generate_with_beam_search(self, prompts: Union[str, List[str]], config: GenerationConfig, beam_width: int = 3) -> Union[str, List[str]]:
        """
        Generate text using beam search with length normalization for better sentence quality.
        
        Args:
            prompts: Single prompt or list of prompts.
            config: GenerationConfig object with generation parameters.
            beam_width: Number of beams to maintain during search (default: 5).
        
        Returns:
            Generated text(s) as a string or list of strings.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        generated_texts = []
        for prompt in prompts:
            # Encode the input prompt
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=config.max_length,
                add_special_tokens=False
            ).to(self.device)
            
            # Initialize beams: (sequence, score)
            beams = [(input_ids.clone(), 0.0)]  # Start with input and score 0
            finished_beams = []
            
            current_length = input_ids.size(1)
            
            while current_length < config.max_length and beams:
                new_beams = []
                
                for beam_seq, beam_score in beams:
                    # Stop if EOS token is already in the sequence
                    if beam_seq[0, -1].item() == config.eos_token_id:
                        finished_beams.append((beam_seq, beam_score))
                        continue
                    
                    # Create causal mask for current sequence
                    causal_mask = self._create_causal_mask(beam_seq.size(0), beam_seq.size(1))
                    
                    # Forward pass
                    outputs = self.model(beam_seq, attention_mask=causal_mask)
                    logits = outputs[:, -1, :] / config.temperature
                    
                    # Apply repetition penalty
                    for previous_token in beam_seq[0].unique():
                        logits[0, previous_token] /= config.repetition_penalty
                    
                    # Apply top-k and top-p filtering
                    if config.top_k != 0:
                        logits = self._top_k(logits, config.top_k)
                    if config.top_p != 1.0:
                        logits = self._top_p(logits, config.top_p)
                    
                    # Get probabilities
                    probs = F.softmax(logits, dim=-1)
                    
                    # Get top beam_width candidates
                    top_probs, top_tokens = torch.topk(probs, beam_width)
                    
                    # Expand beams
                    for prob, token in zip(top_probs[0], top_tokens[0]):
                        new_seq = torch.cat([beam_seq, token.unsqueeze(0).unsqueeze(0)], dim=-1)
                        # Update score with log probability and length normalization
                        new_score = beam_score + torch.log(prob).item()
                        new_score = new_score / (new_seq.size(1) ** 0.7)  # Length normalization
                        new_beams.append((new_seq, new_score))
                
                # Sort and keep top beam_width beams
                new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                beams = new_beams
                current_length += 1
                
                # Move completed sequences to finished_beams
                beams = [(seq, score) for seq, score in beams if seq[0, -1].item() != config.eos_token_id]
                finished_beams.extend([(seq, score) for seq, score in beams if seq[0, -1].item() == config.eos_token_id])
            
            # If no beams finished, take the best unfinished beam
            if not finished_beams and beams:
                finished_beams = beams
            
            # Select the best sequence
            best_seq, _ = max(finished_beams, key=lambda x: x[1])
            text = self.tokenizer.decode(
                best_seq[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            generated_texts.append(text)
        
        return generated_texts if len(generated_texts) > 1 else generated_texts[0]

    def generate(self, prompts: Union[str, List[str]], config: GenerationConfig) -> Union[str, List[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
        
        generated_texts = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(
            prompt,
            return_tensors='pt',
            truncation=True,           
            max_length=config.max_length,
            add_special_tokens=False   
            ).to(self.device)
            generated = input_ids.clone()
            
            # Early stopping variables
            current_length = generated.size(1)
            eos_reached = False
            
            while current_length < config.max_length and not eos_reached:
                # Create causal mask for current sequence
                casual_mask = self._create_causal_mask(generated.size(0), current_length)
                
                # Model forward pass with MTP
                outputs = self.model(generated, attention_mask=casual_mask)
                
                # Use the main model output for next token prediction
                logits = outputs[:, -1, :] / config.temperature
                
                # Apply repetition penalty
                for i in range(logits.size(0)):
                    for previous_token in generated[i].unique():
                        logits[i, previous_token] /= config.repetition_penalty

                # Top-k and top-p filtering
                if config.top_k != 0:
                    logits = self._top_k(logits, config.top_k)
                if config.top_p != 1.0:
                    logits = self._top_p(logits, config.top_p)

                # Sample next token
                probs = F.softmax(logits, dim=-1)
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
    
    def _top_k(self, logits, top_k):
        """Filter logits to the top k values."""
        min_logits = torch.topk(logits, top_k)[0][:, -1, None]
        logits[logits < min_logits] = -float('inf')
        return logits
    
    def _top_p(self, logits, top_p):
        """Filter logits to retain only the top-p cumulative probability."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
        return logits

def load_model_and_tokenizer(model_path: str, model_config: dict) -> tuple:
    """Load model and tokenizer with proper error handling."""
    try:
        tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')        
        model_config["vocab_size"] = len(tokenizer) 
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
            model_path="checkpoints/checkpoint_epoch_99.pt",
            model_config=model_config
        )
        # Define prompts
        prompts = [
            "how awful sparse and misrepresented information on dash cams is i was totally",
            " surprised how under developed they are this is a video that is really necessary"
        ]
        generator = TextGenerator(model, tokenizer)   
        gen_config = GenerationConfig(
            max_length=50,
            temperature=2.0,
            top_k=50,
            top_p=0.95,
            repetition_penalty=2.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Test original generation
        print("=== Original Generation ===")
        for prompt in prompts:
            generated_text = generator.generate(prompt, gen_config)
            print(f"Prompt: {prompt}\nGenerated: {generated_text}\n")
            print("--------------------------------------------------")
        
        # Test beam search generation
        print("=== Beam Search Generation ===")
        for prompt in prompts:
            generated_text = generator.generate_with_beam_search(prompt, gen_config, beam_width=5)
            print(f"Prompt: {prompt}\nGenerated: {generated_text}\n")
            print("--------------------------------------------------")
            
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()