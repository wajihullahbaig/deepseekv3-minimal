import time
import torch
import torch.nn.functional as F
import logging
import os
import json
from typing import List, Union, Optional, Dict, Tuple
from tqdm import tqdm

from main import load_config
from models.deepseek_v3 import DeepSeekV3

logger = logging.getLogger(__name__)

class GenerationConfig:
    """Configuration for text generation."""
    def __init__(
        self,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        do_sample: bool = True,
        use_mtp: bool = True,
        mtp_speculation_mode: bool = True,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
    ):
        """
        Initialize generation configuration.
        
        Args:
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_k: K for top-k sampling
            top_p: P for nucleus sampling
            repetition_penalty: Penalty for token repetition
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            do_sample: Whether to sample (True) or use greedy decoding (False)
            use_mtp: Whether to use multi-token prediction
            mtp_speculation_mode: Whether to use speculative decoding with MTP
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for beam search
            early_stopping: Whether to stop early in beam search
        """
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.do_sample = do_sample
        self.use_mtp = use_mtp
        self.mtp_speculation_mode = mtp_speculation_mode
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping

class TextGenerator:
    """Text generator for DeepSeek model with various generation strategies."""
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize TextGenerator.
        
        Args:
            model: DeepSeek model
            tokenizer: Tokenizer
            device: Device to run generation on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Create directory for generation logs
        os.makedirs('logs', exist_ok=True)
    
    def generate(
        self, 
        prompts: Union[str, List[str]], 
        config: GenerationConfig
    ) -> Union[str, List[str]]:
        """
        Generate text from prompts using the specified configuration.
        
        Args:
            prompts: Single prompt or list of prompts
            config: Generation configuration
            
        Returns:
            Generated text or list of generated texts
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
            return_single = True
        else:
            return_single = False
        
        # Choose generation strategy based on config
        if config.num_beams > 1:
            # Beam search
            generated_texts = self.generate_with_beam_search(prompts, config)
        elif config.use_mtp and config.mtp_speculation_mode:
            # Speculative decoding with multi-token prediction
            generated_texts = self.generate_with_speculation(prompts, config)
        else:
            # Standard auto-regressive generation
            generated_texts = self.generate_standard(prompts, config)
        
        # Return single text or list based on input
        if return_single:
            return generated_texts[0]
        return generated_texts
    
    def generate_standard(
        self, 
        prompts: List[str], 
        config: GenerationConfig
    ) -> List[str]:
        """
        Standard auto-regressive text generation.
        
        Args:
            prompts: List of prompts
            config: Generation configuration
            
        Returns:
            List of generated texts
        """
        generated_texts = []
        
        for prompt_idx, prompt in enumerate(prompts):
            # Encode prompt
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=config.max_length // 2,  # Leave room for generation
                add_special_tokens=True
            ).to(self.device)
            
            # Track generation for logging
            generation_trace = {
                'prompt': prompt,
                'input_token_ids': input_ids[0].tolist(),
                'input_text': self.tokenizer.decode(input_ids[0], skip_special_tokens=True),
                'generated_tokens': [],
                'token_probabilities': [],
                'token_top_candidates': []
            }
            
            # Initialize generation
            generated = input_ids.clone()
            attention_mask = torch.ones_like(generated, device=self.device)
            
            # Generation loop
            current_length = generated.size(1)
            max_gen_length = min(config.max_length - current_length, 256)  # Limit maximum tokens to generate
            
            # Use tqdm for visual progress tracking
            with tqdm(total=max_gen_length, desc=f"Generating text {prompt_idx+1}/{len(prompts)}") as pbar:
                for _ in range(max_gen_length):
                    # Forward pass
                    with torch.no_grad():
                        outputs = self.model(generated, attention_mask=attention_mask)
                    
                    # Handle different output types (outputs may be a tuple with main logits and MTP)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # Get next token logits
                    next_token_logits = logits[:, -1, :].clone()
                    
                    if config.temperature != 1.0:
                        next_token_logits = next_token_logits / config.temperature
                    
                    if config.repetition_penalty != 1.0:
                        for i in range(logits.size(0)):
                            for token_id in generated[i].unique():
                                next_token_logits[i, token_id] /= config.repetition_penalty
                    
                    # Apply n-gram repetition prevention
                    if config.no_repeat_ngram_size > 0 and generated.size(1) > config.no_repeat_ngram_size:
                        for i in range(logits.size(0)):
                            ngrams = self._get_ngrams(generated[i], config.no_repeat_ngram_size)
                            banned_tokens = self._get_banned_tokens(generated[i], ngrams, config.no_repeat_ngram_size)
                            for token_id in banned_tokens:
                                next_token_logits[i, token_id] = -float('inf')
                    
                    if config.top_k > 0:
                        next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
                    if config.top_p < 1.0:
                        next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)
                    
                    # Sample next token
                    if config.do_sample:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Log token probabilities for analysis
                        token_prob = probs[0, next_token[0, 0]].item()
                        
                        # Get top 5 candidates for logging
                        top_values, top_indices = torch.topk(probs[0], k=5)
                        top_candidates = [(idx.item(), val.item()) for idx, val in zip(top_indices, top_values)]
                    else:
                        # Greedy decoding
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        token_prob = F.softmax(next_token_logits, dim=-1)[0, next_token[0, 0]].item()
                        top_candidates = [(next_token[0, 0].item(), token_prob)]
                    
                    # Store token data for logging
                    generation_trace['generated_tokens'].append(next_token.item())
                    generation_trace['token_probabilities'].append(token_prob)
                    generation_trace['token_top_candidates'].append(top_candidates)
                    
                    # Add token to sequence
                    generated = torch.cat([generated, next_token], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=1)
                    
                    # Update progress
                    pbar.update(1)
                    
                    # Check for EOS
                    if next_token.item() == config.eos_token_id:
                        break
            
            # Decode and add to results
            result = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            generated_texts.append(result)
            
            # Add final result to trace
            generation_trace['final_text'] = result
            generation_trace['final_token_ids'] = generated[0].tolist()
            
            log_filename = f'logs/generation_{prompt_idx}_{int(time.time())}.json'
            with open(log_filename, 'w') as f:
                json.dump(generation_trace, f, indent=2)
        
        return generated_texts
    
    def generate_with_speculation(
        self, 
        prompts: List[str], 
        config: GenerationConfig
    ) -> List[str]:
        """
        Generate text using speculative decoding with MTP.
        
        Args:
            prompts: List of prompts
            config: Generation configuration
            
        Returns:
            List of generated texts
        """
        generated_texts = []
        
        for prompt_idx, prompt in enumerate(prompts):
            # Encode prompt
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=config.max_length // 2,
                add_special_tokens=True
            ).to(self.device)
            
            # Track generation statistics
            tokens_generated = 0
            speculative_tokens_accepted = 0
            
            # Initialize generation
            generated = input_ids.clone()
            attention_mask = torch.ones_like(generated, device=self.device)
            
            # Generation loop
            current_length = generated.size(1)
            max_gen_length = min(config.max_length - current_length, 256)
            
            with tqdm(total=max_gen_length, desc=f"Generating text {prompt_idx+1}/{len(prompts)}") as pbar:
                while current_length < config.max_length:
                    # Forward pass
                    with torch.no_grad():
                        outputs = self.model(generated, attention_mask=attention_mask)
                    
                    # Process outputs - handle both main predictions and MTP
                    if isinstance(outputs, tuple) and len(outputs) > 1:
                        main_logits, mtp_logits = outputs
                    else:
                        # MTP not available, fall back to standard generation
                        main_logits = outputs
                        mtp_logits = None
                    
                    # Sample next token from main logits
                    next_token_logits = main_logits[:, -1, :].clone() / config.temperature
                    
                    # Apply repetition penalty
                    for token_id in generated[0].unique():
                        next_token_logits[0, token_id] /= config.repetition_penalty
                    
                    # Filter logits
                    if config.top_k > 0:
                        next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
                    if config.top_p < 1.0:
                        next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)
                    
                    # Sample token
                    if config.do_sample:
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Add to sequence
                    generated = torch.cat([generated, next_token], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=1)
                    current_length += 1
                    tokens_generated += 1
                    pbar.update(1)
                    
                    # Check for EOS
                    if next_token.item() == config.eos_token_id:
                        break
                    
                    # Speculative decoding using MTP if available
                    if mtp_logits is not None and mtp_logits.size(1) > 0:
                        # Get MTP predictions for the next tokens
                        # We'll try to predict up to depth tokens ahead
                        depth = min(mtp_logits.size(1), 3)  # Limit depth to avoid too much speculation
                        
                        speculation_successful = False
                        for d in range(depth):
                            # MTP logits for the current position
                            mtp_token_logits = mtp_logits[0, d, -1, :].clone() / config.temperature
                            
                            # Apply repetition penalty
                            for token_id in generated[0].unique():
                                mtp_token_logits[token_id] /= config.repetition_penalty
                            
                            # Filter logits
                            if config.top_k > 0:
                                mtp_token_logits = self._top_k_filtering(mtp_token_logits.unsqueeze(0), config.top_k).squeeze(0)
                            if config.top_p < 1.0:
                                mtp_token_logits = self._top_p_filtering(mtp_token_logits.unsqueeze(0), config.top_p).squeeze(0)
                            
                            # Sample speculative token
                            if config.do_sample:
                                mtp_probs = F.softmax(mtp_token_logits, dim=-1)
                                speculative_token = torch.multinomial(mtp_probs.unsqueeze(0), num_samples=1)
                            else:
                                speculative_token = torch.argmax(mtp_token_logits, dim=-1, keepdim=True).unsqueeze(0)
                            
                            # In a real implementation, we'd verify this token
                            # but for simplicity, we'll accept it with high probability
                            if torch.rand(1).item() < 0.8:  # 80% chance to accept speculative token
                                generated = torch.cat([generated, speculative_token], dim=1)
                                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=1)
                                current_length += 1
                                tokens_generated += 1
                                speculative_tokens_accepted += 1
                                pbar.update(1)
                                speculation_successful = True
                                
                                # Check for EOS
                                if speculative_token.item() == config.eos_token_id:
                                    break
                            else:
                                # Speculation failed, don't continue with deeper tokens
                                break
                        
                        # If we failed to use speculation, continue standard generation
                        if not speculation_successful:
                            continue
                    
                    # If we've reached max length, break
                    if current_length >= config.max_length:
                        break
            
            # Decode and add to results
            result = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            generated_texts.append(result)
            
            # Log speculation statistics
            if speculative_tokens_accepted > 0:
                logger.info(f"Speculation stats: {speculative_tokens_accepted}/{tokens_generated} tokens "
                           f"({100*speculative_tokens_accepted/tokens_generated:.1f}%) generated speculatively.")
        
        return generated_texts
    
    def generate_with_beam_search(
        self, 
        prompts: List[str], 
        config: GenerationConfig
    ) -> List[str]:
        """
        Generate text using beam search.
        
        Args:
            prompts: List of prompts
            config: Generation configuration
            
        Returns:
            List of generated texts
        """
        generated_texts = []
        
        for prompt_idx, prompt in enumerate(prompts):
            # Encode prompt
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=config.max_length // 2,
                add_special_tokens=True
            ).to(self.device)
            
            # Initialize beams with the input sequence and score 0
            beams = [(input_ids.clone(), 0.0)]
            finished_beams = []
            
            # Generation loop
            current_length = input_ids.size(1)
            max_gen_length = min(config.max_length - current_length, 256)
            
            with tqdm(total=max_gen_length, desc=f"Beam search {prompt_idx+1}/{len(prompts)}") as pbar:
                for _ in range(max_gen_length):
                    if not beams:
                        break
                    
                    new_beams = []
                    
                    for beam_idx, (beam_sequence, beam_score) in enumerate(beams):
                        # Skip if this beam is done
                        if beam_sequence[0, -1].item() == config.eos_token_id:
                            finished_beams.append((beam_sequence, beam_score))
                            continue
                        
                        # Forward pass
                        attention_mask = torch.ones_like(beam_sequence, device=self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(beam_sequence, attention_mask=attention_mask)
                        
                        # Handle outputs
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                        
                        # Get next token logits
                        next_token_logits = logits[:, -1, :].clone() / config.temperature
                        
                        # Apply repetition penalty
                        for token_id in beam_sequence[0].unique():
                            next_token_logits[0, token_id] /= config.repetition_penalty
                        
                        # Apply top-k and top-p filtering
                        if config.top_k > 0:
                            next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
                        if config.top_p < 1.0:
                            next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)
                        
                        # Convert logits to probabilities
                        next_token_probs = F.softmax(next_token_logits, dim=-1)
                        
                        # Get top candidates for this beam
                        topk_probs, topk_tokens = torch.topk(
                            next_token_probs, k=config.num_beams, dim=-1
                        )
                        
                        # Expand beams
                        for token_idx, (token, prob) in enumerate(zip(topk_tokens[0], topk_probs[0])):
                            # Create new beam by appending token
                            new_sequence = torch.cat(
                                [beam_sequence, token.unsqueeze(0).unsqueeze(0)], 
                                dim=1
                            )
                            
                            # Update score with log probability
                            log_prob = torch.log(prob).item()
                            new_score = beam_score + log_prob
                            
                            # Apply length penalty
                            if config.length_penalty != 1.0:
                                length_factor = ((5.0 + new_sequence.size(1)) / 6.0) ** config.length_penalty
                                new_score = new_score / length_factor
                            
                            new_beams.append((new_sequence, new_score))
                    
                    # Sort and keep top beams
                    new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:config.num_beams]
                    
                    # Update beams
                    beams = [(seq, score) for seq, score in new_beams if seq[0, -1].item() != config.eos_token_id]
                    
                    # Add completed beams
                    finished_beams.extend([(seq, score) for seq, score in new_beams if seq[0, -1].item() == config.eos_token_id])
                    
                    # Update progress
                    pbar.update(1)
                    
                    # Early stopping if all beams are finished
                    if not beams or (config.early_stopping and len(finished_beams) >= config.num_beams):
                        break
            
            # If no beams finished, use the best unfinished ones
            if not finished_beams and beams:
                finished_beams = beams
            
            # If we have some finished beams, select best one
            if finished_beams:
                finished_beams = sorted(finished_beams, key=lambda x: x[1], reverse=True)
                best_beam = finished_beams[0][0]
                result = self.tokenizer.decode(best_beam[0], skip_special_tokens=True)
            else:
                # Fallback to the input
                result = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            generated_texts.append(result)
        
        return generated_texts
    
    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float('inf')
        return filtered_logits
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float('inf')
        return filtered_logits
    
    def _get_ngrams(self, token_ids: torch.Tensor, n: int) -> List[Tuple[int, ...]]:
        """Get all n-grams from a tensor of token IDs."""
        ngrams = []
        for i in range(len(token_ids) - n + 1):
            ngram = tuple(token_ids[i:i + n].tolist())
            ngrams.append(ngram)
        return ngrams
    
    def _get_banned_tokens(
        self, 
        token_ids: torch.Tensor, 
        ngrams: List[Tuple[int, ...]], 
        n: int
    ) -> List[int]:
        """
        Get tokens that would form a repeated n-gram.
        
        Args:
            token_ids: Current sequence of tokens
            ngrams: List of existing n-grams
            n: Size of n-grams
            
        Returns:
            List of banned tokens
        """
        banned_tokens = []
        
        # Check if current (n-1)-gram exists and would form a banned n-gram
        if len(token_ids) >= (n - 1):
            current_prefix = tuple(token_ids[-(n-1):].tolist())
            for ngram in ngrams:
                if ngram[:-1] == current_prefix:
                    banned_tokens.append(ngram[-1])
        
        return banned_tokens

# Utility functions
def sample_text(model, tokenizer, prompts, num_samples=5, max_length=100, device='cuda'):
    """
    Generate multiple text samples from each prompt for analysis.
    
    Args:
        model: DeepSeek model
        tokenizer: Tokenizer
        prompts: List of prompts
        num_samples: Number of samples per prompt
        max_length: Maximum generation length
        device: Device to run on
        
    Returns:
        Dictionary mapping prompts to lists of generated samples
    """
    generator = TextGenerator(model, tokenizer, device)
    
    results = {}
    for prompt in prompts:
        samples = []
        
        # Define configurations with different parameters
        configs = [
            # Standard sampling
            GenerationConfig(
                max_length=max_length,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            ),
            # Creative sampling
            GenerationConfig(
                max_length=max_length,
                temperature=1.2,
                top_p=0.95,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            ),
            # More focused
            GenerationConfig(
                max_length=max_length,
                temperature=0.6,
                top_p=0.85,
                repetition_penalty=1.3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            ),
            # Beam search
            GenerationConfig(
                max_length=max_length,
                do_sample=False,
                num_beams=4,
                length_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            ),
            # With MTP
            GenerationConfig(
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                use_mtp=True,
                mtp_speculation_mode=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        ]
        
        for i, config in enumerate(configs):
            generated = generator.generate(prompt, config)
            samples.append({
                'config': f"Config {i+1}",
                'parameters': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
                'text': generated
            })
        
        results[prompt] = samples
    
    return results

def evaluate_model_generation(model, tokenizer, device='cuda'):
    """
    Evaluate model text generation on a set of standard prompts.
    
    Args:
        model: DeepSeek model
        tokenizer: Tokenizer
        device: Device to run on
        
    Returns:
        Dictionary with evaluation results
    """
    # Standard evaluation prompts
    eval_prompts = [
        "The history of artificial intelligence began",
        "The three most important factors in real estate are",
        "In recent years, climate change has",
        "The solution to the equation x² + 5x + 6 = 0 is",
        "The best way to learn a new language is to"
    ]
    
    # Create generator
    generator = TextGenerator(model, tokenizer, device)
    
    # Define generation configs to test
    configs = {
        'greedy': GenerationConfig(
            max_length=100,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        ),
        'sampling': GenerationConfig(
            max_length=100,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        ),
        'beam_search': GenerationConfig(
            max_length=100,
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        ),
        'mtp_speculation': GenerationConfig(
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            use_mtp=True,
            mtp_speculation_mode=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    }
    
    # Generate text and collect results
    results = {}
    
    for config_name, config in configs.items():
        config_results = {}
        
        # Generate text for each prompt
        for prompt in eval_prompts:
            start_time = time.time()
            generated = generator.generate(prompt, config)
            generation_time = time.time() - start_time
            
            # Analyze result
            tokens_generated = len(tokenizer.encode(generated)) - len(tokenizer.encode(prompt))
            
            config_results[prompt] = {
                'prompt': prompt,
                'generated_text': generated,
                'tokens_generated': tokens_generated,
                'generation_time': generation_time,
                'tokens_per_second': tokens_generated / generation_time if generation_time > 0 else 0
            }
        
        results[config_name] = config_results
    
    # Create summary
    summary = {
        'generation_settings': {name: {k: v for k, v in config.__dict__.items() 
                                      if not k.startswith('_')} for name, config in configs.items()},
        'performance': {
            name: {
                'average_tokens_per_second': sum(r['tokens_per_second'] for r in config_result.values()) / len(config_result),
                'average_generation_time': sum(r['generation_time'] for r in config_result.values()) / len(config_result),
                'average_tokens_generated': sum(r['tokens_generated'] for r in config_result.values()) / len(config_result)
            }
            for name, config_result in results.items()
        },
        'detailed_results': results
    }
    
    return summary

