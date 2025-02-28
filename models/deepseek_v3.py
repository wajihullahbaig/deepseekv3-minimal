import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class RotaryPositionalEmbedding(nn.Module):
    """Applies Rotary Positional Embeddings (RoPE) to enhance positional awareness."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Precompute inverse frequencies for efficiency
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        positions = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        sinusoid = positions[:, None] * self.inv_freq[None, :]  # [seq_len, dim/2]
        sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)
        sin = sin[None, None, :, :].expand(batch_size, num_heads, -1, -1)
        cos = cos[None, None, :, :].expand(batch_size, num_heads, -1, -1)

        # Rotate pairs of dimensions
        x_rot = x.view(batch_size, num_heads, seq_len, head_dim // 2, 2)
        x1, x2 = x_rot.unbind(dim=-1)
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated.view(batch_size, num_heads, seq_len, head_dim)

class MultiHeadLatentAttention(nn.Module):
    """Implements Multi-head Latent Attention (MLA) for efficient inference."""
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int,
                 kv_compression_dim: int, query_compression_dim: int, rope_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim

        # Key/Value compression layers
        self.kv_down = nn.Linear(hidden_dim, kv_compression_dim)
        self.key_up = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.value_up = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.key_rope = nn.Linear(kv_compression_dim, num_heads * rope_dim)

        # Query compression layers
        self.query_down = nn.Linear(hidden_dim, query_compression_dim)
        self.query_up = nn.Linear(query_compression_dim, num_heads * head_dim)
        self.query_rope = nn.Linear(query_compression_dim, num_heads * rope_dim)

        self.rope = RotaryPositionalEmbedding(rope_dim)
        self.output_proj = nn.Linear(num_heads * head_dim, hidden_dim)


    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of Multi-head Latent Attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask [batch_size, seq_len] or [batch_size, 1, seq_len, seq_len]
            
        Returns:
            Output tensor after attention [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Compress and project keys/values
        kv_compressed = self.kv_down(x)
        keys_c = self.key_up(kv_compressed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_up(kv_compressed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys_r = self.key_rope(kv_compressed).view(batch_size, seq_len, self.num_heads, self.rope_dim).transpose(1, 2)
        keys_r = self.rope(keys_r)
        
        # Compress and project queries
        query_compressed = self.query_down(x)
        queries_c = self.query_up(query_compressed).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries_r = self.query_rope(query_compressed).view(batch_size, seq_len, self.num_heads, self.rope_dim).transpose(1, 2)
        queries_r = self.rope(queries_r)
        
        # Concatenate rotary and non-rotary parts
        queries = torch.cat([queries_c, queries_r], dim=-1)
        keys = torch.cat([keys_c, keys_r], dim=-1)

        # Compute attention scores - shape: [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.head_dim + self.rope_dim)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
        
        # Process attention mask if provided
        if attention_mask is not None:
            # Convert from [batch_size, seq_len] to 4D if needed
            if attention_mask.dim() == 2:
                # Convert to boolean mask first (1 -> True, 0 -> False)
                attention_mask = attention_mask.bool()
                # Then expand to 4D: [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # Expand if needed to match attention score dims
                attention_mask = attention_mask.expand(-1, -1, seq_len, -1)
                
                # Invert the mask - we want True where tokens should be ignored
                attention_mask = ~attention_mask
            elif attention_mask.dim() == 4:
                # If already 4D, just ensure it's boolean
                attention_mask = attention_mask.bool()
            
            # Combine with causal mask - we want to apply both masks
            # (True wherever we want to mask out)
            combined_mask = causal_mask | attention_mask
        else:
            # Just use causal mask
            combined_mask = causal_mask
        
        # Apply mask - set masked positions to negative infinity
        attn_scores = attn_scores.masked_fill(combined_mask, float("-1e9"))
        
        # Apply softmax to get attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_probs, values)  # [batch_size, num_heads, seq_len, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.output_proj(context)

class ExpertFFN(nn.Module):
    """Single Feed-Forward Network (FFN) for MoE experts."""
    def __init__(self, hidden_dim: int, expansion_factor: int = 4):
        super().__init__()
        intermediate_dim = hidden_dim * expansion_factor
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.down = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.gelu(self.up(x)))

class DeepSeekMoE(nn.Module):
    """Mixture of Experts (MoE) with auxiliary-loss-free load balancing."""
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.shared_expert = ExpertFFN(hidden_dim)
        self.experts = nn.ModuleList([ExpertFFN(hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_experts))  # For load balancing
        self.bias_update_speed = 0.001
        self.register_buffer("expert_load", torch.zeros(num_experts))

        # Initialize gate with smaller variance
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02 / math.sqrt(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]

        # Compute gating scores with bias for routing
        scores = F.sigmoid(self.gate(x_flat) + self.bias)  # [bs * seq_len, num_experts]
        top_scores, top_indices = scores.topk(self.top_k, dim=-1)  # [bs * seq_len, top_k]
        
        # Normalize top scores for weighting
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Update load balancing bias (keep your original implementation)
        mask = F.one_hot(top_indices, self.num_experts).sum(dim=1).float()  # [bs * seq_len, num_experts]
        expert_load = mask.sum(dim=0)  # [num_experts]
        self.bias.data += self.bias_update_speed * (expert_load - self.expert_load)
        self.expert_load.lerp_(expert_load, 0.1)  # Exponential moving average

        # Apply experts with proper weighting
        combined = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_indices = top_indices[:, i]  # [batch_size * seq_len]
            coefficient = top_scores[:, i].unsqueeze(-1)  # [batch_size * seq_len, 1]
            
            # Process inputs for each expert
            for expert_idx, expert in enumerate(self.experts):
                # Find inputs that should go to this expert
                mask = (expert_indices == expert_idx)
                if mask.any():
                    # Get inputs for this expert
                    expert_inputs = x_flat[mask]
                    # Process inputs and apply coefficient
                    expert_outputs = expert(expert_inputs) * coefficient[mask]
                    # Add outputs to the combined tensor
                    combined.index_add_(0, torch.where(mask)[0], expert_outputs)

        # Add shared expert output
        shared_out = self.shared_expert(x_flat) * 0.1  # Scale shared output
        combined = combined + shared_out
        
        # Reshape back to original dimensions
        return combined.view(batch_size, seq_len, hidden_dim)

class MultiTokenPrediction(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, depth: int = 1):
        super().__init__()
        self.depth = depth
        self.proj_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        predictions = []
        current_hidden = hidden_states
        
        for d in range(self.depth):
            # Project the hidden states
            projected = self.proj_layers[d](current_hidden)
            normalized = self.norm(projected)
            # Apply output head directly here
            logits = self.output_head(normalized)
            predictions.append(logits)
            current_hidden = projected
            
        return torch.stack(predictions, dim=1)

class DeepSeekV3(nn.Module):
    """DeepSeek-V3: A Mixture-of-Experts Transformer with Multi-Token Prediction."""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["hidden_dim"])
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn_norm": nn.LayerNorm(config["hidden_dim"]),
                "attention": MultiHeadLatentAttention(
                    hidden_dim=config["hidden_dim"],
                    num_heads=config["num_heads"],
                    head_dim=config["head_dim"],
                    kv_compression_dim=config["kv_compression_dim"],
                    query_compression_dim=config["query_compression_dim"],
                    rope_dim=config["rope_dim"]
                ),
                "moe_norm": nn.LayerNorm(config["hidden_dim"]),
                "moe": DeepSeekMoE(
                    hidden_dim=config["hidden_dim"],
                    num_experts=config["num_experts"],
                    top_k=config["activated_experts"]
                )
            }) for _ in range(config["num_layers"])
        ])
        self.final_norm = nn.LayerNorm(config["hidden_dim"])
        self.output_head = nn.Linear(config["hidden_dim"], config["vocab_size"])
        self.mtp = MultiTokenPrediction(config["hidden_dim"], config["vocab_size"], depth=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                target_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embedding layer
        x = self.embedding(input_ids)
        
        # Process through transformer layers
        for layer in self.layers:
            # Attention block
            attn_input = layer["attn_norm"](x)
            # Center and shift normalization
            attn_input = attn_input - attn_input.mean(dim=-1, keepdim=True) + 1.0
            attn_output = layer["attention"](attn_input, attention_mask)
            x = x + attn_output
            
            # MoE block
            moe_input = layer["moe_norm"](x)
            moe_output = layer["moe"](moe_input)
            x = x + moe_output
        
        # Final normalization
        x = self.final_norm(x)
        
        # Main logits from final hidden state
        logits = self.output_head(x)
        
        # During training or when explicitly requested, also compute MTP predictions
        if (self.training and target_ids is not None) or not self.training:
            # Get multi-token predictions
            mtp_outputs = self.mtp(x)
            return logits, mtp_outputs
        
        return logits