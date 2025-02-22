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

        # Initialize weights with balanced variance
        self._initialize_weights(hidden_dim)

    def _initialize_weights(self, hidden_dim: int):
        std = 0.02 / math.sqrt(hidden_dim)
        for layer in [self.kv_down, self.key_up, self.value_up, self.key_rope,
                      self.query_down, self.query_up, self.query_rope, self.output_proj]:
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Compress and project keys/values
        kv_compressed = self.kv_down(x)
        keys_c = self.key_up(kv_compressed).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value_up(kv_compressed).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys_r = self.key_rope(kv_compressed).view(batch_size, seq_len, self.num_heads, self.rope_dim)
        keys_r = self.rope(keys_r)
        keys = torch.cat([keys_c, keys_r], dim=-1)

        # Compress and project queries
        query_compressed = self.query_down(x)
        queries_c = self.query_up(query_compressed).view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries_r = self.query_rope(query_compressed).view(batch_size, seq_len, self.num_heads, self.rope_dim)
        queries_r = self.rope(queries_r)
        queries = torch.cat([queries_c, queries_r], dim=-1)

        # Compute attention scores
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", queries, keys) / math.sqrt(self.head_dim + self.rope_dim)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-1e9"))
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        context = torch.einsum("bhqk,bkhd->bqhd", attn_probs, values)
        return self.output_proj(context.reshape(batch_size, seq_len, -1))

class ExpertFFN(nn.Module):
    """Single Feed-Forward Network (FFN) for MoE experts."""
    def __init__(self, hidden_dim: int, expansion_factor: int = 4):
        super().__init__()
        intermediate_dim = hidden_dim * expansion_factor
        self.up = nn.Linear(hidden_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.down = nn.Linear(intermediate_dim, hidden_dim)
        self._initialize_weights(hidden_dim)

    def _initialize_weights(self, hidden_dim: int):
        std = 0.02 / math.sqrt(hidden_dim)
        nn.init.normal_(self.up.weight, mean=0.0, std=std)
        nn.init.normal_(self.down.weight, mean=0.0, std=std)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.down.bias)

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
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-6)  # Normalize gating

        # Update load balancing bias
        mask = F.one_hot(top_indices, self.num_experts).sum(dim=1).float()  # [bs * seq_len, num_experts]
        expert_load = mask.sum(dim=0)  # [num_experts]
        self.bias.data += self.bias_update_speed * (expert_load - self.expert_load)
        self.expert_load.lerp_(expert_load, 0.1)  # Exponential moving average

        # Compute expert outputs
        combined = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            selected = (top_indices == expert_idx).any(dim=-1)
            if not selected.any():
                continue
            indices = torch.where(selected)[0]
            expert_in = x_flat[indices]
            expert_out = expert(expert_in) * 0.1  # Scale to prevent explosion
            weights = top_scores[indices, (top_indices[indices] == expert_idx).nonzero(as_tuple=True)[1]]
            combined.index_add_(0, indices, expert_out * weights.unsqueeze(-1))

        # Add shared expert output
        shared_out = self.shared_expert(x) * 0.1  # Scale shared output
        moe_out = shared_out + combined.view_as(x)

        # Normalize per sequence to prevent extreme values
        moe_out = moe_out / (moe_out.abs().max(dim=-1, keepdim=True)[0] + 1e-6)
        return moe_out

class MultiTokenPrediction(nn.Module):
    """Predicts multiple future tokens to enhance training efficiency."""
    def __init__(self, hidden_dim: int, vocab_size: int, depth: int = 1):
        super().__init__()
        self.depth = depth
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        self._initialize_weights(hidden_dim)

    def _initialize_weights(self, hidden_dim: int):
        std = 0.02 / math.sqrt(hidden_dim)
        for layer in self.proj_layers:
            nn.init.normal_(layer[0].weight, mean=0.0, std=std)
            nn.init.normal_(layer[2].weight, mean=0.0, std=std)
            nn.init.zeros_(layer[0].bias)
            nn.init.zeros_(layer[2].bias)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=std)

    def forward(self, hidden_states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = []
        seq_len = hidden_states.size(1)
        for d in range(self.depth):
            target_slice = targets[:, d * seq_len:(d + 1) * seq_len].unsqueeze(-1)
            combined = torch.cat([hidden_states, target_slice], dim=-1)
            projected = self.proj_layers[d](combined)
            normalized = self.norm(projected)
            predictions.append(normalized)
            hidden_states = projected
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
        self._initialize_weights()

    def _initialize_weights(self):
        std = 0.02 / math.sqrt(self.config["hidden_dim"])
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=std)
        for layer in self.layers:
            nn.init.normal_(layer["attn_norm"].weight, mean=1.0, std=0.02)
            nn.init.normal_(layer["attn_norm"].bias, mean=0.0, std=0.02)
            nn.init.normal_(layer["moe_norm"].weight, mean=1.0, std=0.02)
            nn.init.normal_(layer["moe_norm"].bias, mean=0.0, std=0.02)
        nn.init.normal_(self.final_norm.weight, mean=1.0, std=0.02)
        nn.init.normal_(self.final_norm.bias, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                target_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(input_ids)

        for layer in self.layers:
            # Attention block
            attn_input = layer["attn_norm"](x)
            attn_input = attn_input - attn_input.mean(dim=-1, keepdim=True) + 1.0  # Center and shift
            attn_output = layer["attention"](attn_input, attention_mask)
            x = x + attn_output

            # MoE block
            moe_input = layer["moe_norm"](x)
            moe_output = layer["moe"](moe_input)
            x = x + moe_output

        x = self.final_norm(x)
        logits = self.output_head(x)
        logits = logits - logits.mean(dim=-1, keepdim=True)  # Ensure balanced logits

        if target_ids is not None:
            mtp_output = self.mtp(x, target_ids)
            return logits, mtp_output
        return logits

