import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

class MultiTokenPrediction(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, depth: int = 1):
        super().__init__()
        self.depth = depth
        self.hidden_dim = hidden_dim
        
        # Projection layers for each depth
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim * 2),  # Input size is hidden_dim + 1 (hidden states + target)
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)        # Output size is hidden_dim
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)                 # Layer normalization
        self.output_head = nn.Linear(hidden_dim, vocab_size) # Final output projection to vocabulary size

    def forward(self, hidden_states: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        predictions = []
        seq_len = hidden_states.size(1)

        for d in range(self.depth):
            # Slice the targets to get the current depth's target tokens
            shifted_targets = targets[:, d * seq_len : (d + 1) * seq_len].unsqueeze(-1)
            
            # Combine hidden states with shifted targets
            combined = torch.cat([hidden_states, shifted_targets], dim=-1)  # [batch_size, seq_len, hidden_dim + 1]
            
            # Project and predict
            projected = self.proj[d](combined)
            normalized = self.norm(projected)
            predictions.append(normalized)  # Store normalized hidden states
            
            # Update hidden states for the next depth
            hidden_states = projected

        # Return the list of hidden states instead of logits
        return torch.stack(predictions, dim=1)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int,
                 kv_compression_dim: int, query_compression_dim: int,
                 rope_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        # Compressed projections
        self.d_kv = nn.Linear(hidden_dim, kv_compression_dim)
        self.u_k = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.u_v = nn.Linear(kv_compression_dim, num_heads * head_dim)
        
        # Query projections with separate components
        self.d_q = nn.Linear(hidden_dim, query_compression_dim)
        self.u_q = nn.Linear(query_compression_dim, num_heads * head_dim)
        self.qr_proj = nn.Linear(query_compression_dim, num_heads * rope_dim)
        
        # RoPE for decoupled components
        self.rope = RotaryPositionalEmbedding(rope_dim)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # Key/Value compression (Eq 1-5)
        kv_compressed = self.d_kv(x)
        k_c = self.u_k(kv_compressed).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k_r = self.qr_proj(kv_compressed).view(batch_size, seq_len, self.num_heads, self.rope_dim)
        
        # Value projection
        v = self.u_v(kv_compressed).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Query compression (Eq 6-9)
        q_compressed = self.d_q(x)
        q_c = self.u_q(q_compressed).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q_r = self.qr_proj(q_compressed).view(batch_size, seq_len, self.num_heads, self.rope_dim)

        # Apply RoPE to both q and k
        q_r = self.rope(q_r)
        k_r = self.rope(k_r)

        # Combine components (Eq 4,9)
        q = torch.cat([q_c, q_r], dim=-1)  # Shape: [batch_size, seq_len, num_heads, head_dim + rope_dim]
        k = torch.cat([k_c, k_r], dim=-1)  # Shape: [batch_size, seq_len, num_heads, head_dim + rope_dim]

        # Attention computation (Eq 10-11)
        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(self.head_dim + self.rope_dim)

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.einsum('bhqk,bkhd->bqhd', attn_probs, v)
        return self.out_proj(context.reshape(batch_size, seq_len, -1))


class DeepSeekMoE(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int,
                 capacity_factor: float = 1.0, drop_tokens: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens

        # Shared expert
        self.shared_expert = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        
        # Routed experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Linear(4 * hidden_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating with auxiliary-loss-free balancing
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.bias_update_speed = 0.001
        
        # Expert load tracking (initialized as a buffer)
        self.register_buffer('expert_load', torch.zeros(num_experts))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        combined = torch.zeros_like(x_flat)

        # Calculate scores with bias
        scores = self.gate(x_flat) + self.bias.unsqueeze(0)
        scores = F.sigmoid(scores)
        top_scores, top_indices = scores.topk(self.top_k, dim=-1)

        # Adjust top_indices to include all selected experts
        mask = F.one_hot(top_indices, self.num_experts).float().sum(dim=1)
        expert_load = mask.sum(dim=0)
        self.bias.data += self.bias_update_speed * (expert_load - self.bias.data)
        self.expert_load = 0.9 * self.expert_load + 0.1 * expert_load

        # Collect outputs and indices for selected experts
        expert_outputs = []
        indices = []

        for expert_idx in range(self.num_experts):
            selected = (top_indices == expert_idx).any(dim=-1)
            selected_indices = torch.where(selected)[0]
            if not selected.any():
                continue

            expert_in = x_flat[selected]
            expert_out = self.experts[expert_idx](expert_in)

            # Get scores and expand them
            positions = (top_indices[selected] == expert_idx).nonzero(as_tuple=True)[1]
            expert_weights = top_scores[selected, positions].unsqueeze(-1)
            weighted_out = expert_weights * expert_out

            expert_outputs.append(weighted_out)
            indices.append(selected_indices)

        # Scatter the expert outputs into combined
        if expert_outputs:
            outputs_cat = torch.cat(expert_outputs, dim=0)
            indices_cat = torch.cat(indices, dim=0)
            indices_expanded = indices_cat.unsqueeze(-1).expand(-1, outputs_cat.size(-1))
            combined.scatter_add_(0, indices_expanded, outputs_cat)

        # Add shared expert output
        shared_out = self.shared_expert(x)
        return shared_out + combined.view_as(x)

class DeepSeekV3(nn.Module):
    """Full model with Multi-Token Prediction (MTP) and FP8Linear integration."""
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['hidden_dim'])
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn_norm': nn.LayerNorm(config['hidden_dim']),
                'attention': MultiHeadLatentAttention(
                    config['hidden_dim'],
                    config['num_heads'],
                    config['head_dim'],
                    config['kv_compression_dim'],
                    config['query_compression_dim'],
                    config['rope_dim']
                ),
                'moe_norm': nn.LayerNorm(config['hidden_dim']),
                'moe': DeepSeekMoE(
                    config['hidden_dim'],
                    config['num_experts'],
                    config['activated_experts']
                )
            }) for _ in range(config['num_layers'])
        ])
        self.final_norm = nn.LayerNorm(config['hidden_dim'])
        self.output_head = nn.Linear(config['hidden_dim'], config['vocab_size'])
        
        # Multi-Token Prediction (MTP) module
        self.mtp = MultiTokenPrediction(config['hidden_dim'], config['vocab_size'], depth=1)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None, target_ids: Optional[Tensor] = None) -> Tensor:
        x = self.embedding(input_ids)  # Embed the input tokens
        
        for layer in self.layers:
            # Attention block with pre-layer normalization
            attn_norm = layer['attn_norm'](x)
            attn_out = layer['attention'](attn_norm, attention_mask)
            x = x + attn_out  # Residual connection
            
            # Mixture of Experts (MoE) block with pre-layer normalization
            moe_norm = layer['moe_norm'](x)
            moe_out = layer['moe'](moe_norm)
            x = x + moe_out  # Residual connection
        
        x = self.final_norm(x)  # Final normalization
        
        # Main model output logits
        main_output = self.output_head(x)
        
        # Multi-Token Prediction (MTP) output if target_ids are provided
        if target_ids is not None:
            # Ensure target_ids has the correct shape [batch_size, depth * seq_len]
            mtp_output = self.mtp(x, target_ids)
            return main_output, mtp_output
        else:
             return main_output
    
class RotaryPositionalEmbedding(nn.Module):
    """Correct RoPE implementation matching paper equations"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        position = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        
        sinusoid = torch.einsum('i,j->ij', position, self.inv_freq.to(x.device))
        sin = torch.sin(sinusoid).unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, -1)
        cos = torch.cos(sinusoid).unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, -1)
        
        # Split and rotate
        x_rot = x.view(batch_size, num_heads, seq_len, head_dim // 2, 2)
        x1, x2 = x_rot.unbind(dim=-1)
        
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated.view(batch_size, num_heads, seq_len, head_dim)
        