import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

class MultiTokenPrediction(nn.Module):
    """Multi-Token Prediction (MTP) module for predicting multiple future tokens."""
    def __init__(self, hidden_dim: int, vocab_size: int, depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.depth = depth

        # Shared embedding and output head
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)

        # Transformer blocks for each prediction depth
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,  # Adjust based on your model's configuration
                dim_feedforward=hidden_dim * 4,
                dropout=0.1
            ) for _ in range(depth)
        ])

        # Projection matrices for combining representations
        self.projection_matrices = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(depth)
        ])

    def forward(self, x: Tensor, target_ids: Tensor) -> Tensor:
        batch_size, seq_len, hidden_dim = x.shape

        # Initialize the hidden states for each prediction depth
        hidden_states = [x]  # The first hidden state is the input

        # Predict additional tokens at each depth
        for k in range(self.depth):
            # Combine the current hidden state with the embedding of the (k+1)-th target token
            target_embedding = self.embedding(target_ids[:, k:k+1])
            combined = torch.cat([hidden_states[k][:, -1:, :], target_embedding], dim=-1)
            combined = self.projection_matrices[k](combined)

            # Pass through the Transformer block
            new_hidden = self.transformer_blocks[k](combined)
            hidden_states.append(new_hidden)

        # Compute the final predictions for each depth
        predictions = []
        for k in range(1, self.depth + 1):
            predictions.append(self.output_head(hidden_states[k]))

        return torch.stack(predictions, dim=1)  


class MultiHeadLatentAttention(nn.Module):
    """MLA with proper RoPE and attention masking"""
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, 
                 kv_compression_dim: int, query_compression_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Projections
        self.down_proj_kv = nn.Linear(hidden_dim, kv_compression_dim)
        self.up_proj_k = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.up_proj_v = nn.Linear(kv_compression_dim, num_heads * head_dim)
        self.down_proj_q = nn.Linear(hidden_dim, query_compression_dim)
        self.up_proj_q = nn.Linear(query_compression_dim, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim)
        
        self.rope = RotaryPositionalEmbedding(head_dim)


    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Key/Value compression
        kv = self.down_proj_kv(x)
        k = self.up_proj_k(kv).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.up_proj_v(kv).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Query compression
        q = self.down_proj_q(x)
        q = self.up_proj_q(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.rope(q), self.rope(k)
        
        # Attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        

        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_probs @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)

        return self.out_proj(attn_output)


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


class DeepSeekMoE(nn.Module):
    """MoE with auxiliary-loss-free load balancing"""
    def __init__(self, hidden_dim: int, num_experts: int, activated_experts: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.activated_experts = activated_experts
        
        # Experts
        self.shared_expert = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating with bias
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.bias_update_speed = 0.001
        
        # Tracking expert usage
        self.register_buffer('expert_counts', torch.zeros(num_experts))

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.hidden_dim)
        
        # Gating with bias
        gate_logits = self.gate(x_flat) + self.bias
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Top-k expert selection
        topk_probs, topk_indices = torch.topk(gate_probs, self.activated_experts, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Expert computation
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x_flat[mask])
                # Expand topk_probs to match the shape of expert_out
                expert_out = expert_out * topk_probs[mask, (topk_indices[mask] == i).nonzero()[:,1]].unsqueeze(-1)
                expert_outputs.append((expert_out, mask))
        
        # Combine expert outputs
        combined = torch.zeros_like(x_flat)
        for expert_out, mask in expert_outputs:
            combined[mask] += expert_out
            
        # Update expert bias
        expert_counts = torch.bincount(topk_indices.flatten(), minlength=self.num_experts)
        self.bias.data += self.bias_update_speed * (expert_counts - expert_counts.float().mean())
        
        # Add shared expert
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
                    config['query_compression_dim']
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
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            # Attention block (pre-LN)
            attn_norm = layer['attn_norm'](x)
            attn_out = layer['attention'](attn_norm, attention_mask)
            x = x + attn_out
            
            # MoE block (pre-LN)
            moe_norm = layer['moe_norm'](x)
            moe_out = layer['moe'](moe_norm)
            x = x + moe_out
            
        x = self.final_norm(x)
        
        # Main model output
        main_output = self.output_head(x)
        
        # Multi-Token Prediction (MTP) output
        if target_ids is not None:
            mtp_output = self.mtp(x, target_ids)
            return main_output, mtp_output
        else:
            return main_output
    


        