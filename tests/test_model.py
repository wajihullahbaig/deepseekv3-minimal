import pytest
import torch

from conftest import freeze_router
from models.deepseek_v3 import DeepSeekMoE, DeepSeekV3, RotaryPositionalEmbedding

B, S = 2, 16


def _ids(cfg, b=B, s=S, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, cfg["vocab_size"], (b, s), generator=g)


def test_forward_shapes_in_train_and_eval(tiny_model, tiny_config):
    ids = _ids(tiny_config)
    V = tiny_config["vocab_size"]
    tiny_model.train()
    assert tiny_model(ids).shape == (B, S, V)                      # no targets -> logits only
    logits, mtp = tiny_model(ids, target_ids=ids)                  # targets -> (logits, mtp)
    assert logits.shape == (B, S, V) and mtp.shape == (B, 1, S, V)
    tiny_model.eval()
    logits, mtp = tiny_model(ids)                                  # eval always returns the tuple
    assert logits.shape == (B, S, V) and torch.isfinite(logits).all()


def test_causal_mask_blocks_future_tokens(tiny_model, tiny_config):
    tiny_model.eval()
    freeze_router(tiny_model)
    ids = _ids(tiny_config)
    j = 9
    alt = ids.clone()
    alt[:, j:] = 2 + (alt[:, j:] + 7) % (tiny_config["vocab_size"] - 2)
    with torch.no_grad():
        base, _ = tiny_model(ids)
        out, _ = tiny_model(alt)
    assert torch.allclose(base[:, :j], out[:, :j], atol=1e-5)
    assert not torch.allclose(base[:, j:], out[:, j:], atol=1e-5)


def test_left_padding_mask_is_honoured(tiny_model, tiny_config):
    """RoPE is relative and the FFN/MoE are position-wise, so with a correct key mask the
    logits of the real tokens must not depend on left padding."""
    tiny_model.eval()
    freeze_router(tiny_model)
    p = 5
    ids = _ids(tiny_config)
    left = torch.cat([torch.zeros(B, p, dtype=torch.long), ids], dim=1)
    mask = torch.cat([torch.zeros(B, p, dtype=torch.long), torch.ones(B, S, dtype=torch.long)], dim=1)
    with torch.no_grad():
        ref, _ = tiny_model(ids)
        masked, _ = tiny_model(left, attention_mask=mask)
        unmasked, _ = tiny_model(left)
    assert torch.allclose(ref, masked[:, p:], atol=1e-4)
    assert not torch.allclose(ref, unmasked[:, p:], atol=1e-4)


def test_rope_preserves_norm_and_depends_only_on_relative_offset():
    rope = RotaryPositionalEmbedding(8)
    x = torch.randn(1, 1, 6, 8)
    assert torch.allclose(x.norm(dim=-1), rope(x).norm(dim=-1), atol=1e-5)
    q, k = torch.randn(1, 1, 6, 8), torch.randn(1, 1, 6, 8)
    s1 = rope(q) @ rope(k).transpose(-1, -2)
    pad = torch.zeros(1, 1, 2, 8)
    s2 = (rope(torch.cat([pad, q], 2)) @ rope(torch.cat([pad, k], 2)).transpose(-1, -2))[..., 2:, 2:]
    assert torch.allclose(s1, s2, atol=1e-4)


def test_moe_routes_every_token_to_exactly_top_k_experts():
    torch.manual_seed(0)
    moe = DeepSeekMoE(hidden_dim=32, num_experts=4, top_k=2, dropout_rate=0.0)
    counts = []
    for e in moe.experts:
        e.register_forward_hook(lambda m, i, o: counts.append(i[0].shape[0]))
    x = torch.randn(2, 8, 32)
    y = moe(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert sum(counts) == 2 * 16


def test_moe_gradients_reach_all_experts_when_all_active():
    torch.manual_seed(0)
    moe = DeepSeekMoE(hidden_dim=32, num_experts=4, top_k=4, dropout_rate=0.0)
    x = torch.randn(2, 8, 32, requires_grad=True)
    moe(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(e.up.weight.grad is not None for e in moe.experts)


def test_full_model_backward_is_finite(tiny_model, tiny_config):
    tiny_model.train()
    ids = _ids(tiny_config)
    logits, mtp = tiny_model(ids, target_ids=ids)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), ids.reshape(-1))
    loss += 0.1 * torch.nn.functional.cross_entropy(mtp[:, 0].reshape(-1, mtp.size(-1)), ids.reshape(-1))
    loss.backward()
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_eval_forward_leaves_the_model_unchanged(tiny_model, tiny_config):
    tiny_model.eval()
    before = {k: v.clone() for k, v in tiny_model.state_dict().items()}
    with torch.no_grad():
        tiny_model(_ids(tiny_config))
    changed = [k for k, v in tiny_model.state_dict().items() if not torch.equal(before[k], v)]
    assert changed == [], changed


def test_router_bias_penalises_the_overloaded_expert():
    torch.manual_seed(0)
    moe = DeepSeekMoE(hidden_dim=32, num_experts=4, top_k=1, dropout_rate=0.0)
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.bias.zero_()
        moe.bias[0] = 1.0                    # expert 0 wins every token
    moe.train()
    moe(torch.randn(2, 8, 32))
    assert moe.bias[0] < 1.0
    assert (moe.bias[1:] > 0).all()
