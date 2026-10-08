"""Reproducible metrics for the repo.  Run from anywhere:

    python benchmarks/bench.py --all            # throughput, MoE balance, data coverage (~1-2 min on GPU)
    python benchmarks/bench.py --train-epoch    # one real epoch on the YouTube corpus with the current configs
    python benchmarks/bench.py --generation     # decoding timings on the checkpoint from --train-epoch

Outputs go to benchmarks/out/ (results.json plus the trainer's checkpoints/plots/logs/cache).
"""
import argparse
import json
import os
import pathlib
import sys
import time
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "benchmarks" / "out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
os.chdir(OUT)                       # trainer/generator write logs/, checkpoints/, plots/ relative to cwd

import torch                        # noqa: E402
import torch.nn.functional as F     # noqa: E402
import yaml                         # noqa: E402

from data.chunking import chunk_token_ids, split_by_source  # noqa: E402
from data.tokenizer import get_tokenizer                     # noqa: E402
from models.deepseek_v3 import DeepSeekMoE, DeepSeekV3       # noqa: E402

YT_CSV = os.environ.get("YT_CSV", "C:/Users/Precision/Onus/Data/YoutubeCommentsDataSet.csv")
MT5_VOCAB = 250100
SPM_VOCAB = 8000
RESULTS = OUT / "results.json"


def load_yaml(name):
    with open(REPO / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save(section, data):
    all_results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    all_results[section] = data
    RESULTS.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n== {section} ==\n{json.dumps(data, indent=2, ensure_ascii=True)}")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- throughput
def bench_throughput(vocab, device, bs=16, seq=128, steps=20):
    cfg = load_yaml("model_yt.yaml")
    cfg["vocab_size"] = vocab
    torch.manual_seed(0)
    model = DeepSeekV3(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    vocab_params = model.embedding.weight.numel() + model.output_head.weight.numel() + model.mtp.output_head.weight.numel()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    crit = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.15)
    ids = torch.randint(2, vocab, (bs, seq + 1), device=device)
    x, y = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
    mask = torch.ones_like(x)

    def step():
        logits, mtp = model(x, attention_mask=mask, target_ids=y)
        loss = 0.8 * crit(logits.view(-1, vocab), y.view(-1)) + 0.2 * crit(mtp[:, 0].reshape(-1, vocab), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        opt.zero_grad()

    model.train()
    for _ in range(3):
        step()
    sync()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync()
    dt = (time.perf_counter() - t0) / steps

    # forward-only time split (hooks synchronise, so this is a share, not an absolute)
    times = Counter()
    stamps = {}

    def pre(name):
        def hook(m, i):
            sync()
            stamps[name] = time.perf_counter()
        return hook

    def post(name):
        def hook(m, i, o):
            sync()
            times[name] += time.perf_counter() - stamps[name]
        return hook

    handles = []
    for layer in model.layers:
        handles += [layer["attention"].register_forward_pre_hook(pre("attention")),
                    layer["attention"].register_forward_hook(post("attention")),
                    layer["moe"].register_forward_pre_hook(pre("moe")),
                    layer["moe"].register_forward_hook(post("moe"))]
    handles += [model.output_head.register_forward_pre_hook(pre("output_head")),
                model.output_head.register_forward_hook(post("output_head")),
                model.mtp.register_forward_pre_hook(pre("mtp")),
                model.mtp.register_forward_hook(post("mtp"))]
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(x, attention_mask=mask)
        times.clear()
        sync()
        t0 = time.perf_counter()
        for _ in range(10):
            model(x, attention_mask=mask)
        sync()
        fwd = (time.perf_counter() - t0) / 10
    for h in handles:
        h.remove()
    share = {k: round(100 * v / 10 / fwd, 1) for k, v in times.items()}

    res = dict(device=device, vocab_size=vocab, batch=bs, seq_len=seq, params_M=round(n_params / 1e6, 2),
               vocab_params_pct=round(100 * vocab_params / n_params, 1),
               train_step_ms=round(dt * 1000, 1), tokens_per_s=int(bs * seq / dt),
               fwd_ms=round(fwd * 1000, 1), fwd_time_share_pct=share)
    if device == "cuda":
        res["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------- MoE balance
class OldRuleMoE(DeepSeekMoE):
    """The routing rule the repo had before the fix: bias added to the gating scores, updated by
    bias += speed * (load - ema_load) (sign inverted w.r.t. DeepSeek-V3), also in eval mode."""

    def forward(self, x):
        b, s, h = x.shape
        x_flat = x.reshape(-1, h)
        scores = torch.sigmoid(self.gate(x_flat) + self.bias)
        top_scores, top_idx = scores.topk(self.top_k, dim=-1)
        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-6)
        load = F.one_hot(top_idx, self.num_experts).sum(dim=1).float().sum(dim=0)
        self.bias += self.bias_update_speed * (load - self.expert_load)
        self.expert_load.lerp_(load, 0.1)
        combined = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            idx, coef = top_idx[:, i], top_scores[:, i].unsqueeze(-1)
            for e, expert in enumerate(self.experts):
                m = idx == e
                if m.any():
                    combined.index_add_(0, torch.where(m)[0], expert(x_flat[m]) * coef[m])
        return (combined + self.shared_expert(x_flat) * 0.1).view(b, s, h)


def bench_moe_balance(device, steps=200, bs=16, seq=128):
    """Feed a mildly skewed input stream and watch what each routing-bias rule does to the load."""
    out = {}
    for name, cls, speed in [("old_rule", OldRuleMoE, 0.001), ("no_bias_update", DeepSeekMoE, 0.0),
                             ("current_v3_rule", DeepSeekMoE, 0.001)]:
        torch.manual_seed(0)
        moe = cls(hidden_dim=128, num_experts=4, top_k=2, dropout_rate=0.0).to(device)
        with torch.no_grad():
            moe.gate.weight.normal_(0, 0.1)         # strong enough preferences to start unbalanced
        moe.bias_update_speed = speed
        moe.train()
        gen = torch.Generator(device=device).manual_seed(1)
        drift = torch.randn(128, generator=gen, device=device)
        counts = Counter()
        hooks = [e.register_forward_hook(lambda m, i, o, e_idx=e_idx: counts.__setitem__(e_idx, counts[e_idx] + i[0].shape[0]))
                 for e_idx, e in enumerate(moe.experts)]
        first = None
        with torch.no_grad():
            for _ in range(steps):
                x = torch.randn(bs, seq, 128, generator=gen, device=device) + 0.5 * drift
                counts.clear()
                moe(x)
                loads = torch.tensor([counts[i] for i in range(4)], dtype=torch.float)
                if first is None:
                    first = (loads / loads.sum()).tolist()
        for h in hooks:
            h.remove()
        frac = (loads / loads.sum()).tolist()
        out[name] = dict(step0_load_frac=[round(v, 3) for v in first], final_load_frac=[round(v, 3) for v in frac],
                         final_max_over_min=round(max(frac) / max(min(frac), 1e-9), 1),
                         final_bias=[round(v, 3) for v in moe.bias.detach().tolist()])
    return out


# --------------------------------------------------------------------------- data coverage
def bench_data_coverage(max_length=128, stride=32, sample_size=18000, seed=42):
    import pandas as pd
    base = load_yaml("base.yaml")
    texts = pd.read_csv(YT_CSV)["Comment"].dropna().astype(str).tolist()[:sample_size]
    tok = get_tokenizer(base, texts, cache_dir=str(OUT / "cache"), name="yt")
    t0 = time.perf_counter()
    enc = [tok.encode(t, add_special_tokens=True, truncation=False) for t in texts]
    tok_s = time.perf_counter() - t0
    lens = [len(e) for e in enc]
    chunks, owner = [], []
    for ci, ids in enumerate(enc):
        for c in chunk_token_ids(ids, max_length, stride, 5):
            chunks.append(c)
            owner.append(ci)
    train, val, test = split_by_source(chunks, owner, seed=seed)
    owner_of = {id(c): o for c, o in zip(chunks, owner)}
    train_owners = {owner_of[id(c)] for c in train}
    leak = lambda split: round(100 * sum(owner_of[id(c)] in train_owners for c in split) / max(len(split), 1), 1)
    targets = Counter()
    for c in chunks:
        targets.update(c[1:])
    top_tok, top_cnt = targets.most_common(1)[0]
    used = len(set(owner))
    return dict(tokenizer=f"{type(tok).__name__} ({len(tok)} pieces)", comments=len(texts), tokenize_s=round(tok_s, 1),
                median_tokens=int(sorted(lens)[len(lens) // 2]),
                comments_used=used, comments_used_pct=round(100 * used / len(texts), 1),
                tokens_total=sum(lens), chunks=len(chunks), chunks_per_used_comment=round(len(chunks) / max(used, 1), 2),
                split_chunks=dict(train=len(train), val=len(val), test=len(test)),
                val_chunks_whose_comment_is_in_train_pct=leak(val),
                test_chunks_whose_comment_is_in_train_pct=leak(test),
                most_frequent_target=dict(token=tok.convert_ids_to_tokens(top_tok),
                                          share_pct=round(100 * top_cnt / sum(targets.values()), 2)))


# --------------------------------------------------------------------------- one real epoch
def bench_train_epoch():
    from data.youtube_dataloaders import create_yt_and_loaders, load_yt_texts
    from seeding import set_seed
    from training import trainer

    base, train_cfg, model_cfg = load_yaml("base.yaml"), load_yaml("train_yt.yaml"), load_yaml("model_yt.yaml")
    train_cfg["num_epochs"] = 1
    set_seed(base["seed"])
    texts = load_yt_texts(YT_CSV, train_cfg["sample_size"])
    tok = get_tokenizer(base, texts, cache_dir=str(OUT / "cache"), name="yt")
    model_cfg["vocab_size"] = len(tok)
    model = DeepSeekV3(model_cfg)
    tr, va, te = create_yt_and_loaders(YT_CSV, tok, batch_size=train_cfg["batch_size"], min_length=model_cfg["min_seq_len"],
                                       max_length=model_cfg["max_seq_len"], stride=model_cfg["stride"],
                                       use_augmentations=train_cfg["use_augmentations"], device=base["device"],
                                       num_workers=0, sample_size=train_cfg["sample_size"], cache_dir=str(OUT / "cache"))
    base["pad_token_id"] = tok.pad_token_id
    cfg = {**base, **train_cfg}
    t0 = time.perf_counter()
    hist = trainer.train(model, tr, te, va, cfg, tokenizer=tok)
    wall = time.perf_counter() - t0
    crit = torch.nn.CrossEntropyLoss(ignore_index=tok.pad_token_id, label_smoothing=cfg["label_smoothing"])
    v1 = trainer.validate(model, va, crit, tok.pad_token_id, cfg["device"])
    v2 = trainer.validate(model, va, crit, tok.pad_token_id, cfg["device"])
    return dict(tokenizer=f"{type(tok).__name__} ({len(tok)} pieces)",
                params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                batches=dict(train=len(tr), val=len(va), test=len(te)), batch_size=train_cfg["batch_size"],
                epoch_wall_s=round(wall), train_step_s=round(wall / len(tr), 3),
                train_loss=round(hist["train_losses"][0], 4), train_acc=round(hist["train_accuracies"][0], 4),
                val_loss=round(hist["val_losses"][0], 4), val_acc=round(hist["val_accuracies"][0], 4),
                test_loss=round(hist["test_losses"][0], 4), test_acc=round(hist["test_accuracies"][0], 4),
                validate_twice=dict(first=[round(v, 5) for v in v1], second=[round(v, 5) for v in v2]),
                checkpoint="best_model.pt")


# --------------------------------------------------------------------------- generation
def bench_generation(device):
    from run_text_generation import run_generation_comparison
    ckpt = OUT / "checkpoints" / "best_model.pt"
    if not ckpt.exists():
        raise SystemExit("run --train-epoch first")
    run_generation_comparison(str(ckpt), None, device)
    times, sentences = {}, {}
    for f in sorted((OUT / "logs").glob("prompt_*_comparison.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for name, r in data["results"].items():
            times.setdefault(name, []).append(r["generation_time_seconds"])
            if name == "Greedy":
                sentences[data["prompt"]] = r["text"]
    return dict(checkpoint=ckpt.name, device=device,
                mean_seconds_per_prompt={k: round(sum(v) / len(v), 2) for k, v in times.items()},
                greedy_sentences=sentences)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for flag in ["all", "throughput", "moe-balance", "data-coverage", "train-epoch", "generation"]:
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.all or a.throughput:
        save("throughput", {"spm_vocab_8000": bench_throughput(SPM_VOCAB, a.device),
                            "yaml_vocab_50258": bench_throughput(50258, a.device),
                            "mt5_vocab_250100": bench_throughput(MT5_VOCAB, a.device)})
    if a.all or a.moe_balance:
        save("moe_balance", bench_moe_balance(a.device))
    if a.all or a.data_coverage:
        save("data_coverage", bench_data_coverage())
    if a.train_epoch:
        save("train_epoch", bench_train_epoch())
    if a.generation:
        save("generation", bench_generation(a.device))
