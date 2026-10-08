import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from conftest import freeze_router
from models.deepseek_v3 import DeepSeekV3
from training import trainer
from training.trainer import compute_accuracy, get_optimizer, get_scheduler


def _loader(cfg, n_batches=8, bs=4, seq=16, seed=0, repeat_one_batch=False):
    g = torch.Generator().manual_seed(seed)
    n = bs if repeat_one_batch else n_batches * bs
    ids = torch.randint(2, cfg["vocab_size"], (n, seq + 1), generator=g)
    if repeat_one_batch:
        ids = ids.repeat(n_batches, 1)
    x, y = ids[:, :-1], ids[:, 1:]
    return DataLoader(TensorDataset(x, torch.ones_like(x), y), batch_size=bs, shuffle=False)


def _train_cfg(**overrides):
    cfg = dict(device="cpu", pad_token_id=0, batch_size=4, learning_rate=3e-3, num_epochs=3,
               optimizer="adamw", scheduler="cosine", min_lr=1e-4, gradient_clip=1.0,
               label_smoothing=0.0, gradient_accumulation_steps=1, patience=10,
               eval_steps=10_000, eval_nth_epoch=1, mtp_weight=0.2)
    cfg.update(overrides)
    return cfg


def test_compute_accuracy_ignores_pad_targets():
    logits = torch.zeros(1, 4, 5)
    logits[0, :, 3] = 1.0                                          # always predicts token 3
    targets = torch.tensor([[3, 3, 2, 0]])                         # last target is pad
    assert compute_accuracy(logits, targets, pad_token_id=0) == pytest.approx(2 / 3)


def test_train_smoke_runs_end_to_end(tiny_config, stub_tokenizer, tmp_path, monkeypatch, caplog):
    caplog.set_level("INFO")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    torch.manual_seed(0)
    model = DeepSeekV3(tiny_config)
    dl = _loader(tiny_config)
    hist = trainer.train(model, dl, dl, dl, _train_cfg(sample_prompts=["t5 t9"]), tokenizer=stub_tokenizer)
    assert len(hist["train_losses"]) == len(hist["val_losses"]) == len(hist["test_losses"]) == 3
    assert all(torch.isfinite(torch.tensor(hist["train_losses"])))
    assert hist["train_losses"][-1] < hist["train_losses"][0]     # it can fit a fixed tiny set
    assert (tmp_path / "checkpoints" / "checkpoint_epoch_2_7.pt").exists()
    assert (tmp_path / "checkpoints" / "checkpoint_epoch_2_7_metadata.json").exists()
    assert (tmp_path / "plots" / "training_metrics.png").exists()
    assert (tmp_path / "checkpoints" / "best_model.pt").exists()
    assert any("[greedy] 't5 t9' ->" in r.message for r in caplog.records)   # sentences logged at the end


def test_linear_warmup_scheduler_anneals_to_zero(tiny_model, tiny_config):
    dl = _loader(tiny_config, n_batches=50)
    cfg = dict(optimizer="adamw", learning_rate=1e-3, num_epochs=10, scheduler="linear_warmup",
               warmup_steps=5, gradient_accumulation_steps=1)
    opt = get_optimizer(tiny_model, cfg)
    sched = get_scheduler(opt, dl, cfg)
    for _ in range(10 * 50):                                       # trainer steps it per optimizer step
        opt.step()
        sched.step()
    assert sched.get_last_lr()[0] < 1e-6


def test_cosine_scheduler_anneals_to_min_lr(tiny_model, tiny_config):
    dl = _loader(tiny_config, n_batches=50)
    cfg = dict(optimizer="sgd", learning_rate=1e-3, num_epochs=10, scheduler="cosine", min_lr=1e-5,
               gradient_accumulation_steps=1)
    opt = get_optimizer(tiny_model, cfg)
    s = get_scheduler(opt, dl, cfg)
    for _ in range(10 * 50):                                       # trainer: one scheduler.step() per optimizer step
        opt.step()
        s.step()
    assert s.get_last_lr()[0] == pytest.approx(1e-5, rel=1e-3)


def test_cosine_warm_restarts_scheduler_cycles_in_epochs(tiny_model, tiny_config):
    dl = _loader(tiny_config, n_batches=50)
    cfg = dict(optimizer="sgd", learning_rate=1e-3, num_epochs=10, scheduler="cosine_warmup", min_lr=1e-5,
               restart_epochs=5, restart_mult=1, gradient_accumulation_steps=1)
    opt = get_optimizer(tiny_model, cfg)
    s = get_scheduler(opt, dl, cfg)
    for _ in range(125):                                           # half of the first 5-epoch cycle
        opt.step()
        s.step()
    assert 0.4e-3 < s.get_last_lr()[0] < 0.6e-3
    for _ in range(125):
        opt.step()
        s.step()
    assert s.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-3)   # restart at epoch 5


def test_layernorm_weights_are_excluded_from_weight_decay(tiny_model):
    opt = get_optimizer(tiny_model, {"optimizer": "adamw", "learning_rate": 1e-3})
    ln = {id(m.weight) for m in tiny_model.modules() if isinstance(m, torch.nn.LayerNorm)}
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    assert ln.isdisjoint(decayed)


def test_weight_decay_comes_from_config(tiny_model):
    opt = get_optimizer(tiny_model, {"optimizer": "adamw", "learning_rate": 1e-3, "weight_decay": 0.0})
    assert all(g["weight_decay"] == 0.0 for g in opt.param_groups)


def test_epoch_average_is_correct_after_an_early_break(tiny_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    def run(**overrides):
        torch.manual_seed(0)
        model = DeepSeekV3(tiny_config)
        freeze_router(model)                                       # keep the model exactly constant
        dl = _loader(tiny_config, n_batches=8, repeat_one_batch=True)
        cfg = _train_cfg(learning_rate=0.0, min_lr=0.0, optimizer="sgd", num_epochs=1, **overrides)
        return trainer.train(model, dl, dl, dl, cfg)["train_losses"][0]

    full = run()                                                   # whole epoch, 8 identical batches
    broken = run(eval_steps=2, patience=1)                         # early stop after batch 4 of 8
    assert broken == pytest.approx(full, rel=1e-3)
