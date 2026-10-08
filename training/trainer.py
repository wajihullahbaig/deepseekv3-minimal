import logging
import os
import time

import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from training.checkpointing import save_checkpoint
from visualization.metrics import plot_metrics

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/training.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("TRAINER")


def compute_accuracy(logits, target_ids, pad_token_id):
    """Token-level accuracy over non-pad targets."""
    active_mask = target_ids != pad_token_id
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == target_ids) & active_mask
    total = active_mask.sum().item()
    return correct.sum().item() / total if total > 0 else 0.0


def validate(model, data_loader, criterion, pad_token_id, device):
    """Mean loss and accuracy over a loader. Leaves the model in train mode."""
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for input_ids, attention_mask, target_ids in tqdm(data_loader, desc="Validation", leave=False):
            input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            total_loss += criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)).item()
            total_acc += compute_accuracy(logits, target_ids, pad_token_id)
            n += 1
    model.train()
    return total_loss / max(n, 1), total_acc / max(n, 1)


def get_optimizer(model, config):
    """AdamW / SGD with weight decay from the config; biases and LayerNorm params are not decayed."""
    weight_decay = config.get("weight_decay", 0.01)
    no_decay = {id(p) for m in model.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters()}
    no_decay |= {id(p) for n, p in model.named_parameters() if n.endswith(".bias")}
    groups = [
        {"params": [p for p in model.parameters() if id(p) not in no_decay], "weight_decay": weight_decay},
        {"params": [p for p in model.parameters() if id(p) in no_decay], "weight_decay": 0.0},
    ]
    optimizer_type = config.get("optimizer", "sgd").lower()
    if optimizer_type == "sgd":
        return SGD(groups, lr=config["learning_rate"], momentum=config.get("momentum", 0.9),
                   nesterov=config.get("nesterov", True))
    if optimizer_type == "adamw":
        return AdamW(groups, lr=config["learning_rate"], eps=1e-8)
    raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def get_scheduler(optimizer, train_loader, config):
    """All schedulers are sized in optimizer steps and must be stepped once per optimizer step."""
    scheduler_type = config.get("scheduler", "cosine").lower()
    accum = config.get("gradient_accumulation_steps", 1)
    steps_per_epoch = max(1, -(-len(train_loader) // accum))
    total_steps = config["num_epochs"] * steps_per_epoch
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.get("min_lr", 1e-6))
    if scheduler_type == "cosine_warmup":
        return CosineAnnealingWarmRestarts(optimizer, T_0=config.get("restart_epochs", 10) * steps_per_epoch,
                                           T_mult=config.get("restart_mult", 2), eta_min=config.get("min_lr", 1e-6))
    if scheduler_type == "linear_warmup":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config["warmup_steps"],
                                               num_training_steps=total_steps)
    raise ValueError(f"Unsupported scheduler: {scheduler_type}")


def mtp_loss(mtp_outputs, target_ids, criterion, pad_token_id):
    """Depth k of the MTP head predicts target[t + k + 1]; positions without a target are ignored."""
    depth = mtp_outputs.size(1)
    total = 0.0
    for k in range(depth):
        shifted = torch.full_like(target_ids, pad_token_id)
        if k + 1 < target_ids.size(1):
            shifted[:, :-(k + 1)] = target_ids[:, (k + 1):]
        total = total + criterion(mtp_outputs[:, k].reshape(-1, mtp_outputs.size(-1)), shifted.reshape(-1))
    return total / depth


def log_samples(model, tokenizer, prompts, device, max_length=40):
    """Generate a few sentences with greedy and sampled decoding and log them."""
    from text_generation import GenerationConfig, TextGenerator

    generator = TextGenerator(model, tokenizer, device)
    common = dict(max_length=max_length, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                  use_mtp=False, mtp_speculation_mode=False)
    configs = {"greedy": GenerationConfig(do_sample=False, **common),
               "sampling(t=0.8)": GenerationConfig(do_sample=True, temperature=0.8, top_k=50, top_p=0.95, **common)}
    logger.info("Sample generations:")
    for prompt in prompts:
        for name, cfg in configs.items():
            logger.info("  [%s] %r -> %r", name, prompt, generator.generate(prompt, cfg))
    model.train()


def train(model, train_loader, test_loader, val_loader, config, tokenizer=None):
    """Training loop with MTP loss, gradient accumulation, step/epoch validation and early stopping."""
    logger.info("Starting training with configuration: %s", {k: v for k, v in config.items() if k != "sample_prompts"})
    device = config["device"]
    pad_token_id = config.get("pad_token_id", -100)
    model.to(device)

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, train_loader, config)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=config.get("label_smoothing", 0.1))
    os.makedirs("checkpoints", exist_ok=True)

    steps_per_epoch = len(train_loader)
    accum = config.get("gradient_accumulation_steps", 1)
    mtp_weight = config.get("mtp_weight", 0.3)
    eval_steps = config.get("eval_steps") or max(1, steps_per_epoch // 5)
    eval_epochs = config.get("eval_nth_epoch", 5)
    patience = config.get("patience", 3)
    min_delta = config.get("early_stop_min_delta", 0.0)

    history = {k: [] for k in ("train_losses", "val_losses", "test_losses",
                               "train_accuracies", "val_accuracies", "test_accuracies")}
    best_val_loss = float("inf")
    patience_counter = 0
    stop = False
    start_time = time.time()

    for epoch in range(config["num_epochs"]):
        model.train()
        sum_loss = sum_acc = 0.0
        batches_seen = 0
        optimizer.zero_grad()
        progress = tqdm(enumerate(train_loader), total=steps_per_epoch,
                        desc=f"Epoch {epoch + 1}/{config['num_epochs']}", leave=True)

        for batch_idx, (input_ids, attention_mask, target_ids) in progress:
            input_ids, attention_mask, target_ids = input_ids.to(device), attention_mask.to(device), target_ids.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, target_ids=target_ids)
            if isinstance(outputs, tuple):
                logits, mtp_outputs = outputs
                main_loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
                total_loss = (1.0 - mtp_weight) * main_loss + mtp_weight * mtp_loss(mtp_outputs, target_ids, criterion, pad_token_id)
            else:
                logits = outputs
                total_loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

            (total_loss / accum).backward()
            accuracy = compute_accuracy(logits, target_ids, pad_token_id)
            sum_loss += total_loss.item()
            sum_acc += accuracy
            batches_seen += 1

            if (batch_idx + 1) % accum == 0 or batch_idx + 1 == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get("gradient_clip", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.set_postfix({"loss": f"{total_loss.item():.3f}", "acc": f"{accuracy:.3f}",
                                      "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            if (batch_idx + 1) % eval_steps == 0 and (epoch + 1) % eval_epochs == 0 and batch_idx + 1 < steps_per_epoch:
                val_loss, val_accuracy = validate(model, val_loader, criterion, pad_token_id, device)
                logger.info(f"Step {batch_idx + 1}/{steps_per_epoch}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(model, optimizer, epoch, batch_idx, tokenizer=tokenizer,
                                    metrics={"val_loss": val_loss, "val_accuracy": val_accuracy}, is_best=True)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {patience} validations without improvement")
                        stop = True
                        break

        avg_train_loss = sum_loss / max(batches_seen, 1)
        avg_train_accuracy = sum_acc / max(batches_seen, 1)
        history["train_losses"].append(avg_train_loss)
        history["train_accuracies"].append(avg_train_accuracy)
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']} completed ({batches_seen} batches). "
                    f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}")

        val_loss, val_accuracy = validate(model, val_loader, criterion, pad_token_id, device)
        test_loss, test_accuracy = validate(model, test_loader, criterion, pad_token_id, device)
        for k, v in (("val_losses", val_loss), ("val_accuracies", val_accuracy),
                     ("test_losses", test_loss), ("test_accuracies", test_accuracy)):
            history[k].append(v)
        logger.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        logger.info(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

        improved = val_loss < best_val_loss - min_delta
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
        save_checkpoint(model, optimizer, epoch, steps_per_epoch - 1, tokenizer=tokenizer, is_best=improved,
                        metrics={"train_loss": avg_train_loss, "val_loss": val_loss, "test_loss": test_loss,
                                 "train_accuracy": avg_train_accuracy, "val_accuracy": val_accuracy,
                                 "test_accuracy": test_accuracy})
        if not improved:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                stop = True
        if stop:
            break

    total = time.time() - start_time
    logger.info(f"Training completed in {int(total // 3600)}h {int(total % 3600 // 60)}m {total % 60:.2f}s "
                f"(best val loss {best_val_loss:.4f})")
    plot_metrics(history["train_losses"], history["val_losses"], history["test_losses"],
                 history["train_accuracies"], history["val_accuracies"], history["test_accuracies"])

    if tokenizer is not None and config.get("sample_prompts"):
        log_samples(model, tokenizer, config["sample_prompts"], device)
    return history
