import torch
from torch.optim import AdamW
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
import os
import logging
from training.checkpointing import save_checkpoint
from visualization.metrics import plot_metrics
from transformers import get_linear_schedule_with_warmup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def compute_accuracy(logits, target_ids, pad_token_id):
    """Compute token-level prediction accuracy."""
    # Ignore pad tokens
    active_mask = target_ids != pad_token_id
    
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)
    
    # Calculate accuracy only on non-pad tokens
    correct = (predictions == target_ids) & active_mask
    total = active_mask.sum().item()
    
    if total > 0:
        return correct.sum().item() / total
    return 0.0

def validate(model, data_loader, criterion, pad_token_id, device):
    """Run validation on the provided data loader."""
    model.eval()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation", leave=False):
            input_ids, attention_mask, target_ids = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass - some models return a tuple in validation too
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Handle different return types
            if isinstance(outputs, tuple):
                logits = outputs[0]  # Main output logits
            else:
                logits = outputs
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            epoch_loss += loss.item()
            
            # Compute accuracy
            accuracy = compute_accuracy(logits, target_ids, pad_token_id)
            epoch_accuracy += accuracy
    
    # Calculate averages
    avg_loss = epoch_loss / num_batches
    avg_accuracy = epoch_accuracy / num_batches
    
    model.train()
    return avg_loss, avg_accuracy

def train(model, train_loader, test_loader, val_loader, config):
    """
    Training loop with improved handling for MTP outputs and stability enhancements.
    
    Args:
        model: The DeepSeek model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        val_loader: DataLoader for validation data
        config: Configuration dictionary with training parameters
    """
    logger.info("Starting training with configuration: %s", config)
    
    # Move model to the specified device
    device = config['device']
    model.to(device)
    
    # Set up optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_params = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_params,
        lr=config['learning_rate'],
        eps=1e-8
    )
    
    # Calculate total training steps for scheduler
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    total_steps = config['num_epochs'] * (len(train_loader) // gradient_accumulation_steps)
    
    # Create learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Set up loss function with label smoothing
    criterion = nn.CrossEntropyLoss(
        ignore_index=config.get('pad_token_id', -100),
        label_smoothing=0.1  # Add label smoothing for better generalization
    )
    
    # Create directories for checkpoints and logs if they don't exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Track metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # For early stopping
    best_val_loss = float('inf')
    patience = config.get('patience', 3)
    patience_counter = 0
    
    # Record start time
    start_time = time.time()
    
    # Main training loop
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        epoch_mtp_accuracy = 0.0
        
        # Track batch losses for this epoch
        batch_losses = []
        
        # Progress bar for batches
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{config['num_epochs']}",
            leave=True
        )
        
        # Accumulated gradients and loss
        accumulated_loss = 0.0
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, batch in progress_bar:
            # Unpack batch
            input_ids, attention_mask, target_ids = batch
            
            # Move tensors to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, target_ids=target_ids)
            
            # Handle different return types (main outputs and MTP outputs)
            if isinstance(outputs, tuple):
                logits, mtp_outputs = outputs
                
                # Compute main next-token prediction loss
                main_loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                # Compute MTP loss for future token predictions
                mtp_loss = 0.0
                if mtp_outputs is not None:
                    if len(mtp_outputs.shape) == 4:  # [batch, depth, seq_len, vocab_size]
                        depth = mtp_outputs.size(1)
                        for k in range(depth):
                            # Create target for next+k tokens (shift target by k+1)
                            shifted_target = torch.full_like(target_ids, config.get('pad_token_id', -100))
                            if k+1 < target_ids.size(1):
                                shifted_target[:, :-(k+1)] = target_ids[:, (k+1):]
                                
                                # Compute loss for this depth
                                depth_loss = criterion(
                                    mtp_outputs[:, k].contiguous().view(-1, mtp_outputs.size(-1)), 
                                    shifted_target.contiguous().view(-1)
                                )
                                mtp_loss += depth_loss
                        mtp_loss /= depth
                
                # Combine losses with weighting
                mtp_weight = config.get('mtp_weight', 0.5)
                total_loss = main_loss + mtp_weight * mtp_loss
                
                # Compute main accuracy
                main_accuracy = compute_accuracy(logits, target_ids, config.get('pad_token_id', -100))
                epoch_train_accuracy += main_accuracy
                
                # Log both losses
                batch_losses.append({
                    'main_loss': main_loss.item(),
                    'mtp_loss': mtp_loss.item() if isinstance(mtp_loss, torch.Tensor) else 0.0,
                    'total_loss': total_loss.item()
                })
            else:
                # Single output (no MTP)
                logits = outputs
                total_loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                # Compute accuracy
                main_accuracy = compute_accuracy(logits, target_ids, config.get('pad_token_id', -100))
                epoch_train_accuracy += main_accuracy
                
                # Log loss
                batch_losses.append({'total_loss': total_loss.item()})
            
            # Scale loss if using gradient accumulation
            if gradient_accumulation_steps > 1:
                total_loss = total_loss / gradient_accumulation_steps
            
            # Backward pass
            total_loss.backward()
            accumulated_loss += total_loss.item() * gradient_accumulation_steps
            
            # Update weights if needed
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=config.get('gradient_clip', 1.0)
                )
                
                # Update parameters
                optimizer.step()
                scheduler.step()  # Update learning rate
                optimizer.zero_grad()  # Reset gradients
                
                # Update progress bar with loss
                progress_bar.set_postfix({
                    'loss': accumulated_loss / gradient_accumulation_steps,
                    'acc': main_accuracy,
                    'lr': scheduler.get_last_lr()[0]
                })
                accumulated_loss = 0.0
            
            epoch_train_loss += total_loss.item() * gradient_accumulation_steps
            
            # Validation during training (every N steps)
            eval_steps = config.get('eval_steps', len(train_loader) // 5)  # Default: 5 times per epoch
            if (batch_idx + 1) % eval_steps == 0:
                # Run validation
                val_loss, val_accuracy = validate(
                    model, val_loader, criterion, 
                    config.get('pad_token_id', -100), device
                )
                
                # Log validation results
                logger.info(
                    f"Step {batch_idx+1}/{len(train_loader)}: "
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
                )
                
                # Save checkpoint if it's the best model so far
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, epoch, batch_idx, 
                                   metrics={'val_loss': val_loss, 'val_accuracy': val_accuracy},
                                   is_best=True)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                    # Check for early stopping
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {patience} validations without improvement")
                        # Save final checkpoint
                        save_checkpoint(model, optimizer, epoch, batch_idx,
                                      metrics={'val_loss': val_loss, 'val_accuracy': val_accuracy})
                        break
                
                model.train()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_accuracy = epoch_train_accuracy / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        
        logger.info(
            f"Epoch {epoch+1}/{config['num_epochs']} completed. "
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}"
        )
        
        val_loss, val_accuracy = validate(
            model, val_loader, criterion, 
            config.get('pad_token_id', -100), device
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        logger.info(
            f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}"
        )
        
        test_loss, test_accuracy = validate(
            model, test_loader, criterion, 
            config.get('pad_token_id', -100), device
        )
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        logger.info(
            f"Test - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}"
        )
        
        save_checkpoint(
            model, optimizer, epoch, len(train_loader) - 1,
            metrics={
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'train_accuracy': avg_train_accuracy,
                'val_accuracy': val_accuracy,
                'test_accuracy': test_accuracy
            }
        )
        
        # Check for early stopping at epoch level
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(
        f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s"
    )
    
    # Plot final metrics
    plot_metrics(
        train_losses, val_losses, test_losses, 
        train_accuracies, val_accuracies, test_accuracies
    )
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_accuracies': test_accuracies
    }

