import torch
import os
import json
import shutil
import logging
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, batch_id, metrics=None, is_best=False, tokenizer=None):
    """
    Save model checkpoint with enhanced metadata and best model tracking.
    Also saves the tokenizer to ensure compatibility when loading the model.
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        batch_id: Current batch ID
        metrics: Dictionary of metrics to save (optional)
        is_best: Whether this is the best model so far (optional)
        tokenizer: The tokenizer used with the model (optional)
    """
    os.makedirs('checkpoints', exist_ok=True)
    
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'batch_id': batch_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'vocab_size': model.config["vocab_size"]  # Store vocab size explicitly
    }
    
    # Add metrics if provided
    if metrics:
        checkpoint['metrics'] = metrics
    
    checkpoint_filename = f'checkpoints/checkpoint_epoch_{epoch}_{batch_id}.pt'
    
    torch.save(checkpoint, checkpoint_filename)
    logger.info(f"Saved checkpoint: {checkpoint_filename}")
    
    # Save tokenizer if provided
    if tokenizer:
        tokenizer_filename = f'checkpoints/tokenizer_epoch_{epoch}_{batch_id}.pkl'
        with open(tokenizer_filename, 'wb') as f:
            pickle.dump(tokenizer, f)
        logger.info(f"Saved tokenizer: {tokenizer_filename}")
    
    # Save metadata separately for easy access
    metadata = {
        'epoch': epoch,
        'batch_id': batch_id,
        'timestamp': checkpoint['timestamp'],
        'vocab_size': checkpoint['vocab_size'],
    }
    if metrics:
        metadata['metrics'] = metrics
    
    metadata_filename = f'checkpoints/checkpoint_epoch_{epoch}_{batch_id}_metadata.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # If this is the best model, save a separate copy
    if is_best:
        best_model_path = 'checkpoints/best_model.pt'
        shutil.copyfile(checkpoint_filename, best_model_path)
        logger.info(f"Saved as best model: {best_model_path}")
        
        # Also save best model's tokenizer
        if tokenizer:
            best_tokenizer_path = 'checkpoints/best_model_tokenizer.pkl'
            shutil.copyfile(tokenizer_filename, best_tokenizer_path)
            logger.info(f"Saved best model tokenizer: {best_tokenizer_path}")
        
        # Also save best model metadata
        best_metadata_path = 'checkpoints/best_model_metadata.json'
        with open(best_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda', load_tokenizer=True):
    """
    Load model and optimizer state from checkpoint.
    Also loads the corresponding tokenizer if available.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        device: Device to load the model onto
        load_tokenizer: Whether to load the tokenizer if available
        
    Returns:
        Dictionary containing checkpoint data and tokenizer if available
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    
    # Check if vocab size matches
    if 'vocab_size' in checkpoint and checkpoint['vocab_size'] != model.config["vocab_size"]:
        logger.warning(f"Vocabulary size mismatch: checkpoint has {checkpoint['vocab_size']}, "
                     f"but model has {model.config['vocab_size']}. Updating model config.")
        model.config["vocab_size"] = checkpoint['vocab_size']
        # Reinitialize embedding to match checkpoint size
        model.embedding = torch.nn.Embedding(checkpoint['vocab_size'], model.config["hidden_dim"])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Try to load tokenizer if requested
    tokenizer = None
    if load_tokenizer:
        tokenizer_path = checkpoint_path.replace('.pt', '_tokenizer.pkl')
        if not os.path.exists(tokenizer_path):
            # Try alternative location patterns
            tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), 
                                         f"tokenizer_{os.path.basename(checkpoint_path).replace('checkpoint_', '').replace('.pt', '.pkl')}")
        
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
        else:
            logger.warning(f"No tokenizer found at {tokenizer_path}, using default T5Tokenizer")
            from transformers import T5Tokenizer
            tokenizer = T5Tokenizer.from_pretrained('google/mt5-base')
    
    # Log checkpoint info
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint, tokenizer

def get_latest_checkpoint():
    """
    Find the latest checkpoint in the checkpoints directory.
    
    Returns:
        Path to the latest checkpoint, or None if no checkpoints are found
    """
    if not os.path.exists('checkpoints'):
        return None
    
    # Find all checkpoint files
    checkpoint_files = [f for f in os.listdir('checkpoints') if f.endswith('.pt') and not f.startswith('best_model')]
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join('checkpoints', x)), reverse=True)
    
    return os.path.join('checkpoints', checkpoint_files[0])