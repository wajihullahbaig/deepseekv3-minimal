import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(train_losses, val_losses, test_losses, 
                train_accuracies=None, val_accuracies=None, test_accuracies=None):
    """
    Plot training, validation, and test metrics (loss and accuracy).
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        test_losses: List of test losses per epoch
        train_accuracies: List of training accuracies per epoch (optional)
        val_accuracies: List of validation accuracies per epoch (optional)
        test_accuracies: List of test accuracies per epoch (optional)
    """
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Create figure with two subplots if accuracy data is provided
    if any(acc is not None for acc in [train_accuracies, val_accuracies, test_accuracies]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss')
    ax1.plot(epochs, val_losses, 'g-', marker='s', label='Validation Loss')
    ax1.plot(epochs, test_losses, 'r-', marker='^', label='Test Loss')
    
    # Add moving average for smoothing
    if len(train_losses) > 3:
        window_size = min(5, len(train_losses) // 3)
        train_ma = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        val_ma = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
        test_ma = np.convolve(test_losses, np.ones(window_size)/window_size, mode='valid')
        
        ma_epochs = range(window_size, len(train_losses) + 1)
        ax1.plot(ma_epochs, train_ma, 'b--', alpha=0.5, label='Training MA')
        ax1.plot(ma_epochs, val_ma, 'g--', alpha=0.5, label='Validation MA')
        ax1.plot(ma_epochs, test_ma, 'r--', alpha=0.5, label='Test MA')
    
    # Configure loss plot
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    
    # Add loss values at endpoints
    for i, loss_list in enumerate([train_losses, val_losses, test_losses]):
        if loss_list:
            color = ['blue', 'green', 'red'][i]
            name = ['Train', 'Val', 'Test'][i]
            ax1.text(epochs[-1], loss_list[-1], f'{name}: {loss_list[-1]:.4f}', 
                    color=color, fontweight='bold', ha='right', va='bottom')
    
    # Plot accuracies if provided
    if any(acc is not None for acc in [train_accuracies, val_accuracies, test_accuracies]):
        if train_accuracies:
            ax2.plot(epochs, train_accuracies, 'b-', marker='o', label='Training Accuracy')
        if val_accuracies:
            ax2.plot(epochs, val_accuracies, 'g-', marker='s', label='Validation Accuracy')
        if test_accuracies:
            ax2.plot(epochs, test_accuracies, 'r-', marker='^', label='Test Accuracy')
        
        # Add moving average for smoothing if enough data points
        if train_accuracies and len(train_accuracies) > 3:
            window_size = min(5, len(train_accuracies) // 3)
            
            if train_accuracies:
                train_acc_ma = np.convolve(train_accuracies, np.ones(window_size)/window_size, mode='valid')
                ax2.plot(ma_epochs, train_acc_ma, 'b--', alpha=0.5, label='Training Acc MA')
            
            if val_accuracies:
                val_acc_ma = np.convolve(val_accuracies, np.ones(window_size)/window_size, mode='valid')
                ax2.plot(ma_epochs, val_acc_ma, 'g--', alpha=0.5, label='Validation Acc MA')
            
            if test_accuracies:
                test_acc_ma = np.convolve(test_accuracies, np.ones(window_size)/window_size, mode='valid')
                ax2.plot(ma_epochs, test_acc_ma, 'r--', alpha=0.5, label='Test Acc MA')
        
        # Configure accuracy plot
        ax2.set_title('Accuracy vs. Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='lower right')
        
        # Add accuracy values at endpoints
        for i, acc_list in enumerate([train_accuracies, val_accuracies, test_accuracies]):
            if acc_list:
                color = ['blue', 'green', 'red'][i]
                name = ['Train', 'Val', 'Test'][i]
                ax2.text(epochs[-1], acc_list[-1], f'{name}: {acc_list[-1]:.4f}', 
                        color=color, fontweight='bold', ha='right', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('plots/training_metrics.png', dpi=300, bbox_inches='tight')
    
    # Save loss and accuracy separately as well
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss')
    plt.plot(epochs, val_losses, 'g-', marker='s', label='Validation Loss')
    plt.plot(epochs, test_losses, 'r-', marker='^', label='Test Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/loss_plot.png', dpi=300, bbox_inches='tight')
    
    if all(acc is not None for acc in [train_accuracies, val_accuracies, test_accuracies]):
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_accuracies, 'b-', marker='o', label='Training Accuracy')
        plt.plot(epochs, val_accuracies, 'g-', marker='s', label='Validation Accuracy')
        plt.plot(epochs, test_accuracies, 'r-', marker='^', label='Test Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('plots/accuracy_plot.png', dpi=300, bbox_inches='tight')
    
    plt.close('all')

def plot_learning_rate(learning_rates, step_numbers):
    """
    Plot learning rate schedule.
    
    Args:
        learning_rates: List of learning rates
        step_numbers: List of corresponding step numbers
    """
    plt.figure(figsize=(10, 5))
    plt.plot(step_numbers, learning_rates, 'b-')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('plots/learning_rate_schedule.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_expert_activation(expert_activations, layer_idx=None):
    """
    Plot expert activation counts.
    
    Args:
        expert_activations: Dictionary mapping expert index to activation count
        layer_idx: Layer index (optional, for multi-layer visualization)
    """
    plt.figure(figsize=(12, 5))
    
    experts = list(expert_activations.keys())
    counts = list(expert_activations.values())
    
    # Sort by expert index
    sorted_experts = sorted(zip(experts, counts), key=lambda x: x[0])
    experts, counts = zip(*sorted_experts)
    
    # Plot as bar chart
    bars = plt.bar(experts, counts, alpha=0.7)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    # Configure plot
    title = f'Expert Activation Counts (Layer {layer_idx})' if layer_idx is not None else 'Expert Activation Counts'
    plt.title(title)
    plt.xlabel('Expert Index')
    plt.ylabel('Activation Count')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save
    filename = f'plots/expert_activation_layer_{layer_idx}.png' if layer_idx is not None else 'plots/expert_activation.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_patterns(attention_weights, head_idx=0, layer_idx=0):
    """
    Plot attention patterns for visualization.
    
    Args:
        attention_weights: Tensor of attention weights [batch, heads, seq_len, seq_len]
        head_idx: Attention head to visualize
        layer_idx: Layer to visualize
    """
    # Get attention weights for specified head
    attn = attention_weights[0, head_idx].cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.savefig(f'plots/attention_map_L{layer_idx}_H{head_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()