import torch
import os

def save_checkpoint(model, optimizer, epoch, batch_id):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(checkpoint, f'checkpoints/checkpoint_epoch_{epoch}_{batch_id}.pt')