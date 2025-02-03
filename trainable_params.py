import torch

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_trainable_parameters(model: torch.nn.Module, unit: str = "M") -> None:
    """
    Print the total number of trainable parameters in the model.
    
    Args:
        model: The PyTorch model.
        unit: The unit to display the parameters in. Options: "M" (millions), "B" (billions).
    """
    total_params = count_trainable_parameters(model)
    
    if unit == "M":
        print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
    elif unit == "B":
        print(f"Total trainable parameters: {total_params / 1e9:.2f}B")
    else:
        print(f"Total trainable parameters: {total_params}")