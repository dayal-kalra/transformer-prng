import math
import torch

# Set default device for computations
device = "cuda"

def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    """Calculate learning rate with warmup and cosine decay.
    
    Args:
        it (int): Current training iteration
        max_lr (float): Maximum learning rate
        min_lr (float): Minimum learning rate
        warmup_steps (int): Number of warmup steps
        max_steps (int): Total number of training steps
    
    Returns:
        float: Learning rate for current iteration
    """
    # Linear warmup phase
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    # After max steps, return minimum learning rate
    if it > max_steps:
        return min_lr
    
    # Cosine decay phase between warmup and max steps
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    # Cosine decay coefficient (goes from 1 to 0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)