import math

def linear_warmup_cosine(it, lr_init, lr_min, lr_max, warmup_steps, max_steps):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return (lr_max - lr_init) * (it+1) / warmup_steps + lr_init
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return lr_min
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return lr_min + coeff * (lr_max - lr_min)