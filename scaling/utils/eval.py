import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from .gpt2 import GPTConfig_abacus, GPT_oth_abacus
from .datasets import BaseBLCG


def load_pretrained_model(model_path, device=None):
    """Load a pre-trained BLCG model from checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint (.pth file)
        device (torch.device, optional): Device to load model on. If None, auto-detects best device.
    
    Returns:
        tuple: (model, config_params) where model is the loaded model and config_params is a dict
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Parse model configuration from filename
    filename = os.path.basename(model_path)
    config_params = _parse_model_config_from_filename(filename)
    
    print(f"Model Configuration:")
    for key, value in config_params.items():
        print(f"  {key:12}: {value}")
    
    # Create model configuration
    config = GPTConfig_abacus(
        block_size=config_params['context_len'],
        n_embd=config_params['n_embd'],
        n_head=config_params['n_head'],
        vocab_size=config_params['vocab_size'],
        n_layer=config_params['n_layer'],
        digits=config_params['digits']
    )
    
    # Initialize model
    model = GPT_oth_abacus(config).to(device)
    
    # Load pre-trained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle _orig_mod prefix from torch.compile()
    if any(key.startswith('_orig_mod.') for key in checkpoint.keys()):
        # print("Detected compiled model, removing _orig_mod prefixes...")
        clean_checkpoint = {}
        for key, value in checkpoint.items():
            if key.startswith('_orig_mod.'):
                new_key = key[10:]  # Remove '_orig_mod.' prefix
                clean_checkpoint[new_key] = value
            else:
                clean_checkpoint[key] = value
        checkpoint = clean_checkpoint
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"✅ Successfully loaded pre-trained model from:")
    print(f"   {model_path}")
    print(f"   Device: {device}")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model, config_params


def _parse_model_config_from_filename(filename):
    """Parse model configuration from filename using regex patterns.
    
    Expected filename format:
    blcg_abacus_n{n_embd}_h{n_head}_d{n_layer}_na{n_a}_nc{n_c}_ne{n_example}_p{p}_b{base}_d{digits}_sl{seq_len}_...
    
    Args:
        filename (str): Model filename containing configuration info
        
    Returns:
        dict: Configuration parameters extracted from filename
    """
    import re
    
    # Default configuration for BLCG abacus models
    config_params = {
        'n_embd': 768,       # Default embedding dimension
        'n_head': 6,         # Default number of attention heads  
        'n_layer': 2,        # Default number of layers
        'p': 65536,          # Default LCG modulus
        'base': 256,         # Default base for number representation
        'digits': 2,         # Default digits per number
        'seq_len': 513,      # Default sequence length
        'vocab_size': 256,   # Will be set to base
        'context_len': 1025  # Will be calculated from digits * seq_len - 1
    }
    
    try:
        # Extract model architecture parameters
        n_embd_match = re.search(r'_n(\d+)_', filename)
        if n_embd_match:
            config_params['n_embd'] = int(n_embd_match.group(1))
        
        n_head_match = re.search(r'_h(\d+)_', filename)
        if n_head_match:
            config_params['n_head'] = int(n_head_match.group(1))
        
        # Extract n_layer (first 'd' parameter after 'h')
        n_layer_match = re.search(r'_h\d+_d(\d+)_', filename)
        if n_layer_match:
            config_params['n_layer'] = int(n_layer_match.group(1))
        
        # Extract LCG parameters
        p_match = re.search(r'_p(\d+)_', filename)
        if p_match:
            config_params['p'] = int(p_match.group(1))
        
        base_match = re.search(r'_b(\d+)_', filename)
        if base_match:
            config_params['base'] = int(base_match.group(1))
            config_params['vocab_size'] = config_params['base']  # vocab_size = base
        
        # Extract digits (second 'd' parameter after 'b')
        digits_match = re.search(r'_b\d+_d(\d+)_', filename)
        if digits_match:
            config_params['digits'] = int(digits_match.group(1))
        
        # Extract sequence length
        seq_len_match = re.search(r'_sl(\d+)_', filename)
        if seq_len_match:
            config_params['seq_len'] = int(seq_len_match.group(1))
        
        # Calculate context_len from digits and seq_len
        config_params['context_len'] = config_params['digits'] * config_params['seq_len'] - 1
        
        # Validate that we have reasonable values
        assert config_params['n_embd'] > 0, f"Invalid n_embd: {config_params['n_embd']}"
        assert config_params['n_head'] > 0, f"Invalid n_head: {config_params['n_head']}"
        assert config_params['n_layer'] > 0, f"Invalid n_layer: {config_params['n_layer']}"
        assert config_params['p'] > 0, f"Invalid modulus: {config_params['p']}"
        assert config_params['base'] > 1, f"Invalid base: {config_params['base']}"
        assert config_params['digits'] > 0, f"Invalid digits: {config_params['digits']}"
        assert config_params['seq_len'] > 0, f"Invalid seq_len: {config_params['seq_len']}"
        
    except Exception as e:
        print(f"Warning: Error parsing filename '{filename}': {e}")
        print("Using default configuration values")
        # Recalculate context_len with defaults
        config_params['context_len'] = config_params['digits'] * config_params['seq_len'] - 1
    
    return config_params


def get_predictions(model, dataset, batch_size, device=None):
    """Get model predictions for a dataset.
    
    Args:
        model (torch.nn.Module): The model to evaluate
        dataset (torch.utils.data.Dataset): Dataset to get predictions for
        batch_size (int): Batch size for evaluation
        device (torch.device, optional): Device to use. If None, uses model's device.
    
    Returns:
        tuple: (ground_truth, predictions) as numpy arrays
    """
    if device is None:
        device = next(model.parameters()).device
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    predictions = []
    truth = []
    
    model.eval()
    with torch.no_grad():
        for x, y in train_dataloader:
            # Move data to appropriate device
            x, y = x.to(device), y.to(device)
            
            # Use device-appropriate autocast for inference
            if device.type == 'cuda':
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            elif device.type == 'mps':
                with torch.autocast(device_type="cpu", dtype=torch.float16):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
            
            # Get predicted classes
            preds = logits.argmax(dim=-1)
            predictions.append(preds)
            truth.append(y)
    
    # Concatenate all batches
    predictions = torch.cat(predictions, dim=0)
    truth = torch.cat(truth, dim=0)
    
    return truth.cpu().numpy(), predictions.cpu().numpy()


@torch.no_grad()
def predict_next_tokens(model, input_tokens, num_predictions=10, device=None):
    """Predict next tokens given input sequence.
    
    Args:
        model (torch.nn.Module): The model to use for prediction
        input_tokens (torch.Tensor): Input token sequence
        num_predictions (int): Number of tokens to predict
        device (torch.device, optional): Device to use. If None, uses model's device.
    
    Returns:
        list: List of predicted token IDs
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    input_tokens = input_tokens.to(device)
    
    # Handle batch dimension
    if input_tokens.dim() == 1:
        input_tokens = input_tokens.unsqueeze(0)
    
    predictions = []
    current_sequence = input_tokens.clone()
    
    for _ in range(num_predictions):
        # Forward pass with device-appropriate autocast
        if device.type == 'cuda':
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(current_sequence)
        elif device.type == 'mps':
            with torch.autocast(device_type="cpu", dtype=torch.float16):
                logits, _ = model(current_sequence)
        else:
            logits, _ = model(current_sequence) 
        
        # Get prediction for next token
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        
        predictions.append(next_token.item())
        
        # Update sequence (maintain context window)
        block_size = getattr(model.config, 'block_size', 1024)
        if current_sequence.size(1) >= block_size:
            current_sequence = torch.cat([
                current_sequence[:, 1:], 
                next_token.unsqueeze(0).unsqueeze(0)
            ], dim=1)
        else:
            current_sequence = torch.cat([
                current_sequence, 
                next_token.unsqueeze(0).unsqueeze(0)
            ], dim=1)
    
    return predictions 


def evaluate_test_accuracy(test_a, test_c, model, p=None, base=None, digits=None, 
                          seq_len=None, batch_size=8, n_example=1, rng=None, device=None):
    """Evaluate test accuracy on LCG sequences with given a and c parameters.
    
    Args:
        test_a (list/array): LCG multiplier values to test
        test_c (list/array): LCG additive constant values to test  
        model (torch.nn.Module): The trained model to evaluate
        p (int, optional): LCG modulus. If None, extracted from model config
        base (int, optional): Number base. If None, extracted from model config
        digits (int, optional): Digits per number. If None, extracted from model config
        seq_len (int, optional): Sequence length. If None, extracted from model config
        batch_size (int): Batch size for evaluation (default: 8)
        n_example (int): Number of examples per (a,c) pair (default: 1)
        rng (np.random.Generator, optional): Random number generator
        device (torch.device, optional): Device to use. If None, uses model's device
    
    Returns:
        dict: Dictionary containing:
            - 'accuracy': numpy array of per-position accuracies
            - 'mean_accuracy': float, mean accuracy across all positions
            - 'std_accuracy': float, standard deviation of accuracy
            - 'final_accuracy': float, accuracy at final position
    """
    if device is None:
        device = next(model.parameters()).device
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Try to extract config parameters from model if not provided
    if hasattr(model, 'config'):
        config = model.config
        if p is None:
            p = getattr(config, 'p', 65536)
        if base is None:
            base = getattr(config, 'vocab_size', 256)
        if digits is None:
            digits = getattr(config, 'digits', 2)
        if seq_len is None:
            # Calculate from block_size
            block_size = getattr(config, 'block_size', 1024)
            seq_len = (block_size + 1) // digits
    
    # Set defaults if still None
    if p is None:
        p = 65536
    if base is None:
        base = 256
    if digits is None:
        digits = 1
    if seq_len is None:
        seq_len = 513
    
    print(f"Evaluating test accuracy with:")
    print(f"  Test a values: {len(test_a)}")
    print(f"  Test c values: {len(test_c)}")
    print(f"  Parameters: p={p}, base={base}, digits={digits}, seq_len={seq_len}")
    
    # Create test dataset
    test_dataset = BaseBLCG(
        p=p, base=base, digits=digits, length=seq_len,
        a_list=test_a, c_list=test_c, rng=rng, num_examples=n_example
    )
    
    print(f"  Dataset size: {len(test_dataset)} sequences")
    
    # Get predictions
    test_truth, test_predictions = get_predictions(
        model=model, dataset=test_dataset, batch_size=batch_size, device=device
    )
    
    # Calculate accuracy (skip first few digits for prediction accuracy)
    test_pred = test_predictions[:, digits-1:]
    test_truth = test_truth[:, digits-1:]
    correct = test_truth == test_pred
    
    # Per-number accuracy (all digits must be correct)
    test_correct = correct[:, ::digits]
    for i in range(1, digits):
        test_correct &= correct[:, i::digits]
    
    # Calculate accuracy statistics
    accuracy = np.mean(test_correct, axis=0)
    mean_accuracy = accuracy.mean()
    std_accuracy = accuracy.std()
    final_accuracy = accuracy[-1] if len(accuracy) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy,
        'final_accuracy': final_accuracy
    } 