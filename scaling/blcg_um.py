# Standard library imports
import os
import math
import time
import argparse


# Third-party imports
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import pandas as pd
from sympy import isprime

# Custom imports
from utils.prng_data import find_as, find_coprimes
from utils.datasets import BaseBLCG
from utils.gpt2 import GPT_oth_abacus, GPTConfig_abacus
from utils.optimize import get_predictions

def generate_powers_in_range(numbers, min_val, max_val):
    """Generate powers of given numbers within a specified range."""
    result = []
    for num in numbers:
        power = 1
        while (value := num ** power) <= max_val:
            if value >= min_val:
                result.append(value)
            power += 1
    return result

def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    """Calculate learning rate using warmup and cosine decay schedule."""
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def create_datasets(config, rng):
    """Create train and test datasets."""
    # Generate test set
    config.prime = [int(x) for x in config.prime.split(',')]
    config.test_p = generate_powers_in_range(config.prime, config.p_min, config.p_max)
    print(f"{len(config.test_p)} test p: {config.test_p}")
    
    # Create test datasets
    test_datasets = []
    val_a_set = set()
    val_c_set = set()
    for p in config.test_p:
        # Check if p is prime and if a file with predefined values exists
        if isprime(p) and os.path.exists(f"results/period_analysis/random_select_p{p}_T{p-1}.txt"):
            # Read values from file
            print(f"reading values from file for p={p}")
            val_a, val_c = read_ac_from_file(p)
        else:
            # Generate values randomly as before
            val_a = find_as(p, rng=rng, num=config.n_test_a)
            val_c = find_coprimes(p, rng=rng, num=config.n_test_c)
        print(f"for p={p}:")
        print(f"a = {' '.join(map(str, val_a))}")
        print(f"c = {' '.join(map(str, val_c))}")
        test_dataset = BaseBLCG(p=p, base=config.base, digits=config.digits, 
                              length=config.seq_len, a_list=val_a, c_list=val_c,
                              rng=rng, num_examples=config.n_example)
        test_datasets.append(test_dataset)
        val_a_set.update(val_a)
        val_c_set.update(val_c)
    
    # Create training datasets
    train_a = [i for i in range(1, config.p_max) if i not in val_a_set]
    train_c = [i for i in range(1, config.p_max) if i not in val_c_set]
    ps = [i for i in range(config.p_min, config.p_max+1) if i not in config.test_p]
    
    if config.n_p:
        ps = rng.choice(ps, min(config.n_p, len(ps)), replace=False)
        ps = np.sort(ps)
    
    train_datasets = []
    all_train_a = set()
    all_train_c = set()
    
    for p in ps:
        temp_a = find_as(p, rng=rng, num=config.n_a)
        temp_a = [x for x in temp_a if x not in val_a_set]
        temp_c = find_coprimes(p, rng=rng, num=config.n_c)
        temp_c = [x for x in temp_c if x not in val_c_set]
        
        if len(temp_a) < config.n_a:
            temp_a = np.concatenate((temp_a, rng.choice(train_a, config.n_a-len(temp_a), replace=False)))
        if len(temp_c) < config.n_c:
            temp_c = np.concatenate((temp_c, rng.choice(train_c, config.n_c-len(temp_c), replace=False)))
            
        if len(temp_a) > 0 and len(temp_c) > 0:
            train_datasets.append(BaseBLCG(p=p, base=config.base, digits=config.digits,
                                         length=config.seq_len, a_list=temp_a, c_list=temp_c,
                                         rng=rng, num_examples=config.n_example))
            all_train_a.update(temp_a)
            all_train_c.update(temp_c)
    
    return (ConcatDataset(train_datasets), ConcatDataset(test_datasets), 
            all_train_a, all_train_c)

def read_ac_from_file(p):
    """Read a and c values from a file for a given prime p."""
    file_path = f"results/period_analysis/random_select_p{p}_T{p-1}.txt"
    a_values = []
    c_values = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('a ='):
                a_values = [int(x) for x in line.replace('a =', '').strip().split()]
            elif line.startswith('c ='):
                c_values = [int(x) for x in line.replace('c =', '').strip().split()]
    
    return a_values, c_values

def train_model(model, train_loader, test_loader, optimizer, config, device):
    """Train the model and evaluate periodically."""
    eval_results = []
    train_loss = train_acc = train_last_acc = 0
    step = 0
    t0 = time.time()
    
    num_epochs = int(np.ceil(config.num_steps / len(train_loader)))
    model.train()
    
    for epoch in range(num_epochs):
        for x, y in train_loader:
            # Training step
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            
            # Update metrics
            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_acc += torch.sum(preds == y).item()
            train_last_acc += torch.sum(torch.all(preds[:,-config.digits:]==y[:,-config.digits:], dim=1)).item()
            
            # Optimization step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(step, config.lr_trgt, config.lr_min, config.warm_steps, config.num_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            step += 1
            
            # Evaluation
            if step % config.eval_interval == 0:
                eval_metrics = evaluate_model(model, train_loader, test_loader, 
                                           train_loss, train_acc, train_last_acc,
                                           config, step, epoch, lr, device)
                eval_results.append(eval_metrics)
                train_loss = train_acc = train_last_acc = 0
                

            if step >= config.num_steps:
                break
        if step >= config.num_steps:
            break
    
    print(f"Training completed in {time.time()-t0:.2f} seconds, {step} steps")
    return eval_results

def evaluate_model(model, train_loader, test_loader, train_loss, train_acc, 
                  train_last_acc, config, step, epoch, lr, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = test_acc = test_last_acc = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            test_loss += loss.item()
            preds = logits.argmax(dim=-1)
            test_acc += torch.sum(preds == y).item()
            test_last_acc += torch.sum(torch.all(preds[:,-config.digits:]==y[:,-config.digits:], dim=1)).item()
    
    # Normalize metrics
    test_loss = test_loss / len(test_loader) 
    train_loss = train_loss / config.eval_interval
    test_acc = test_acc / len(test_loader) / config.batch_size / config.context_len
    train_acc = train_acc / config.eval_interval / config.batch_size / config.context_len
    test_last_acc = test_last_acc / len(test_loader) / config.batch_size
    train_last_acc = train_last_acc / config.eval_interval / config.batch_size

    print(f"step {step:4d} | epoch {epoch} | lr {lr:.4e} | "
          f"train loss: {train_loss:.6f} | test loss: {test_loss:.6f} | "
          f"train acc: {train_acc:.4f} | test acc: {test_acc:.4f} | "
          f"train last acc: {train_last_acc:.4f} | test last acc: {test_last_acc:.4f}")
    
    model.train()
    return [step, lr, train_loss, test_loss, train_acc, test_acc, train_last_acc, test_last_acc]

def save_results(eval_results, model, train_dataset, test_dataset, all_train_a, all_train_c, config):
    """Save evaluation results, model parameters, and sequences."""
    # Create results directory if it doesn't exist
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Save evaluation results
    df_eval = pd.DataFrame(eval_results, columns=['step', 'lr', 'train_loss', 'test_loss', 
                                                 'train_acc', 'test_acc', 'train_last_acc', 
                                                 'test_last_acc'])
    eval_path = (f'{config.results_dir}/eval_b{config.base}_d{config.digits}_'+
                f'prime{"-".join(map(str, config.prime))}_pmax{config.p_max}_'+
                f'pmin{config.p_min}_sl{config.seq_len}_np{config.n_p}_'+
                f'na{config.n_a}_nc{config.n_c}_ne{config.n_example}_'+
                f'n{config.n_embd}_h{config.n_head}_d{config.n_layer}_'+
                f'ds{config.data_seed}_I{config.main_seed}_'+
                f'lr{config.lr_trgt:0.6f}_wd{config.weight_decay}_'+
                f'Twarm{config.warm_steps}_T{config.num_steps}_B{config.batch_size}')
    df_eval.to_csv(eval_path, sep='\t')

    # Save model parameters and training data if requested
    if config.save_params == "True":
        params_path = (f'{config.results_dir}/params_b{config.base}_d{config.digits}_'
                      f'prime{"-".join(map(str, config.prime))}_pmax{config.p_max}_'
                      f'pmin{config.p_min}_sl{config.seq_len}_np{config.n_p}_'
                      f'na{config.n_a}_nc{config.n_c}_ne{config.n_example}_'
                      f'n{config.n_embd}_h{config.n_head}_d{config.n_layer}_'
                      f'ds{config.data_seed}_I{config.main_seed}_'
                      f'lr{config.lr_trgt:0.6f}_Twarm{config.warm_steps}_'
                      f'T{config.num_steps}_B{config.batch_size}_'
                      f'wd{config.weight_decay}.pth')
        
        # Fix for compiled model: get the original model's state dict
        if hasattr(model, '_orig_mod'):
            # For compiled model, use the original module's state dict
            state_dict = model._orig_mod.state_dict()
        else:
            # For non-compiled model, use the model's state dict directly
            state_dict = model.state_dict()
            
        torch.save(state_dict, params_path)
        
        ac_path = (f'{config.results_dir}/trainac_ds{config.data_seed}_I{config.main_seed}_na{config.n_a}_nc{config.n_c}_ne{config.n_example}')
        np.savez(ac_path, train_a=np.array(list(all_train_a)), 
                 train_c=np.array(list(all_train_c)))

    # Save sequences if requested
    if config.save_sequences == "True":
        model.eval()
        # Sample subset of training data
        indices = torch.randperm(len(train_dataset))[:4096]
        subset_dataset = Subset(train_dataset, indices)
        
        # Get predictions
        train_truth, train_predictions = get_predictions(
            model=model, dataset=subset_dataset, batch_size=config.batch_size)
        test_truth, test_predictions = get_predictions(
            model=model, dataset=test_dataset, batch_size=config.batch_size)
        
        # Save predictions
        base_path = (f'{config.results_dir}/{{}}_{config.base}_d{config.digits}_'
                    f'prime{"-".join(map(str, config.prime))}_pmax{config.p_max}_'
                    f'pmin{config.p_min}_sl{config.seq_len}_np{config.n_p}_'
                    f'na{config.n_a}_nc{config.n_c}_ne{config.n_example}_'
                    f'n{config.n_embd}_h{config.n_head}_d{config.n_layer}_'
                    f'ds{config.data_seed}_I{config.main_seed}_'
                    f'lr{config.lr_trgt:0.6f}_Twarm{config.warm_steps}_'
                    f'T{config.num_steps}_B{config.batch_size}_'
                    f'wd{config.weight_decay}.npy')
        
        np.save(base_path.format('test_truth'), test_truth)
        np.save(base_path.format('test_pred'), test_predictions)
        np.save(base_path.format('train_truth'), train_truth)
        np.save(base_path.format('train_pred'), train_predictions)

def main():
    # Parse arguments and set up configuration
    parser = argparse.ArgumentParser(description='Train model on LCG sequences')

    parser.add_argument('--main_seed', type=int, default=1, help='Main random seed')
    parser.add_argument('--data_seed', type=int, default=1, help='Seed for data generation')
    parser.add_argument('--base', type=int, default=256, help='Base for number representation')
    parser.add_argument('--digits', type=int, default=2, help='Number of digits per number')
    parser.add_argument('--prime', type=str, default="2,3,5,7", help='Comma-separated list of prime numbers')
    parser.add_argument('--p_min', type=int, default=256, help='Minimum modulus value')
    parser.add_argument('--p_max', type=int, default=65536, help='Maximum modulus value')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length')
    parser.add_argument('--n_a', type=int, default=8, help='Number of multiplier values')
    parser.add_argument('--n_c', type=int, default=8, help='Number of increment values')
    parser.add_argument('--n_p', type=int, default=2048, help='Number of modulus values')
    parser.add_argument('--n_example', type=int, default=1, help='Examples per (a,c) pair')
    parser.add_argument('--n_test_a', type=int, default=64, help='Number of test multiplier values')
    parser.add_argument('--n_test_c', type=int, default=64, help='Number of test increment values')
    # Model parameters
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer blocks')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=1024, help='Embedding dimension')

    # Training parameters
    parser.add_argument('--num_steps', type=int, default=1000, help='Total training steps')
    parser.add_argument('--warm_steps', type=int, default=200, help='Warmup steps')
    parser.add_argument('--lr_trgt', type=float, default=0.0005, help='Target learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')

    # Evaluation and saving parameters
    parser.add_argument('--eval_interval', type=int, default=100, help='Steps between evaluations')
    parser.add_argument('--results_dir', type=str, default='results/um')
    parser.add_argument('--save_sequences', type=str, default='False')
    parser.add_argument('--save_params', type=str, default='False')
    parser.add_argument('--save_checkpoints', type=str, default='False')
    config = parser.parse_args()
    config.context_len = config.digits * config.seq_len - 1
    config.vocab_size = config.base
    config.p_min = max(config.p_min, config.seq_len)
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(config.main_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.main_seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(config.data_seed)
    
    # Create datasets
    train_dataset, test_dataset, all_train_a, all_train_c = create_datasets(config, rng)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                            shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                           shuffle=True, num_workers=1)
    
    # Initialize model
    model = GPT_oth_abacus(GPTConfig_abacus(
        block_size=config.context_len,
        n_embd=config.n_embd,
        n_head=config.n_head,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        digits=config.digits
    )).to(device)
    model = torch.compile(model)
    
    # Training
    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=6e-4,
        beta1=config.beta1,
        beta2=config.beta2,
        device=device
    )
    
    eval_results = train_model(model, train_loader, test_loader, optimizer, 
                             config, device)
    
    # Save results
    save_results(eval_results, model, train_dataset, test_dataset, 
                all_train_a, all_train_c, config)

if __name__ == "__main__":
    main()