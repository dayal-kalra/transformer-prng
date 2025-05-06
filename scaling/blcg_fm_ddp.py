# Standard library imports

import os
import math
import time
import argparse
from socket import gethostname

# Third-party imports
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import pandas as pd

# Custom imports
from utils.prng_data import find_as, find_coprimes
from utils.datasets import BaseBLCG
from utils.gpt2 import GPTConfig_abacus, GPT_oth_abacus

def train(model, optimizer, scheduler, train_loader, test_loader, num_epoch, eval_results, device, train_sampler, config, rank):
    """
    Main training loop with distributed training and evaluation
    
    Args:
        model: DDP-wrapped neural network model
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        num_epoch: Number of epochs to train
        eval_results: List to store evaluation metrics
        device: Training device (GPU)
        train_sampler: DistributedSampler for training data
    """
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    train_last_acc = 0
    
    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        for x, y in train_loader:
            # Check steps before training
            if step >= config.num_steps:
                return
                
            # Training step
            optimizer.zero_grad(set_to_none=True)
            x, y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y) 
            
            # Compute metrics
            with torch.no_grad():
                train_loss += loss.item()
                preds = logits.argmax(dim=-1)
                train_acc += torch.sum(preds == y).float()
                train_last_acc += torch.sum(torch.all(preds[:,-config.digits:]==y[:,-config.digits:], dim=1)).float()
            
            # Optimization step
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            # Periodic evaluation
            if step % config.eval_interval == 0:
                with torch.no_grad():
                    test_loss = 0.0
                    test_acc = 0
                    test_last_acc = 0
                    total_test_samples = 0

                    for x, y in test_loader:
                        batch_size = x.size(0)
                        total_test_samples += batch_size
                        x, y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits, loss = model(x, y) 
                        test_loss += loss.item() * batch_size
                        preds = logits.argmax(dim=-1)
                        test_acc += torch.sum(preds == y).item()
                        test_last_acc += torch.sum(torch.all(preds[:,-config.digits:]==y[:,-config.digits:], dim=1)).item()

                # First normalize training metrics per process
                train_loss /= config.eval_interval
                train_acc /= config.eval_interval * config.batch_size * config.context_len
                train_last_acc /= config.eval_interval * config.batch_size

                # Average training metrics across processes
                train_metrics = torch.tensor([train_loss, train_acc, train_last_acc], device=device)
                dist.all_reduce(train_metrics, op=dist.ReduceOp.AVG)
                train_loss, train_acc, train_last_acc = train_metrics.tolist()

                # Sum test metrics and samples across processes
                test_metrics = torch.tensor([test_loss, test_acc, test_last_acc, total_test_samples], device=device)
                dist.all_reduce(test_metrics, op=dist.ReduceOp.SUM)
                test_loss, test_acc, test_last_acc, total_samples = test_metrics.tolist()
                
                # Normalize test metrics using total samples
                test_loss /= total_samples
                test_acc /= total_samples * config.context_len
                test_last_acc /= total_samples

                # Log results on rank 0
                if rank == 0:
                    print(f"rank {rank} | step {step:4d} | epoch {epoch} | "
                          f"train loss: {train_loss:.6f} | test loss: {test_loss:.6f} | "
                          f"train acc: {train_acc:.4f} | test acc: {test_acc:.4f} | "
                          f"train last acc: {train_last_acc:.4f} | test last acc: {test_last_acc:.4f}")
                    
                    eval_results.append([step, train_loss, test_loss, train_acc, test_acc, 
                                      train_last_acc, test_last_acc])
                    

                
                # Reset metrics
                train_loss = 0
                train_acc = 0
                train_last_acc = 0
                
                if step >= config.num_steps:
                    return

def setup_distributed(rank, world_size):
    """Initialize distributed training environment"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    return local_rank, device

def setup_random_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")

def create_datasets(config, rank, world_size, rng):
    """Create and return train/test datasets and loaders"""
    t0 = time.time()
    n_test_a = 512
    n_test_c = 64
    
    if rank == 0: print("calculating a and c")
    a_list = find_as(config.p, rng=rng, num=config.n_a+n_test_a)
    c_list = find_coprimes(config.p, rng=rng, num=config.n_c+n_test_c)
    
    assert len(a_list) >= config.n_a+n_test_a, "not enough a values"
    assert len(c_list) >= config.n_c+n_test_c, "not enough c values"
    train_a, val_a = a_list[:config.n_a], a_list[config.n_a:]
    train_c, val_c = c_list[:config.n_c], c_list[config.n_c:]

    train_dataset = BaseBLCG(p=config.p,base=config.base,digits=config.digits,length=config.seq_len,a_list=train_a,c_list=train_c,rng=rng,num_examples=config.n_example)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=config.world_size,
                                                                    rank=rank,
                                                                    shuffle=True,
                                                                    drop_last=True)

    train_loader = DataLoader(train_dataset,
                                batch_size=config.batch_size,
                                sampler=train_sampler,
                                num_workers=8,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=2)

    # Create test dataset
    test_dataset = BaseBLCG(p=config.p, base=config.base, digits=config.digits, length=config.seq_len,
                           a_list=val_a, c_list=val_c, rng=rng, num_examples=config.n_example)
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=config.world_size,
        rank=rank,
        shuffle=False,
        drop_last=True
    )
    
    test_loader = DataLoader(test_dataset,
                            batch_size=config.batch_size,
                            sampler=test_sampler,
                            num_workers=8,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2
                           )

    if rank == 0:
        print(f"train data size: {len(train_dataset)}, test data size: {len(test_dataset)}")
        print(f"data generation took: {time.time()-t0:.2f} seconds")
    
    return train_sampler, train_loader, test_loader, train_dataset, test_dataset, train_a, train_c

def save_results(config, model, eval_results, train_dataset, test_dataset, train_a, train_c):
    """Save training results, model parameters and sequences"""
    df_eval = pd.DataFrame(eval_results, columns=['step', 'train_loss', 'test_loss', 
                                                 'train_acc', 'test_acc','train_last_acc', 'test_last_acc'])
    base_path = f'{config.results_dir}/p{config.p}_b{config.base}_d{config.digits}_sl{config.seq_len}_na{config.n_a}_nc{config.n_c}_ne{config.n_example}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_ds{config.data_seed}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Twarm{config.warm_steps}_T{config.num_steps}_B{config.batch_size * config.world_size}_wd{config.weight_decay:.2f}'
    
    # Save evaluation results
    df_eval.to_csv(f'{base_path}.tab', sep='\t')
    
    # Save model parameters if enabled
    if config.save_params == "True":
        torch.save(model.state_dict(), f'{base_path}.pth')
        np.savez(f'{base_path}_trainac', train_a=train_a, train_c=train_c)
    

def main():
    """Main training pipeline"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train LCG prediction model with distributed training')
    parser.add_argument('--main_seed', type = int, default = 1)
    parser.add_argument('--data_seed', type = int, default = 1)
    ### Dataset hyperparams
    parser.add_argument('--base', type = int, default = 512) 
    parser.add_argument('--digits', type = int, default = 2) 
    parser.add_argument('--p', type = int, default = None) 
    parser.add_argument('--seq_len', type = int, default = 129)
    parser.add_argument('--n_a', type = int, default = 128)
    parser.add_argument('--n_c', type = int, default = 128)
    parser.add_argument('--n_example', type = int, default = 8)

    ### Model hyperparams
    parser.add_argument('--n_layer', type = int, default = 2) # 12 by default
    parser.add_argument('--n_head', type = int, default = 6) # 8 by default
    parser.add_argument('--n_embd', type = int, default = 768) # 768 by default source 

    ### Optimization hyperparams
    parser.add_argument('--num_steps', type = int, default = 5000) 
    parser.add_argument('--warm_steps', type = int, default = 500)
    parser.add_argument('--lr_trgt', type = float, default = 0.0005)
    parser.add_argument('--lr_min', type = float, default = 1e-7)
    parser.add_argument('--batch_size', type = int, default = 64)
    # adamw hyperparams
    parser.add_argument('--weight_decay', type = float, default = 1.0)
    parser.add_argument('--beta1', type = float, default = 0.9)
    parser.add_argument('--beta2', type = float, default = 0.98)
    ### Evaluation hyperparams
    parser.add_argument('--eval_interval', type = int, default = 1000)
    parser.add_argument('--results_dir', type = str, default = 'results/1p')
    parser.add_argument('--save_params', type = str, default = 'False')

    config = parser.parse_args()
    config.context_len = config.digits * config.seq_len - 1
    if config.p is None:
        config.p = config.base ** config.digits
    assert config.p <= config.base ** config.digits, "p is too large for the base and digits"
    config.vocab_size = config.base
    config.world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    
    # Setup distributed training
    local_rank, device = setup_distributed(rank, config.world_size)
    setup_random_seeds(config.main_seed)
    
    # Create model and optimizer
    model = GPT_oth_abacus(GPTConfig_abacus(
        block_size=config.context_len,
        n_embd=config.n_embd,
        n_head=config.n_head,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        digits=config.digits
    )).to(device)
    
    ddp_model = DDP(model, device_ids=[local_rank])
    raw_model = ddp_model.module
    optimizer = raw_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.lr_trgt,
        beta1=config.beta1,
        beta2=config.beta2,
        device=device
    )
    
    # Create custom warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=config.warm_steps
    )
    
    # Create cosine annealing scheduler
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_steps - config.warm_steps,
        eta_min=config.lr_min
    )
    
    # Combine schedulers
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[config.warm_steps]
    )
    
    # Create datasets
    rng = np.random.default_rng(config.data_seed)
    train_sampler, train_loader, test_loader, train_dataset, test_dataset, train_a, train_c = create_datasets(
        config, rank, config.world_size, rng
    )
    
    # Train model
    if rank == 0:
        print("Starting training...")
        t0 = time.time()
    
    eval_results = []
    num_epoch = np.ceil(config.num_steps / len(train_loader)).astype(int)
    train(ddp_model, optimizer, scheduler, train_loader, test_loader, num_epoch, eval_results, device, train_sampler, config, rank)
    
    # Save results on rank 0
    if rank == 0:
        print(f"Training completed in {time.time() - t0:.2f} seconds")
        save_results(config, raw_model, eval_results, train_dataset, test_dataset, train_a, train_c)
    
    dist.destroy_process_group()



if __name__ == "__main__":
    main()
