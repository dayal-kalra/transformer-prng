import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.prng_data as prngs_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
torch.set_printoptions(threshold=float('inf'))

from utils.gpt2 import GPT, GPTConfig
from utils.prng_data import lcg_vectorized, find_as, find_coprimes

# usual imports
import pickle as pl
import pandas as pd
import argparse
import os
import copy
import random

from tqdm.auto import tqdm

plt.rcParams.update({"font.size": 20})
sns.set_theme(style="whitegrid")
dpi = 300
cmap = 'coolwarm'

# set the internal precision of float32 matrix multiplications: 
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
# “highest”, float32 matrix multiplications use the float32 datatype (24 mantissa bits with 23 bits explicitly stored) for internal computations.
# “high”, float32 matrix multiplications either use the TensorFloat32 datatype (10 mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers 
torch.set_float32_matmul_precision('high')
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32  # fp16 needs a further change of the code.

## Toggle to true if you want to use GPU
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = 'cuda'
elif USE_GPU and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'



def main(config):
    config.vocab_size = config.p_eval

    ## Set random seeds
    random.seed(config.main_seed)
    np.random.seed(config.main_seed)
    torch.manual_seed(config.main_seed)
    torch.cuda.manual_seed(config.main_seed)


    ## Create model and load checkopints
    model = create_model(config)

    ## Replace the directory and filename below with your own
    ckpt_path = f'{config.results_dir}/chkpt_p{config.p_eval}_Tn{config.context_len}_N{config.total_examples}_ne{config.num_examples_per_prng}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_wd{config.weight_decay}.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    ## Find test (a,c) values using Hull-Dobell theorem
    ## The same values were used in the test set during training
    a_list, c_list = find_prng_parameters_test(config)
    
    ## Loop over all (a,c) values in the test set.
    ## For a quicker run, this can be removed/reduced
    token_ids = []
    accs = []
    for a in a_list:
        for c in c_list:
            accs.append(calculate_per_digit_accuracy(model, a, c, token_ids))
    accs = np.stack(accs, axis=0)
    accs = accs.mean(0)
    
    ## Save the data
    ## Replace the directory and filename with your own
    np.save(f"{config.plots_dir}/digits_p{config.p_eval}_d{config.n_layer}_h{config.n_head}_n{config.n_embd}.npy", accs)
    
    return


def parse_args_fn():
    parser = argparse.ArgumentParser(description = 'Hyperparameters')
    parser.add_argument('--main_seed', type = int, default = 1) # main seed for the experiments
    ### Dataset hyperparams
    parser.add_argument('--p_eval', type = int, default = 2048) # p for mod p
    parser.add_argument('--num_as', type = int, default = 16) # number of as
    parser.add_argument('--num_cs', type = int, default = 16) # number of cs
    parser.add_argument('--num_examples_per_prng', type = int, default = 1) # number of examples
    parser.add_argument('--total_examples', type = int, default = 1000_000) # number of examples
    parser.add_argument('--context_len', type = int, default = 256) # number of examples
    parser.add_argument('--chunk_size', type = int, default = 32) # number of examples
    parser.add_argument('--period_min', type = int, default = 0) # min period of training
    parser.add_argument('--period_max', type = int, default = 512) # max period of training
    ### Model hyperparams
    parser.add_argument('--n_layer', type = int, default = 1) # number of layers
    parser.add_argument('--n_head', type = int, default = 1) # number of heads
    parser.add_argument('--n_embd', type = int, default = 768)  # embedding dimension
    parser.add_argument('--head_dim', type = int, default = 768) # number of heads
    parser.add_argument('--act_name', type = str, default = 'relu') # activation
    ### Optimization hyperparams
    # parser.add_argument('--step', type = int, default = 2000) # number of training steps
    parser.add_argument('--num_steps', type = int, default = 100_000) # number of training steps
    parser.add_argument('--warmup_steps', type = int, default = 2048) # number of warmup steps
    parser.add_argument('--lr_trgt', type = float, default = 0.3e-4) # the target learning rate
    parser.add_argument('--lr_init', type = float, default = 1e-6) # initial learning rate
    parser.add_argument('--lr_min', type = float, default = 1e-6) # final learning rate
    parser.add_argument('--batch_size', type = int, default = 256) # batch size
    # adamw hyperparams
    parser.add_argument('--weight_decay', type = float, default = 1.0) # weight decay
    parser.add_argument('--beta1', type = float, default = 0.9) # beta1 
    parser.add_argument('--beta2', type = float, default = 0.99) # beta2
    ### Evaluation hyperparams
    parser.add_argument('--results_dir', type = str, default = 'results')
    parser.add_argument('--plots_dir', type = str, default = 'plots')
    # Other
    parser.add_argument('--shifts', type = int, default = 0) # position of 1 to p_eval numbers in the sequence
    
    return parser.parse_args()


def find_prng_parameters_test(config):
    """ For a given p, find the possible values of a's and c's according to the Hull–Dobell Theorem: https://en.wikipedia.org/wiki/Linear_congruential_generator """

    a_list = prngs_data.find_as(config.p_eval)
    c_list = prngs_data.find_coprimes(config.p_eval)

    val_as = np.random.choice(a_list, min(64, len(a_list)))
    val_cs = np.random.choice(c_list, min(64, len(c_list)))

    ## use arbitary (a, c) as long as its not in val_a and val_c
    # train_as = [i for i in range(1, config.p_eval) if i not in val_as]
    # train_cs = [i for i in range(1, config.p_eval) if i not in val_cs]

    return val_as, val_cs


def create_model(config):
    gptconfig = GPTConfig(block_size=config.context_len, n_embd=config.n_embd, n_head=config.n_head, vocab_size=config.vocab_size, n_layer=config.n_layer, act_name=config.act_name)
    model = GPT(gptconfig)
    model.to(device)
    return model


@torch.inference_mode()
def calculate_per_digit_accuracy(model, a, c, token_ids, base_save_path: str | None = None) -> None:
    
    ## Average over all seeds
    seq_collections = lcg_vectorized_with_all_seeds(p = config.p_eval, length = config.context_len + 1, a_list = [a], c_list = [c]) # (all_ac_pairs, p, seq_len)
    test_dataset = seq_collections[0]
    
    ## Forward Pass
    x = test_dataset[:, :-1].to(device)
    y = test_dataset[:, 1:].to(device)  
      
    # with torch.autocast(device_type = device, dtype = dtype):
    #     logits, loss = model(x, y)
    if device != 'mps':
        with torch.inference_mode():
            logits, loss = model(x, y)
    else:
        logits, loss = model(x, y)

    preds = logits.argmax(-1).detach().cpu()
    y = y.detach().cpu()
    
    wte = model.transformer.wte.weight.data.cpu()  # (vocab, n_embd)
    wpe = model.transformer.wpe.weight.data.cpu()  # (vocab, n_embd)
    
    ## Get the 
    # n_bits = int(np.log2(config.p_eval))
    pred_digits, places = digits235(preds, config.p_eval)
    y_digits, places = digits235(y, config.p_eval)
    
    results = (pred_digits == y_digits).numpy().mean(0)
    
    # ylim = 80
    # plt.figure(figsize=(5,5))
    # # plt.grid(False)
    # plt.imshow(results[:ylim, :].T, aspect=6)
    # plt.yticks(range(results.shape[-1]), labels=places.numpy())
    # plt.colorbar(fraction=0.05, aspect=15)
    # # plt.tight_layout()
    # plt.show()
    
    return results


"""Data"""
@torch.inference_mode()
def lcg_vectorized_with_all_seeds(p: int = 512, length: int = 8, a_list: list = [45], c_list: list = [123]) -> torch.Tensor:
    """
    Vectorized version of lcg function with sequential seeds starting from 0 to p-1.
    It supports multiple 'a' and 'c' values; but we're feeding them one-at-a-time to avoid memory blow-ups.
    """
    ## Create mesh grid and flatten
    a_mesh, c_mesh = np.meshgrid(a_list, c_list)
    a_flat = torch.tensor(a_mesh.flatten(), dtype=torch.int64)
    c_flat = torch.tensor(c_mesh.flatten(), dtype=torch.int64)

    ## Generate initial seeds from 0 to p-1. To save memory and/or time, this can be changed to average over fewer seeds.
    initial_seeds = torch.arange(p, dtype=torch.int64)[:]

    def single_lcg(a, c, seed):
        @torch.compile
        def next_value(prev):
            return (a*prev + c) % p
        
        sequence = [seed]
        for _ in range(length - 1):
            sequence.append(next_value(sequence[-1]))
        
        return torch.stack(sequence)

    ## Vectorize over a, c, and initial seeds
    results = torch.vmap(torch.vmap(single_lcg, in_dims=(None, None, 0)), in_dims=(0, 0, None), chunk_size=16)(a_flat, c_flat, initial_seeds)
    
    ## Reshape to combine all sequences
    return results.reshape(a_flat.size(0), -1, length)
       

def digits235(tensor:torch.tensor, p):
    
    factors = factorize_235(p)
    digits = {}
    places = {}
    for key in factors.keys():
        digits[key] = []
        places[key] = []
        tensor_copy = copy.deepcopy(tensor).to(torch.float)
        for i in range(1, factors[key]+1):
            digits[key].append(tensor_copy % key)
            places[key].append(key**i)
            tensor_copy = tensor_copy // key
    
    digits_list = []
    places_list = []
    for key in factors.keys():
        digits_list = digits_list + digits[key]
        places_list = places_list + places[key]
        
    places = torch.tensor(places_list)
    digits = torch.stack(digits_list, dim=-1)
    
    sorted_indices = torch.argsort(places)
    places = places[sorted_indices]
    digits = digits[..., sorted_indices]
        
    return digits, places
 

def factorize_235(n: int):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input must be a positive integer")
    
    powers = {2: 0, 3: 0, 5: 0}
    remainder = n
    
    # Factor out 2s, 3s, and 5s
    for prime in (2, 3, 5):
        while remainder % prime == 0:
            powers[prime] += 1
            remainder //= prime
    
    return powers
    

if __name__ == "__main__":
    config = parse_args_fn()
    main(config)