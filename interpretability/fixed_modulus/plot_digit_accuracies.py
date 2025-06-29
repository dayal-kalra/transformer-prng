import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
torch.set_printoptions(threshold=float('inf'))

import argparse

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
USE_GPU = False
if USE_GPU and torch.cuda.is_available():
    device = 'cuda'
elif USE_GPU and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


def main(config):
    config.vocab_size = config.p_eval

    # if I am not wrong, this seed only takes care of torch and not numpy
    np.random.seed(config.main_seed)
    torch.manual_seed(config.main_seed)
    torch.cuda.manual_seed(config.main_seed)

    ## Color
    N = (config.p_eval // 6) + 1  # number of colors to extract from each of the base_cmaps below
    base_cmaps = ['Greys', 'Purples', 'Reds', 'Oranges', 'Blues', 'Greens']

    n_base = len(base_cmaps)
    # we go from 0.2 to 0.8 below to avoid having several whites and blacks in the resulting cmaps
    colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.2, 0.8, N)) for name in base_cmaps])
    custom_cmap = mcolors.ListedColormap(colors)

    ## Load numpy array containing per-digit accuracies
    accs = np.load(f"{config.plots_dir}/digits_p{config.p_eval}_d{config.n_layer}_h{config.n_head}_n{config.n_embd}.npy")
    
    places = places235(config.p_eval)
    
    ylim = 80
    plt.figure(figsize=(5,5))
    # plt.grid(False)
    plt.imshow(accs[:ylim, :].T, aspect=6)
    plt.yticks(range(accs.shape[-1]), labels=places)
    plt.colorbar(fraction=0.05, aspect=15)
    # plt.tight_layout()
    
    ## Replace the filename and directory with your own
    plt.savefig(f"{config.plots_dir}/digits_p{config.p_eval}_d{config.n_layer}_h{config.n_head}_n{config.n_embd}.pdf", dpi=400, format='pdf')
    plt.show()


def parse_args_fn():
    parser = argparse.ArgumentParser(description = 'Hyperparameters')
    parser.add_argument('--main_seed', type = int, default = 1) # main seed for the experiments
    ### Dataset hyperparams
    parser.add_argument('--p_eval', type = int, default = 2048) # p for mod p
    parser.add_argument('--num_as', type = int, default = 32) # number of as
    parser.add_argument('--num_cs', type = int, default = 32) # number of cs
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
    parser.add_argument('--num_steps', type = int, default = 10_000) # number of training steps
    parser.add_argument('--warmup_steps', type = int, default = 2048) # number of warmup steps
    parser.add_argument('--lr_trgt', type = float, default = 0.003) # the target learning rate
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


def places235(n: int):
    factors = factorize_235(n)
    places = []
    for key in factors.keys():
        for i in range(1, factors[key]+1):
            places.append(key**i)
    
    return np.array(places)

        
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