import argparse
import math
import random
import numpy as np
import torch
import torch.nn as nn

import utils.prng_data as prngs_data
import utils.prng_tests as prng_tests
from utils.datasets import PRNGsDataset

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors

from utils.prng_data import lcg_vectorized, find_as, find_coprimes
from utils.gpt2 import GPT, GPTConfig

plt.rcParams.update({"font.size": 20})
sns.set_theme(style="whitegrid")
dpi = 300


def invert_color(hex_color):
    """Inverts a hex color."""
    # Remove the '#' and convert to integer
    rgb = int(hex_color[1:], 16)
    # Invert each color component by subtracting from 255
    inverted_rgb = 0xFFFFFF - rgb
    # Convert back to hex and return
    return f'#{inverted_rgb:06x}'

device = 'cuda'

def estimate_epochs(num_examples, batch_size, num_steps):
    steps_per_epoch = math.ceil(num_examples / batch_size)
    estimated_epochs = math.floor(num_steps / steps_per_epoch)
    return estimated_epochs, steps_per_epoch

def find_prng_parameters(config):
    """ For a given p, find the possible values of a's and c's according to the Hull–Dobell Theorem: https://en.wikipedia.org/wiki/Linear_congruential_generator """

    a_list = prngs_data.find_as(config.p_eval)
    c_list = prngs_data.find_coprimes(config.p_eval)

    val_as = np.random.choice(a_list, min(64, len(a_list)))
    val_cs = np.random.choice(c_list, min(64, len(c_list)))

    ## use arbitary (a, c) as long as its not in val_a and val_c
    train_as = [i for i in range(1, config.p_max) if i not in val_as]
    train_cs = [i for i in range(1, config.p_max) if i not in val_cs]

    return train_as, train_cs, val_as, val_cs


def create_model(config):
    gptconfig = GPTConfig(block_size=config.context_len, n_embd=config.n_embd, n_head=config.n_head, vocab_size=config.vocab_size, n_layer=config.n_layer, act_fn=config.act_fn)
    model = GPT(gptconfig)
    model.to(device)
    return model

    
@torch.inference_mode()
def evaluate_model(config, model, test_loader):
    
    test_loss = 0.0
    test_acc = 0
    test_last_acc = 0
    model.eval()

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type = device, dtype = dtype):
            logits, loss = model(x, y) 
        # estimate loss
        test_loss += loss.item()  # Sum up batch loss
        # estimate accuracy
        preds = logits.argmax(dim=-1)
        test_acc += torch.sum(preds == y).item()
        test_last_acc += torch.sum(preds[:, -1] == y[:, -1]).item()
    
    model.train()
    # normalize the metrics
    test_loss = test_loss / len(test_loader) 
    test_acc = test_acc / len(test_loader) / config.batch_size / config.context_len
    test_last_acc = test_last_acc / len(test_loader) / config.batch_size
    return test_loss, test_acc, test_last_acc

"""Interp Utils"""
class Hook_Mean_Pre:
    def __init__(self, module: nn.Module, head_dim: int, head_idx=None, cache_steps=256):
        self.hook = module.register_forward_pre_hook(self.hook_fn)
        self.head_dim = head_dim
        self.head_idx = head_idx
        self.cache_steps = cache_steps
        self.cache = None
        
    def hook_fn(self, module: nn.Module, input: torch.Tensor):
        if module.training is True:
            input = list(input)
            input[0] = input[0].clone()
            if self.cache is None:
                self.cache = input[0][:, :, self.head_dim*self.head_idx : self.head_dim*(self.head_idx+1)] / self.cache_steps
            else:
                self.cache += input[0][:, :, self.head_dim*self.head_idx : self.head_dim*(self.head_idx+1)] / self.cache_steps
            # print(self.cache)
            
        else:
            if (self.head_idx is not None):
                input = list(input)
                input[0] = input[0].clone()
                input[0][:, :, self.head_dim*self.head_idx : self.head_dim*(self.head_idx+1)] = self.cache.mean((0, 1, 2), keepdim=True).expand(input[0][:, :, self.head_dim*self.head_idx : self.head_dim*(self.head_idx+1)].size())
            return tuple(input)
    
    def close(self):
        self.hook.remove()
        

def hook_model(model, config, layer_idx=None, head_idx=None, cache_steps=256):
    head_dim = config.n_embd // config.n_head
    if layer_idx is not None:
        return Hook_Mean_Pre(model.transformer.h[layer_idx].attn.c_proj, head_dim, head_idx=head_idx, cache_steps=cache_steps)
    else:
        return None

parser = argparse.ArgumentParser(description = 'Hyperparameters')
parser.add_argument('--main_seed', type = int, default = 1) # main seed for the experiments
### Dataset hyperparams
parser.add_argument('--p_eval', type = int, default = 512) # p for mod p
parser.add_argument('--p_min', type = int, default = 256)  # smallest p for mod p: p_min = p_eval - num_ps
parser.add_argument('--p_max', type = int, default = 640)  # largest p for mod p: p_max = p_eval + num_ps / 2
parser.add_argument('--num_ps', type = int, default = 256) # number of ps for training
parser.add_argument('--num_as', type = int, default = 32) # number of as
parser.add_argument('--num_cs', type = int, default = 32) # number of cs
parser.add_argument('--num_examples', type = int, default = 1) # number of examples
parser.add_argument('--context_len', type = int, default = 256) # number of examples
parser.add_argument('--chunk_size', type = int, default = 64) # number of examples
parser.add_argument('--period_min', type = int, default = 0) # min period of training
parser.add_argument('--period_max', type = int, default = 512) # max period of training
### Model hyperparams
parser.add_argument('--model_name', type = str, default = 'gpt2') # main seed for the experiments
parser.add_argument('--n_layer', type = int, default = 2) # number of layers
parser.add_argument('--n_head', type = int, default = 6) # number of heads
parser.add_argument('--n_embd', type = int, default = 768)  # embedding dimension
parser.add_argument('--act_name', type = str, default = 'relu') # number of layers
parser.add_argument('--theta', type = int, default = 10000) # p for mod p
### Optimization hyperparams
parser.add_argument('--step', type = int, default = 2000) # number of training steps
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
parser.add_argument('--eval_interval', type = int, default = 100) # evaluation interval
parser.add_argument('--results_dir', type = str, default = './chkpts')
parser.add_argument('--interp_results_dir', type = str, default = 'results/lcg/multiple_ps')
parser.add_argument('--plots_dir', type = str, default = 'plots/lcg/multiple_ps/per_digit_acc_prune')
parser.add_argument('--save_sequences', type = str, default = 'False')
parser.add_argument('--save_checkpoint', type = str, default = 'True')
# Other
parser.add_argument('--show_tqdm_bar', type = str, default = 'True')
parser.add_argument('--shifts', type = int, default = 0)
parser.add_argument('--cache_steps', type = int, default = 512) # number of examples

config = parser.parse_args()
config.p_max = int(1.2 * config.p_eval) # p_max = p_val + p_offset / 2.0
config.p_min = config.context_len+1
config.block_size = 4 * config.context_len  # Max possible sequence length

activations = {'relu': torch.nn.ReLU(), 'gelu': torch.nn.GELU()}
config.act_fn = activations[config.act_name]
print(config.act_fn)

config.vocab_size = config.p_max # vocab size is equal to the maximum p
if config.p_min < config.context_len+1:
    config.p_min = config.context_len+1
    print(f'p_min set to context length')

# if I am not wrong, this seed only takes care of torch and not numpy
np.random.seed(config.main_seed)
torch.manual_seed(config.main_seed)
torch.cuda.manual_seed(config.main_seed)
random.seed(config.main_seed)

# Train data

# find parameters for prngs: for a given p_eval, find (a, c) pairs
train_as, train_cs, val_as, val_cs = find_prng_parameters(config)
config.num_as = min(len(train_as), config.num_as)
config.num_cs = min(len(train_cs), config.num_cs)

print(f'Possible a values: {len(train_as)}, possible c values: {len(train_cs)}')

chosen_ps = [2178,  # 33
             2352,  # 28
             1521,  # 39
             2312,  # 34 
             1936,  # 44
             1800,  # 30
             ]

ps = list(range(config.p_min, config.p_max + 1)) # list of p values
ps.remove(config.p_eval)
for chosen_p in chosen_ps:
    if chosen_p in ps:
        ps.remove(chosen_p)
ps = np.array(ps)

config.num_ps = min(config.num_ps, len(ps))
ps = np.random.choice(ps, config.num_ps, replace = False)
print(np.sort(ps))

subset_as = np.random.choice(train_as, config.num_as, replace = False)
subset_cs = np.random.choice(train_cs, config.num_cs, replace = False)

training_data = {}
periods_data = []
print(f'Generating train data with randomly chosen {config.num_as} a values and {config.num_cs} c values for each p')

for p in ps: # for each p, randomly select num_as, num_cs to create sub dataset
    
    data = lcg_vectorized(p = p, length = p, a_list = subset_as, c_list = subset_cs, num_examples = 1, chunk_size = config.chunk_size)
    # NOTE: only works for num_examples = 1, otherwise the sequence would not be correct
    a_mesh, c_mesh = np.meshgrid(subset_as, subset_cs)
    a_flat, c_flat = a_mesh.flatten(), c_mesh.flatten()

    # estimate the periods upto length p
    periods = np.asarray([prng_tests.first_repeat_index(data[i].numpy()) for i, (a, c) in enumerate(zip(a_flat, c_flat))])
    # print(f'p: {p}, min period: {min(periods)}, max period: {max(periods)}')

    # Update training_data dictionary
    for i, (a, c) in enumerate(zip(a_flat, c_flat)):
        if config.period_min <= periods[i] <= config.period_max:
            training_data[(a, c, p)] = data[i:i+1, :config.context_len+1]
    
    combined_arrays = np.column_stack((a_flat, c_flat, np.full_like(a_flat, p), periods))
    periods_data.append(combined_arrays)

periods_data = np.vstack(periods_data)
# create a dictionary of different periods
periods_dict = {tuple(row[:3]): row[3] for row in periods_data}
# save periods data
# df_periods = pd.DataFrame(periods_data, columns = ['a', 'c', 'p', 'period'], dtype = int)
# periods_path = f'prng-tests/periods_lcg_p{config.p_eval}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_Tn{config.context_len}_N{config.num_examples}.tab'
# df_periods.to_csv(periods_path, sep = '\t')

total_hard_examples = (periods_data[:, -1] > config.context_len).astype(int).sum() # periods is the last column
total_examples = len(periods_data)
print(f'Total Hard = {total_hard_examples}, Total examples = {total_examples}, Percentage of hard examples: {100*total_hard_examples / total_examples:0.2f}%')

# combine the training datasets for training
combined_dataset = torch.utils.data.ConcatDataset([PRNGsDataset(value) for key, value in training_data.items()])

# create train loader
train_loader = torch.utils.data.DataLoader(combined_dataset, batch_size = config.batch_size, shuffle = True, num_workers = 0) 
config.num_train = len(train_loader.dataset)

# Color
N = (config.p_max // 6) + 1  # number of colors to extract from each of the base_cmaps below
base_cmaps = ['Greys', 'Purples', 'Reds', 'Oranges', 'Blues', 'Greens']

n_base = len(base_cmaps)
# we go from 0.2 to 0.8 below to avoid having several whites and blacks in the resulting cmaps
colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.2, 0.8, N)) for name in base_cmaps])
custom_cmap = mcolors.ListedColormap(colors)

torch.set_float32_matmul_precision('high')
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32


# create model
model = create_model(config)

# Load Checkpoints
if config.act_name.lower() == 'gelu':
    ckpt_path = f'{config.results_dir}/chkpt_cps_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}.pth'
else:
    ckpt_path = f'{config.results_dir}/chkpt_cps_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_{config.act_name}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}.pth'
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()


@torch.inference_mode()
def lcg_vectorized_with_fixed_seed(p: int = 512, length: int = 8, a_list: list = [45], c_list: list = [123]) -> torch.Tensor:
    """
    Vectorized version of lcg function with sequential seeds starting from 0 to p-1,
    supporting multiple 'a' and 'c' values.
    """
    # Create mesh grid and flatten
    a_mesh, c_mesh = np.meshgrid(a_list, c_list)
    a_flat = torch.tensor(a_mesh.flatten(), dtype=torch.int64)
    c_flat = torch.tensor(c_mesh.flatten(), dtype=torch.int64)

    # Generate initial seeds from 0 to p-1
    num_seeds = min(128, p)
    initial_seeds = torch.randperm(p)[:num_seeds].to(torch.int64)

    def single_lcg(a, c, seed):
        @torch.compile
        def next_value(prev):
            return (a*prev + c) % p
        
        sequence = [seed]
        for _ in range(length - 1):
            sequence.append(next_value(sequence[-1]))
        
        return torch.stack(sequence)

    # Vectorize over a, c, and initial seeds
    results = torch.vmap(torch.vmap(single_lcg, in_dims=(None, None, 0)), in_dims=(0, 0, None), chunk_size=32)(a_flat, c_flat, initial_seeds)
    
    # Reshape to combine all sequences
    return results.reshape(len(c_list), len(a_list), num_seeds, length)


def prime_factorization(n):
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d * d > n:
            if n > 1:
                factors.append(n)
            break
    return factors

@torch.inference_mode()
def get_power(a, p):
    for i in range(1, p):
        if pow(a, i, p) == 1:
            return i
    return 0

def factorize(n: int):
    """Proper prime factorization of n into a dictionary of {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = 1
    return factors

def digits235(tensor: torch.Tensor, p):
    """Decompose numbers into digits for each prime power in p's factorization, returning digits, places, and labels."""
    factors = factorize(p)
    digits = {}
    places = {}
    labels = []  # Stores labels for each digit position
    
    for prime in sorted(factors.keys()):  # Process primes in sorted order
        exp = factors[prime]
        digits[prime] = []
        places[prime] = []
        tensor_copy = tensor.clone().to(torch.float)
        for i in range(1, exp + 1):
            current_place = prime ** i
            digits[prime].append(tensor_copy % prime)
            places[prime].append(current_place)
            labels.append(f"{prime}$^{{{i}}}$")  # LaTeX-style exponent
            tensor_copy = tensor_copy // prime
    
    # Flatten the digits and places while maintaining prime order
    digits_list = []
    places_list = []
    for prime in sorted(factors.keys()):
        digits_list.extend(digits[prime])
        places_list.extend(places[prime])
    
    # Convert to tensors
    places_tensor = torch.tensor(places_list)
    digits_tensor = torch.stack(digits_list, dim=-1)
    
    return digits_tensor, places_tensor, labels


@torch.inference_mode()
def plot_per_digit_accuracy(model, chosen_p, config, head_idx, layer_idx, num_c_samples=16):
    """
    Create separate plots for different power groups, averaging over multiple c values.
    """
    _, _, labels = digits235(torch.zeros(1), chosen_p)
    # Get all valid multipliers and constants
    a_list = prngs_data.find_as(chosen_p)
    c_list = prngs_data.find_coprimes(chosen_p)
    
    # Sample c values if there are too many
    if len(c_list) > num_c_samples:
        sampled_c_list = np.random.choice(c_list, size=num_c_samples, replace=False)
    else:
        sampled_c_list = c_list
    
    # Generate all sequences at once
    seq_collections = lcg_vectorized_with_fixed_seed(
        p=chosen_p, 
        length=config.context_len + 1,
        a_list=a_list,
        c_list=sampled_c_list
    )  # shape: (len(c_list), len(a_list), num_seeds, seq_len)
    
    all_results = []
    
    # Process each batch of sequences
    for c_idx in range(len(sampled_c_list)):
        for a_idx in range(len(a_list)):
            sequences = seq_collections[c_idx, a_idx]  # shape: (num_seeds, seq_len)
            
            # Prepare input and target
            x = sequences[:, :-1].to(device)
            y = sequences[:, 1:].to(device)
            
            # Forward pass
            with torch.autocast(device_type=device, dtype=dtype):
                logits, loss = model(x, y)
            
            # Get predictions
            preds = logits.argmax(-1).detach().cpu()
            y = y.detach().cpu()
            
            # Calculate digit-wise accuracy
            pred_digits, places, _ = digits235(preds, chosen_p)
            y_digits, places, _ = digits235(y, chosen_p)
            
            results = (pred_digits == y_digits).numpy().mean(0)
            all_results.append(results)
    
    # Average results across all sequences
    avg_results = np.mean(all_results, axis=0)
    
    # Create plot for this power
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    sns.heatmap(
        avg_results[:, :].T,
        cmap='YlGnBu',
        ax=ax,
        cbar=True,
        xticklabels=8,
        yticklabels=labels,
        vmin=0.0,
        vmax=1.0,
    )
    
    # ax.set_title(f'Power = {power} (n={len(selected_a_indices)} a\'s, {len(sampled_c_list)} c\'s)')
    ax.set_xticks(np.arange(0, avg_results.shape[0], avg_results.shape[0] // 8) + 0.5)
    ax.set_xticklabels(np.arange(0, avg_results.shape[0], avg_results.shape[0] // 8) + 1, rotation=0, fontsize=16)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, rotation=45, fontsize=14)
    ax.set_xlabel('token position')
    ax.set_ylabel(f'prime factor digit')
    
    ax.tick_params(
        axis='both',
        which='both',
        length=2,
        width=0.5,
        color='black',
        bottom=True,
        left=True
    )

    # Optional: Ensure axis spines are visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    # Save the figure
    fig_path = f'{config.plots_dir}/acc_digit_new_l{layer_idx}h{head_idx}_p{chosen_p}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_B{config.batch_size}_wd{config.weight_decay}.pdf'
    fig.savefig(fig_path, format='pdf', dpi=dpi)
    plt.close()

    
"""Run"""
with torch.inference_mode():
    chosen_ps = [2178,  # 33
                 2352,  # 28
                 1521,  # 39
                 2312,  # 34 
                 1936,  # 44
                 1800,  # 30
                 ]
    palettes = sns.color_palette('tab10', n_colors=config.n_layer + 1)
    markers = [ 'v', 's', 'p', '*', 'X', 'd', 'o']
    Ts = [100000]
    for step in Ts:
        # for chosen_p in (chosen_ps):
        for chosen_p in ([config.p_eval,] + chosen_ps):
            config.step = step
            for layer_idx in range(config.n_layer):
                for head_idx in range(config.n_head):
                    model.train()
                    
                    # use layer_idx=None and head_idx=None for normal run
                    hook = hook_model(model, config, layer_idx=layer_idx, head_idx=head_idx, cache_steps=config.cache_steps)
                    for t, (x, _) in enumerate(train_loader):
                        if t < config.cache_steps:
                            x = x.to(device)
                            _, _ = model(x)
                    
                    model.eval()
                    
                    plot_per_digit_accuracy(model, chosen_p, config, layer_idx=layer_idx, head_idx=head_idx, num_c_samples=16)
                    
                    hook.close()