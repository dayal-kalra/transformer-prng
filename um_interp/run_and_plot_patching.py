import math
import random
import nnsight.models
import torch
import torch.nn as nn
import utils.prng_data as prngs_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors

from utils.gpt2 import GPT, GPTConfig, MLP, CausalSelfAttention
# from utils.schedules import linear_warmup_cosine
from utils.prng_data import lcg_vectorized, find_as, find_coprimes
# from utils.datasets import PRNGsDataset

# usual imports
import argparse

# Interpretability imports
import nnsight

plt.rcParams.update({"font.size": 20})
sns.set_theme(style="whitegrid")
dpi = 300
cmap = 'coolwarm'

def invert_color(hex_color):
    """Inverts a hex color."""
    # Remove the '#' and convert to integer
    rgb = int(hex_color[1:], 16)
    # Invert each color component by subtracting from 255
    inverted_rgb = 0xFFFFFF - rgb
    # Convert back to hex and return
    return f'#{inverted_rgb:06x}'

device = 'mps'

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
class Hook:
    def __init__(self, module: nn.Module, backward: bool = False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_full_backward_hook(self.hook_fn)

    def hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        with torch.no_grad():
            self.input = input[0]
            self.output = output

    def close(self):
        self.hook.remove()

    
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
parser.add_argument('--chunk_size', type = int, default = 32) # number of examples
parser.add_argument('--period_min', type = int, default = 0) # min period of training
parser.add_argument('--period_max', type = int, default = 512) # max period of training
### Model hyperparams
parser.add_argument('--n_layer', type = int, default = 2) # number of layers
parser.add_argument('--n_head', type = int, default = 6) # number of heads
parser.add_argument('--n_embd', type = int, default = 768)  # embedding dimension
parser.add_argument('--act_name', type = str, default = 'relu') # activation
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
parser.add_argument('--plots_dir', type = str, default = 'plots/lcg/multiple_ps/patching')
parser.add_argument('--save_sequences', type = str, default = 'False')
parser.add_argument('--save_checkpoint', type = str, default = 'True')
# Other
parser.add_argument('--show_tqdm_bar', type = str, default = 'True')
parser.add_argument('--shifts', type = int, default = 0)

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

config.chkpt_steps = [1000, 2000, 5000, 10_000, 20_000, 50_000, config.num_steps]


# Color
N = (config.p_max // 6) + 1  # number of colors to extract from each of the base_cmaps below
base_cmaps = ['Greys', 'Purples', 'Reds', 'Oranges', 'Blues', 'Greens']

n_base = len(base_cmaps)
# we go from 0.2 to 0.8 below to avoid having several whites and blacks in the resulting cmaps
colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.2, 0.8, N)) for name in base_cmaps])
custom_cmap = mcolors.ListedColormap(colors)


# set the internal precision of float32 matrix multiplications: 
# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
# “highest”, float32 matrix multiplications use the float32 datatype (24 mantissa bits with 23 bits explicitly stored) for internal computations.
# “high”, float32 matrix multiplications either use the TensorFloat32 datatype (10 mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers 
torch.set_float32_matmul_precision('high')
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32  # fp16 needs a further change of the code.

# create model
model = create_model(config)

# Load Checkpoints
if config.act_name.lower() == 'gelu':
    ckpt_path = f'{config.results_dir}/chkpt_cps_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}.pth'
else:
    ckpt_path = f'{config.results_dir}/chkpt_cps_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_{config.act_name}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}.pth'
    
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model = nnsight.models.NNsightModel.NNsight(model)
model.eval()
print(model)

# plot_path = f'{config.plots_dir}/p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}_'

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
    initial_seeds = torch.arange(p, dtype=torch.int64)[:128]

    def single_lcg(a, c, seed):
        @torch.compile
        def next_value(prev):
            return (a*prev + c) % p
        
        sequence = [seed]
        for _ in range(length - 1):
            sequence.append(next_value(sequence[-1]))
        
        return torch.stack(sequence)

    # Vectorize over a, c, and initial seeds
    results = torch.vmap(torch.vmap(single_lcg, in_dims=(None, None, 0)), in_dims=(0, 0, None), chunk_size=16)(a_flat, c_flat, initial_seeds)
    
    # Reshape to combine all sequences
    return results.reshape(a_flat.size(0), -1, length)


def patch_attn(model, tracer, x, x_mutate, config, layer_idx=0, head_idx=0):
    head_dim = config.n_embd // config.n_head
    
    # Clean forward pass
    with tracer.invoke(x) as invoker:
        with torch.autocast(device_type=device, dtype=dtype):
            logits_clean = model.lm_head.output.save()
    
    # Prepare the mutated feature
    with tracer.invoke(x_mutate) as invoker:
        with torch.autocast(device_type=device, dtype=dtype):
            per_head_contribution = model.transformer.h[layer_idx].attn.c_proj.input.save()
    
    # Forward pass of clean input, with features patched by mutated feature
    with tracer.invoke(x) as invoker:
        with torch.autocast(device_type=device, dtype=dtype):
            model.transformer.h[layer_idx].attn.c_proj.input[:, :, head_dim*head_idx : head_dim*(head_idx+1)] = 1.25 * per_head_contribution[:, :, head_dim*head_idx : head_dim*(head_idx+1)]
            logits_patched = model.lm_head.output.save()

    return logits_clean, logits_patched


with torch.inference_mode():
    
    chosen_p = 1024
    a_list = [5,]
    c_list = [31,]
    init_seed = 1
    
    for a in a_list:
        for c in c_list:
            seq_collections = lcg_vectorized_with_fixed_seed(p = chosen_p, length = config.context_len + 1, a_list = [a], c_list = [c]) # (all_ac_pairs, p, seq_len)
            for ac_id, seq in enumerate(seq_collections):
                x_chosen = seq[init_seed : init_seed+1, :-1].to(device)
                y_chosen = seq[init_seed : init_seed+1, 1:].to(device)
            
            seq_collections = lcg_vectorized_with_fixed_seed(p = config.p_eval, length = config.context_len + 1, a_list = [a], c_list = [c]) # (all_ac_pairs, p, seq_len)
            for ac_id, seq in enumerate(seq_collections):
                x = seq[init_seed : init_seed+1, :-1].to(device)
                y = seq[init_seed : init_seed+1, 1:].to(device)         
            
            for layer_idx in range(config.n_layer):                    
                for head_idx in range(config.n_head):
                    with model.trace() as tracer:
                        logits_clean, logits_patched = patch_attn(model, tracer, x, x_chosen, config, layer_idx=layer_idx, head_idx=head_idx)    
                    
                    preds_clean = logits_clean.argmax(-1).cpu()[0]
                    preds_patched = logits_patched.argmax(-1).cpu()[0]

                    plot_path = f'{config.plots_dir}/newp{chosen_p}a{a}c{c}_layer{layer_idx+1}_h{head_idx+1}_p{chosen_p}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}'
                    
                    y_temp = y[0].cpu()
                    y_chosen_temp = y_chosen[0].cpu()
                    
                    print(a, c, layer_idx, head_idx)
                    print('clean model, clean label:', (y_temp == preds_clean).float().mean().item(), 'patched model, clean label:', (y_temp == preds_patched).float().mean().item())            
                    print('clean model, patched label:', (y_chosen_temp == preds_clean).float().mean().item(), 'patched pred, patched label:', (y_chosen_temp == preds_patched).float().mean().item())
                    
                    fig, ax = plt.subplots(1, 1, figsize=(12, 3), constrained_layout=True)
                    
                    ax.plot(preds_clean, '^--', label='model prediction', alpha=0.25)
                    ax.plot(preds_patched, '*--', label='patched model prediction', alpha=0.8, color='red')
                    ax.hlines(y=chosen_p, xmin=0, xmax=config.context_len, color='black', linestyle='--', label=f'$m_{{patch}}=1024$')
                    ax.set_xticks(np.asarray([i for i in range(0, 256, 32)]) + 0.5)
                    ax.set_xticklabels([f'{i+1}' for i in range(0, 256, 32)])
                    ax.tick_params(axis='both', which='major', labelsize=20)
                    ax.tick_params(axis='x', which='both', length=8, width=2, bottom=True, labelbottom=True)
                    ax.tick_params(axis='y', which='both', length=8, width=2, left=True, labelleft=True)
                    
                    ax.set_xlim(-1, 256.5)
                    ax.set_xlabel('token position', fontsize=24)
                    ax.set_ylabel(f'$n$', fontsize=24)
                    ax.legend(fontsize=14, loc='upper right')
                    
                    fig.savefig(f'{plot_path}.pdf', format='pdf', dpi=dpi)
                    plt.close()            
                    