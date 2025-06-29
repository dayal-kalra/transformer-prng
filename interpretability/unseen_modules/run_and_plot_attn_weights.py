import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.prng_data as prngs_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors

from utils.gpt2 import GPT, GPTConfig, MLP, CausalSelfAttention

# usual imports
import argparse

plt.rcParams.update({"font.size": 20})
sns.set_theme(style="whitegrid")
dpi = 200
cmap = 'coolwarm'

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

"""PCA"""
@torch.inference_mode()
def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    idx = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, idx])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v

@torch.inference_mode()
def pca(embedding: torch.Tensor, components: tuple = (0, 1)) -> torch.Tensor:
    embedding = embedding.T  # (n_embd, vocab)
    mean = embedding.mean(dim=1, keepdim=True)  # (n_embd, 1)
    centered_data = embedding - mean

    U, S, Vt = torch.linalg.svd(centered_data.T, full_matrices=False)
    
    # Select the specified components
    selected_components = Vt[list(components), :]  # (len(components), n_embd)
    
    U, Vt = svd_flip(U, Vt)
    
    # Calculate and print the ratio of explained variance for the selected components
    total_variance = torch.sum(S)
    selected_variance = torch.sum(S[list(components)])
    variance_ratio = (selected_variance / total_variance).item()
    
    print(f'Selected components {components} explain {variance_ratio:.4f} of the total variance')
    
    return (selected_components + mean.T)

@torch.inference_mode()
def plot_embd_pca(model, components: tuple = (0, 1), base_save_path: str | None = None) -> None:
    wte = model.transformer.wte.weight.data.cpu()  # (vocab, n_embd)
    wpe = model.transformer.wpe.weight.data.cpu()  # (vocab, n_embd)
    
    wte_pca = pca(wte, components=components)
    wpe_pca = pca(wpe, components=components)
    
    fig, axs = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    # Token Embeddings
    ax = axs[0]
    for number, embd_v in enumerate(wte):
        results = (embd_v @ wte_pca[0], embd_v @ wte_pca[1])
        ax.scatter(results[0], results[1], alpha=0)
        ax.annotate(f'{number}', results, color=custom_cmap.colors[number], fontsize=5)

    ax.set_xlabel(f'PCA Component {components[0] + 1}')
    ax.set_ylabel(f'PCA Component {components[1] + 1}')
    ax.set_title("PCA of Token Embeddings")
    
    # Positional embedding
    ax = axs[1]
    for number, embd_v in enumerate(wpe):
        results = (embd_v @ wpe_pca[0], embd_v @ wpe_pca[1])
        ax.scatter(results[0], results[1], alpha=0)
        ax.annotate(f'{number}', results, color=custom_cmap.colors[number], fontsize=5)
    
    ax.set_xlabel(f'PCA Component {components[0] + 1}')
    ax.set_ylabel(f'PCA Component {components[1] + 1}')
    ax.set_title("PCA of Positional Embeddings")
    
    if base_save_path is not None:
        fig.savefig(base_save_path + f'embds_comp{components}.pdf', format='pdf', dpi=dpi)
    else:
        plt.show()
    plt.close()
    
    return

"""Data"""
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
    return results.reshape(len(c_list)*len(a_list), num_seeds, length)

"""Attn"""
@torch.inference_mode()
def set_noncausal_to_nan(attn_scores):
    """
    Set the non-causal part of the attention scores to NaN.
    """
    B, nh, T, _ = attn_scores.shape
    mask = torch.triu(torch.ones(T, T, device=attn_scores.device), diagonal=1).bool()
    mask = mask.expand(B, nh, T, T)
    attn_scores[mask] = float('nan')
    return attn_scores

@torch.inference_mode()
def compute_attn_and_ov(config, input_features: torch.Tensor, QKV: nn.Parameter, O: nn.Parameter) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    input_features: (batch_size, seq_len, n_embd)
    QKV: (3*n_head*head_dim, n_embd)
    O: (n_embd, n_embd)
    
    returns:
    - attn_scores: (batch_size, n_head, seq_len, seq_len)
    - VO_results: (batch_size, n_head, seq_len, n_embd)
    - outputs: (batch_size, n_head, seq_len, n_embd)
    """
    B, T, C = input_features.size()
    head_dim = config.n_embd // config.n_head

    # Linear transformation for QKV
    qkv = F.linear(input_features, QKV)
    q, k, v = qkv.split(config.n_embd, dim=-1)
    
    # Reshape for batch matrix multiplication
    q = q.view(B, T, config.n_head, C // config.n_head).transpose(1, 2)  # (B, nh, T, hs)
    k = k.view(B, T, config.n_head, C // config.n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, config.n_head, C // config.n_head).transpose(1, 2)  # (B, nh, T, hs)

    # Causal Mask
    attn_bias = torch.zeros(T, T, device=q.device, dtype=q.dtype)
    temp_mask = torch.ones(T, T, device=q.device, dtype=torch.bool).tril(diagonal=0)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(q.dtype)
    
    # Compute attention scores
    attn_score = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    attn_score += attn_bias
    attn_score = torch.softmax(attn_score, dim=-1)  # (B, hs, T, T)

    # Output 
    attn_output = attn_score @ v  # (B, nh, T, hs)
    outputs = torch.zeros((B, config.n_head, T, config.n_embd), device=q.device, dtype=q.dtype)
    for i in range(config.n_head):
        outputs[:, i, :, :] = F.linear(attn_output[:, i, :, :], O[:, i*head_dim : (i+1)*head_dim])
    
    # Compute VO_results
    VO_results = torch.zeros((B, config.n_head, T, config.n_embd), device=q.device, dtype=q.dtype)
    for i in range(config.n_head):
        VO_results[:, i, :, :] = F.linear(v[:, i, :, :], O[:, i*head_dim : (i+1)*head_dim])

    return set_noncausal_to_nan(attn_score).cpu(), VO_results.cpu(), outputs.cpu()

@torch.inference_mode()
def plot_attn_map(model, config, test_dataset, a, c, shifts=0, base_save_path: str | None = None):
    head_dim = config.n_embd // config.n_head
    
    # Register Hooks
    attn_hooks = list()
    mlp_hooks = list()
    for i, (id, m) in enumerate(model.named_modules()):
        # Attention Layers
        if isinstance(m, CausalSelfAttention):
            attn_hooks.append(Hook(m, backward=False))
        if isinstance(m, MLP):
            mlp_hooks.append(Hook(m.c_proj, backward=False))
    
    # Perform Forward Pass
    test_dataset = torch.roll(test_dataset, shifts=shifts, dims=-1)
    x = test_dataset[:128, :-1].to(device)
    y = test_dataset[:128, 1:].to(device)    
    with torch.autocast(device_type = device, dtype = dtype):
        logits, loss = model(x, y)
    
    pred = logits.argmax(-1)
    correct = (pred == y).float().mean()
    
    # Record Attention Map and MLP features        
    attn_layer_id = 0
    mlp_layer_id = 0
    sample_id = 1

    for i, (id, m) in enumerate(model.named_modules()):
        if isinstance(m, CausalSelfAttention):
            
            QKV = m.c_attn.weight
            O = m.c_proj.weight  # (fan_out, fan_in)
            
            attn_score, VO_results, outputs = compute_attn_and_ov(config, input_features=attn_hooks[attn_layer_id].input, QKV=QKV, O=O)
            
            # Attention Map Plotting
            fig, axs = plt.subplots(1, config.n_head, figsize=(5*config.n_head + 2, 5), constrained_layout=True)
            for head_id, ax in enumerate(axs):
                # Each head plot one sub-figure
                sns.heatmap(attn_score[sample_id, head_id, :, :],
                            ax=ax, 
                            vmin=0, 
                            vmax=1,
                            cmap=cmap,
                            cbar=True if head_id == config.n_head - 1 else False,
                            # xticklabels=5,
                            # yticklabels=5
                            )
                
                # ax.set_ylim(ax.get_ylim()[::-1])
                # ax.invert_yaxis()
                # ax.set_title(f'Head: {head_id + 1}')
                
                # Remove all spines
                # sns.despine(ax=ax, left=True, bottom=True)
                
                if head_id == 0:
                    ax.set_ylabel('query')
                ax.set_xlabel('key')
                
                ax.set_xlim([0, config.context_len])
                ax.set_ylim([0, config.context_len][::-1])
                # Ensure the heatmap fills the subplot area
                # ax.set_aspect('equal')
                tick_positions = np.arange(0, config.context_len, 32)
                
                ax.set_xticks(tick_positions + 0.5)
                ax.set_xticklabels(tick_positions, rotation=0)
                ax.set_yticks(tick_positions + 0.5)
                ax.set_yticklabels(tick_positions, rotation=45)
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.tick_params(axis='x', which='both', length=4, width=1, bottom=True, labelbottom=True)
                ax.tick_params(axis='y', which='both', length=4, width=1, left=True, labelleft=True)
                
            
            # fig.suptitle(f'Acc {correct.item() * 100:.2f}%, seed={sample_id}, a={a}, c={c}')
            if base_save_path is not None:
                fig.savefig(base_save_path + f'attn_seed{sample_id}_a{a}c{c}_shifts{shifts}_l{attn_layer_id + 1}.pdf', format='pdf', dpi=dpi)
            else:
                plt.show()
            plt.close()
            
            mlp_layer_id += 1
        
    return
    
    
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
parser.add_argument('--act_name', type = str, default = 'gelu') # activation
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
parser.add_argument('--results_dir', type = str, default = './chkpts')
parser.add_argument('--plots_dir', type = str, default = 'plots/lcg/multiple_ps/attn_weights')
parser.add_argument('--save_sequences', type = str, default = 'False')
parser.add_argument('--save_checkpoint', type = str, default = 'False')
# Other
parser.add_argument('--shifts', type = int, default = 0) # position of 1 to p_eval numbers in the sequence

config = parser.parse_args()
config.p_max = int(1.2 * config.p_eval) # p_max = p_val + p_offset / 2.0
config.p_min = config.context_len+1

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

for chosen_p in [2048]:
    if config.act_name.lower() == 'gelu':
        plot_path = f'{config.plots_dir}/cps{chosen_p}_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}_'
    else:
        plot_path = f'{config.plots_dir}/cps{chosen_p}_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_{config.act_name}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}_'

    # Test set
    # a_list = find_as(config.p_eval)
    # c_list = find_coprimes(config.p_eval)
    
    match chosen_p:
        case 2048:
            # a_list = [1, 5, 17, 65, 129, 193, 257, 513, 1025]
            # c_list = [1, 3, 5, 31, 51]
            a_list = [1,]
            c_list = [1, 3, 5, 7, 9, 15, 31,]
        case 2352:
            a_list = [1, 85, 589, 1681]
            c_list = [1, 5, 53]
        case _:
            raise NotImplementedError('Not implemented')
        
    seq_collections = lcg_vectorized_with_fixed_seed(p = chosen_p, length = config.context_len + 1, a_list = a_list, c_list = c_list) # (all_ac_pairs, p, seq_len)

    # plot_embd_pca(model, components=(1, 2), base_save_path=plot_path)
    # a_list = [1]
    # c_list = [1]
    for c_id, c in enumerate(c_list):
        for a_id, a in enumerate(a_list):
            if a < config.p_eval and c < config.p_eval:
                print(a, c)
                test_dataset = seq_collections[c_id * len(a_list) + a_id, :, :]
                plot_attn_map(model, config, test_dataset, a, c, shifts=config.shifts, base_save_path=plot_path)
        

# # save training data data
# df_train = pd.DataFrame(train_results, columns = ['step', 'lr_step', 'loss_step', 'accuracy_step', 'last_accuracy_step'])
# df_train['num_train'] = config.num_train
# path = f'{config.results_dir}/train_p{config.p_eval}_sl{config.seq_len}_na{config.num_as}_nc{config.num_cs}_pmin{config.p_min}_pmax{config.p_max}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_wd{config.weight_decay}.tab'
# df_train.to_csv(path, sep = '\t') 


# # save evaluation data
# df_eval = pd.DataFrame(eval_results, columns = ['step', 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'train_last_acc', 'test_last_acc'])
# df_eval['num_test'] = config.num_test
# path = f'{config.results_dir}/eval_p{config.p_eval}_sl{config.seq_len}_na{config.num_as}_nc{config.num_cs}_pmin{config.p_min}_pmax{config.p_max}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_wd{config.weight_decay}.tab'
# df_eval.to_csv(path, sep = '\t') 