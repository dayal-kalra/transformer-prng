import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors


# usual imports
import argparse


plt.rcParams.update({"font.size": 16})
sns.set_theme(style="darkgrid")
dpi = 100
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
parser.add_argument('--interp_results_dir', type = str, default = 'results/lcg/multiple_ps')
parser.add_argument('--plots_dir', type = str, default = 'plots/lcg/multiple_ps/cosine')
parser.add_argument('--save_sequences', type = str, default = 'False')
parser.add_argument('--save_checkpoint', type = str, default = 'False')
# Other
parser.add_argument('--show_tqdm_bar', type = str, default = 'True')
parser.add_argument('--shifts', type = int, default = 0)

config = parser.parse_args()
config.p_max = config.p_eval + int(config.num_ps / 2.0) # p_max = p_val + p_offset / 2.0
config.p_min = config.p_eval - config.num_ps

activations = {'relu': torch.nn.ReLU(), 'gelu': torch.nn.GELU()}
config.act_fn = activations[config.act_name]

config.vocab_size = config.p_max # vocab size is equal to the maximum p
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

with torch.inference_mode():
    
    chosen_p = config.p_eval
    a_list = [257, 1]
    c_list = [31, 3, 5, 15, 51, 1]

    for a in a_list:
        for c in c_list:
            if a < chosen_p and c < chosen_p:
                if config.act_name.lower() == 'gelu':
                    data_dict = np.load(f'{config.interp_results_dir}/cosine_a{a}c{c}_shifts{config.shifts}_cps{chosen_p}_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}.npy', allow_pickle=True).item()
                else:
                    data_dict = np.load(f'{config.interp_results_dir}/cosine_a{a}c{c}_shifts{config.shifts}_cps{chosen_p}_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_{config.act_name}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}.npy', allow_pickle=True).item()

                for seed in [4]:
                    for saved_key in ['cosine_embd']:
                        plot_path = f'{config.plots_dir}/{saved_key}_a{a}c{c}_shifts{config.shifts}_cps{chosen_p}_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}_'

                        for layer_idx, (key, value) in enumerate(data_dict['cosine_embd'].items()):
                            if '0' in key:  # Layer 1
                                for head_idx in range(value.shape[2]):
                                    if head_idx == 5:  # Head 6
                                        value_to_plot = value[:, :, head_idx, :]
                                        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
                                        print(a, c, value_to_plot.shape, value_to_plot.T.max(), value_to_plot.min())
                                        print(value_to_plot[seed].argmax(-1), '\n', data_dict['seq'][seed].max(-1), '\n', data_dict['seq'][seed])
                                        sns.heatmap(value_to_plot[seed].T, 
                                                    vmin=value.min(), 
                                                    vmax=value.max(), 
                                                    cmap='coolwarm', 
                                                    ax=ax)
                                        ax.invert_yaxis()
                                        
                                        ax.set_ylabel(f'$x_{{n}}$')
                                        ax.set_xlabel(f'ctx')
                                        ax.set_xlim(0, config.context_len // 2)
                                        # ax.set_ylim([0, config.vocab_size][::-1])
                                        # ax.set_title(f'layer {layer_idx + 1} | head{head_idx + 1} | a={a} | c={c}')
                                        
                                        ytick_positions = np.arange(0, config.vocab_size, 512)
                                        xtick_positions = np.arange(0, config.context_len // 2, 32)
                                        
                                        ax.set_xticks(xtick_positions + 0.5)
                                        ax.set_xticklabels(xtick_positions, rotation=45)
                                        ax.set_yticks(ytick_positions + 0.5)
                                        ax.set_yticklabels(ytick_positions, rotation=45)
                                        
                                        ax.tick_params(axis='both', which='major', labelsize=12)
                                        ax.tick_params(axis='x', which='both', length=4, width=1, bottom=True, labelbottom=True)
                                        ax.tick_params(axis='y', which='both', length=4, width=1, left=True, labelleft=True)
                                        
                                        # fig.savefig(f'{plot_path}seed{seed}_{key}_h{head_idx + 1}.pdf', format='pdf', dpi=dpi)
                                        plt.close()