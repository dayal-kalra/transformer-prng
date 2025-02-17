import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.prng_data as prngs_data
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from utils.gpt2 import GPT, GPTConfig, MLP, CausalSelfAttention

# usual imports
import argparse

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
    def __init__(self, module: nn.Module, backward: bool = False, config = None):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_full_backward_hook(self.hook_fn)
        
        self.config = config

    def hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        with torch.no_grad():
            if self.config is not None:
                B, T, C = input[0].shape
                head_dim = C // self.config.n_head
                self.input = input[0].view(B, T, self.config.n_head, head_dim) # (B, T, nh, nc // nh)
                self.output = torch.zeros(B, T, self.config.n_head, C, device=self.input.device, dtype=self.input.dtype)  # per head projection
                O = module.weight.data
                for head_idx in range(self.config.n_head):
                    self.output[:, :, head_idx, :] = self.input[:, :, head_idx, :] @ O.T[head_idx * head_dim: (head_idx + 1) * head_dim, :]
            else:
                self.input = input[0]
                self.output = output

    def close(self):
        self.hook.remove()


"""Hook"""
@torch.inference_mode()
def hook_layers(model, config):
    hook_dict = dict()
    
    for i, (id, m) in enumerate(model.named_modules()):
        # Attention Layers
        if 'attn.c_proj' in id:
            hook_dict[id] = Hook(m, backward=False, config=config)
        
    return hook_dict


@torch.inference_mode()
def compute_cosine_similarity(model, hook_dict):
    """Calculate cosine similarity between intermediate features and embedding layers"""
    cosine_dict_embd = dict()
    # cosine_dict_ln_embd = dict()
    # cosine_dict_lmhead = dict()
    # cosine_dict_ln_lmhead = dict()
    
    wte = model.transformer.wte.weight.data
    # lm_head = model.lm_head.weight.data
    
    wte_normalized = F.normalize(wte, p=2, dim=-1)
    # lm_head_normalized = F.normalize(lm_head, p=2, dim=-1)
    
    for key, hook in hook_dict.items():
        # Normalize hook output
        hook_output_normalized = F.normalize(hook.output, p=2, dim=-1)
        
        # with torch.autocast(device_type=device, dtype=dtype):
            # hook_output_normalized_ln = F.normalize(model.transformer.ln_f(hook.output), p=2, dim=-1)
        
        # Calculate cosine similarity using einsum
        result_wte = torch.einsum('bche,ve->bchv', hook_output_normalized.float(), wte_normalized)
        cosine_dict_embd[key] = result_wte.cpu().numpy()
        
        # # Calculate cosine similarity using einsum
        # result_ln_wte = torch.einsum('bce,ve->bcv', hook_output_normalized_ln.float(), wte_normalized)
        # cosine_dict_ln_embd[key] = result_ln_wte.cpu().numpy()
        
        # # Head
        # result_lmhead = torch.einsum('bce,ve->bcv', hook_output_normalized.float(), lm_head_normalized)
        # cosine_dict_lmhead[key] = result_lmhead.cpu().numpy()
        
        # # LN + Head
        # result_ln_lmhead = torch.einsum('bce,ve->bcv', hook_output_normalized_ln.float(), lm_head_normalized)
        # cosine_dict_ln_lmhead[key] = result_ln_lmhead.cpu().numpy()
    
    # return (cosine_dict_embd, cosine_dict_ln_embd), (cosine_dict_lmhead, cosine_dict_ln_lmhead)
    return (cosine_dict_embd, None), (None, None)


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
config.p_max = int(1.2 * config.p_eval) # p_max = p_val + p_offset / 2.0
config.p_min = config.context_len+1

activations = {'relu': torch.nn.ReLU(), 'gelu': torch.nn.GELU()}
config.act_fn = activations[config.act_name]

config.vocab_size = config.p_max # vocab size is equal to the maximum p
if config.p_min < config.context_len+1:
    config.p_min = config.context_len+1
    print(f'p_min set to context length')

# if I am not wrong, this seed only takes care of torch and not numpy
np.random.seed(config.main_seed)
torch.manual_seed(config.main_seed)
torch.cuda.manual_seed(config.main_seed)
random.seed(config.main_seed)

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
model.eval()

chosen_p = 1800

if config.act_name.lower() == 'gelu':
    plot_path = f'{config.plots_dir}/cps{chosen_p}_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}_'
else:
    plot_path = f'{config.plots_dir}/cps{chosen_p}_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_{config.act_name}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}_'

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
    initial_seeds = torch.arange(p, dtype=torch.int64)[:256]

    def single_lcg(a, c, seed):
        @torch.compile
        def next_value(prev):
            return (a*prev + c) % p
        
        sequence = [seed]
        for _ in range(length - 1):
            sequence.append(next_value(sequence[-1]))
        
        return torch.stack(sequence)

    # Vectorize over a, c, and initial seeds
    results = torch.vmap(torch.vmap(single_lcg, in_dims=(None, None, 0)), in_dims=(0, 0, None))(a_flat, c_flat, initial_seeds)
    
    # Reshape to combine all sequences
    return results.reshape(-1, length)


with torch.inference_mode():
    # get the first batch of data
    # x, y = next(iter(test_loader))
    # x = x.to(device)
    # y = y.to(device)
    chosen_p = config.p_eval
    a_list = [1, 5, 17, 65, 129, 193, 257, 513, 1025]
    c_list = [1, 3, 5, 15, 31, 51]
    # chosen_p = 256
    # a_list = [1, 33, 65, 89, 129]
    # c_list = [1, 13, 51]
    
    for a in a_list:
        for c in c_list:
            if a < chosen_p and c < chosen_p:
                seq = lcg_vectorized_with_fixed_seed(p = chosen_p, length = config.context_len + 1, a_list = [a], c_list = [c])
                # seq = torch.roll(seq, shifts=config.shifts, dims=-1)
                x = seq[:16, :-1].to(device)
                y = seq[:16, 1:].to(device)
                
                # print(seq.shape)
                hook_dict = hook_layers(model, config)
                with torch.autocast(device_type=device, dtype=dtype):
                    logits, loss = model(x, y)
                
                (cosine_dict_embd, _), (_, _) = compute_cosine_similarity(model, hook_dict)
                
                preds = torch.argmax(logits, dim=-1)
                
                data_dict = {'seq': seq[:16].cpu().numpy(),
                            'pred': preds.cpu().numpy(),
                            'cosine_embd': cosine_dict_embd,
                            # 'cosine_ln_embd': cosine_dict_ln_embd,
                            # 'cosine_lmhead': cosine_dict_lmhead,
                            # 'cosine_ln_lmhead': cosine_dict_ln_lmhead
                            }
                
                if config.act_name.lower() == 'gelu':
                    np.save(f'{config.interp_results_dir}/cosine_a{a}c{c}_shifts{config.shifts}_cps{chosen_p}_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}.npy', data_dict)
                else:
                    np.save(f'{config.interp_results_dir}/cosine_a{a}c{c}_shifts{config.shifts}_cps{chosen_p}_p{config.p_eval}_T{config.period_min}_T{config.period_max}_Tn{config.context_len}_na{config.num_as}_nc{config.num_cs}_np{config.num_ps}_ne{config.num_examples}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_{config.act_name}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.step}_B{config.batch_size}_wd{config.weight_decay}.npy', data_dict)