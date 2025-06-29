import math
import torch
import utils.prng_data as prngs_data
import utils.prng_tests as prng_tests
import numpy as np

from utils.gpt2 import GPT, GPTConfig
from utils.schedules import linear_warmup_cosine
from utils.prng_data import lcg, lcg_vectorized
from utils.datasets import PRNGsDataset

## usual imports
import pickle as pl
import pandas as pd
import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import psutil
import random

from tqdm.auto import tqdm

## Toggle to true if you want to use GPU
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = 'cuda'
elif USE_GPU and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
    
## Set the internal precision of float32 matrix multiplications: 
torch.set_float32_matmul_precision('high')
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32  # fp16 needs a further change of the code.
  

def main(config):
    config.n_head = config.n_embd // config.head_dim
    config.num_ps = 1
    config.num_as = max(1, math.isqrt(config.total_examples))
    config.num_cs = config.num_as
    config.act_name = config.act_name.lower()

    config.vocab_size = config.p_eval # vocab size is equal to the maximum p
    print(config)

    ## Set random seeds
    random.seed(config.main_seed)
    np.random.seed(config.main_seed)
    torch.manual_seed(config.main_seed)
    torch.cuda.manual_seed(config.main_seed)
    

    ## Find parameters for prngs: for a given p_eval, find (a, c) pairs. Use the Hull-Dobell theorem for the test pairs.
    train_as, train_cs, val_as, val_cs = find_prng_parameters(config)
    config.num_as = min(len(train_as), config.num_as)
    config.num_cs = min(len(train_cs), config.num_cs)
    print(f'Possible a values: {len(train_as)}, possible c values: {len(train_cs)}')
    

    ## Generate train data using different a's and c's
    print(f'Generating train data with randomly chosen {config.num_as} a values and {config.num_cs} c values for each p')
    examples_lst = generate_dataset(config, train_as, train_cs)

    ## combine the training datasets for training
    combined_dataset = torch.utils.data.ConcatDataset([PRNGsDataset(value) for value in examples_lst])

    ## create train loader
    train_loader = torch.utils.data.DataLoader(combined_dataset, batch_size = config.batch_size, shuffle = True, num_workers = 0, drop_last=True) 
    config.num_train = len(train_loader)

    ## Generate test data using different a's an c's
    print(f'Generating test data with {len(val_as)} values of a and {len(val_cs)} values of c')
    test_data = lcg_vectorized(p = config.p_eval, length =  config.context_len+1, a_list = val_as, c_list = val_cs, num_examples = 1, chunk_size = config.chunk_size)

    ## create test loader
    test_loader = torch.utils.data.DataLoader(PRNGsDataset(test_data), batch_size = config.batch_size, shuffle = True, num_workers = 0, drop_last=True)
    config.num_test = len(test_loader.dataset)

    seq_collections = lcg_vectorized_with_fixed_seed(p = config.p_eval, length = config.context_len + 1, a_list = val_as, c_list = val_cs) # (all_ac_pairs, p, seq_len)


    ## Define the paths for saving checkpoints and training results
    train_path = f'{config.results_dir}/train_np1_p{config.p_eval}_Tn{config.context_len}_N{config.total_examples}_ne{config.num_examples_per_prng}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_wd{config.weight_decay}.tab'
    train_dataset_path = f'{config.results_dir}/train_dataset_np1_p{config.p_eval}_Tn{config.context_len}_N{config.total_examples}_ne{config.num_examples_per_prng}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_wd{config.weight_decay}.tab'
    eval_path = f'{config.results_dir}/eval_p{config.p_eval}_np1_Tn{config.context_len}_N{config.total_examples}_ne{config.num_examples_per_prng}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_wd{config.weight_decay}.tab'


    ## train model only if the path does not exist -- to avoid over-writing.
    assert not os.path.exists(eval_path), "The configuration for this run already exists in current results. Change it to avoid over-writing."
    
    
    ## create model
    model = create_model(config)


    ## create optimizer
    optimizer = model.configure_optimizers(weight_decay=config.weight_decay, learning_rate=6e-4, beta1=config.beta1, beta2=config.beta2, device=device)

    # Compile model, both models point to the same underlying ram but compiled model has some more static stuff for faster computations
    # unoptimized_model = model
    # model = torch.compile(model)
    # model = unoptimized_model

    ## Optimization
    model, optimizer, train_results, eval_results = train_and_evaluate(config, model, optimizer, train_loader, test_loader, seq_collections, val_as, val_cs)

    ## Save training and evaluation data
    df_train = pd.DataFrame(train_results, columns = ['step', 'lr_step', 'loss_step'])
    df_train['num_train'] = config.num_train
    df_train.to_csv(train_path, sep = '\t') 

    df_eval = pd.DataFrame(eval_results, columns = ['step', 'train_loss', 'test_loss', 'test_accuracy', 'test_token_accuracy', 'token_idx'])
    df_eval['num_test'] = config.num_test
    df_eval.to_csv(eval_path, sep = '\t')

    ## Save checkpoint
    if config.save_checkpoint == 'True':
        path = f'{config.results_dir}/{config.act_name}/chkpt_r_star_p{config.p_eval}_Tn{config.context_len}_N{config.total_examples}_ne{config.num_examples_per_prng}_n{config.n_embd}_h{config.n_head}_d{config.n_layer}_I{config.main_seed}_lr{config.lr_trgt:0.6f}_Tw{config.warmup_steps}_T{config.num_steps}_B{config.batch_size}_wd{config.weight_decay}.pth'
        torch.save({
                'step': config.num_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, path)
    

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
    ### Model hyperparams
    parser.add_argument('--n_layer', type = int, default = 1) # number of layers
    parser.add_argument('--n_embd', type = int, default = 768)  # embedding dimension
    parser.add_argument('--n_head', type = int, default = 1) # number of heads
    parser.add_argument('--head_dim', type = int, default = 768) # number of heads
    parser.add_argument('--act_name', type = str, default = 'relu') # activation function
    ### Optimization hyperparams
    parser.add_argument('--num_steps', type = int, default = 100_000) # number of training steps
    parser.add_argument('--warmup_steps', type = int, default = 2048) # number of warmup steps
    parser.add_argument('--lr_trgt', type = float, default = 3e-4) # the target learning rate
    parser.add_argument('--lr_init', type = float, default = 1e-6) # initial learning rate
    parser.add_argument('--lr_min', type = float, default = 1e-6) # final learning rate
    parser.add_argument('--batch_size', type = int, default = 256) # batch size
    # adamw hyperparams
    parser.add_argument('--weight_decay', type = float, default = 1.0) # weight decay
    parser.add_argument('--beta1', type = float, default = 0.9) # beta1 
    parser.add_argument('--beta2', type = float, default = 0.99) # beta2
    ### Evaluation hyperparams
    parser.add_argument('--eval_interval', type = int, default = 1000) # evaluation interval
    parser.add_argument('--results_dir', type = str, default = 'results')
    # Other
    parser.add_argument('--show_tqdm_bar', type = str, default = 'True')
    parser.add_argument('--save_sequences', type = str, default = 'False')
    parser.add_argument('--save_checkpoint', type = str, default = 'False')
    parser.add_argument('--save_train_dataset_metrics', type = str, default = 'False')
    
    return parser.parse_args()


def find_prng_parameters(config):
    """ For a given p, find the possible values of a's and c's according to the Hull–Dobell Theorem: https://en.wikipedia.org/wiki/Linear_congruential_generator """

    a_list = prngs_data.find_as(config.p_eval)
    c_list = prngs_data.find_coprimes(config.p_eval)

    val_as = np.random.choice(a_list, min(64, len(a_list)))
    val_cs = np.random.choice(c_list, min(64, len(c_list)))

    ## use arbitary (a, c) as long as its not in val_a and val_c
    train_as = [i for i in range(1, config.p_eval) if i not in val_as]
    train_cs = [i for i in range(1, config.p_eval) if i not in val_cs]

    return train_as, train_cs, val_as, val_cs


def generate_dataset(config, train_as, train_cs):
    print(f'p={config.p_eval}, Train as: {len(train_as)}, Train cs: {len(train_cs)}')

    examples_lst = []

    max_num_iter = 10_000
    num_iter = 0
    ps = set()

    show_tqdm_bar = (config.show_tqdm_bar == 'True')
    with tqdm(total = max_num_iter, desc = "Processing Examples", unit = "step", dynamic_ncols=True, disable=(not show_tqdm_bar)) as pbar:

        while len(examples_lst) < config.total_examples:
            pbar.set_postfix({'examples': len(examples_lst)})
            pbar.update(1)
            num_iter += 1
            
            # subset_as = np.random.choice(train_as, config.num_as, replace = False)
            # subset_cs = np.random.choice(train_cs, config.num_cs, replace = False)
            shuffled_as = np.random.permutation(train_as)
            shuffled_cs = np.random.permutation(train_cs)
            for a, c in zip(shuffled_as, shuffled_cs):
                data = lcg(p=config.p_eval, length=config.context_len+1, a=a, c=c, num_examples=config.num_examples_per_prng)
                examples_lst.append(data)
            
            if num_iter > max_num_iter:
                print(f'Max iterations reached')
                break  
    print_memory_usage(f"end of data generation")
    return examples_lst[:config.total_examples]


def print_memory_usage(location):
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    print(f"Memory usage at {location}: {memory_gb:.2f} GB")
    # print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")


@torch.inference_mode()
def lcg_vectorized_with_fixed_seed(p: int = 512, length: int = 8, a_list: list = [45], c_list: list = [123]) -> torch.Tensor:
    """
    Vectorized version of lcg function with sequential seeds starting from 0 to p-1,
    supporting multiple 'a' and 'c' values.
    """
    ## Create mesh grid and flatten
    a_mesh, c_mesh = np.meshgrid(a_list, c_list)
    a_flat = torch.tensor(a_mesh.flatten(), dtype=torch.int64)
    c_flat = torch.tensor(c_mesh.flatten(), dtype=torch.int64)

    ## Generate initial seeds from 0 to min(127, p-1)
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

    ## Vectorize over a, c, and initial seeds
    results = torch.vmap(torch.vmap(single_lcg, in_dims=(None, None, 0)), in_dims=(0, 0, None), chunk_size=32)(a_flat, c_flat, initial_seeds)
    
    ## Reshape to combine all sequences
    return results.reshape(len(c_list), len(a_list), num_seeds, length)


def create_model(config):
    gptconfig = GPTConfig(block_size=config.context_len, n_embd=config.n_embd, n_head=config.n_head, vocab_size=config.vocab_size, n_layer=config.n_layer, act_name=config.act_name)
    model = GPT(gptconfig)
    model.to(device)
    return model


def train_and_evaluate(config, model, optimizer, train_loader, test_loader, seq_collections, val_as, val_cs):
    
    # Create empty lists for storing the results
    train_results = list()
    eval_results = list()
    train_loss = 0
    
    ## Train mode
    model.train()

    ## Estimate the number of epochs
    num_epochs = np.ceil(config.num_steps / len(train_loader)).astype(int)
    print(f'Number of steps per epoch: {len(train_loader)}') 

    ## Optimization
    step = 0
    show_tqdm_bar = (config.show_tqdm_bar == 'True')
    
    with tqdm(total=config.num_steps, desc="", unit="step", dynamic_ncols=True, disable=(not show_tqdm_bar)) as pbar:
        while step < config.num_steps:
            for x, y in train_loader:
                step += 1
                # forward pass
                optimizer.zero_grad(set_to_none=True) # zero the grads

                x, y = x.to(device), y.to(device)

                # forward pass in bfloat16
                with torch.autocast(device_type = device, dtype = dtype):
                    logits, loss = model(x, y) 
                # compute loss and accuracy
                loss_step = loss.item()
                train_loss += loss_step

                # compute gradients
                loss.backward()
                # clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # update the learning rate
                lr_step = linear_warmup_cosine(step, config.lr_init, config.lr_min, config.lr_trgt, config.warmup_steps, config.num_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_step
                # take an optimizer step
                optimizer.step()    

                train_results.append([step, lr_step, loss_step])

                pbar.update(1)  # update the tqdm progress bar
                if step % (config.eval_interval if step >= 1000 else 100) == 0:
                    # normalize training metrics
                    train_loss = train_loss / config.eval_interval
                    
                    columns = np.column_stack((
                                    np.full(config.context_len, step),
                                    np.full(config.context_len, train_loss),
                                    ))
                    
                    # estimate test metrics
                    eval_columns = evaluate_model(config, model, test_loader)
                    
                    result_step = np.hstack((columns, eval_columns))
                    test_loss = eval_columns[0][0]
                    test_acc = eval_columns[0][1]
                    
                    eval_results.append(result_step)                
                                    
                    pbar.set_postfix({"lr": f'{lr_step:.2e}',
                                    "loss": f'{loss.item():.2e}',
                                    "test_loss": f'{test_loss:.2e}',
                                    "test_acc": f'{test_acc * 100:.2f}%',
                                    }
                                    )
                
                    # reset train metrics
                    train_loss = 0
    train_results = np.array(train_results)        
    eval_results = np.vstack(eval_results)
    return model, optimizer, train_results, eval_results

    
@torch.inference_mode()
def evaluate_model(config, model, test_loader):
    
    test_loss = 0.0
    test_acc = 0
    token_correct = torch.zeros(config.context_len, device = device)

    model.eval()

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type = device, dtype = dtype):
            logits, loss = model(x, y) 
        # estimate loss
        test_loss += loss.item()  # Sum up batch loss
        # estimate accuracy
        preds = logits.argmax(dim = -1)
        B, T = preds.shape

        # estimate average accuracy
        test_acc += torch.sum(preds == y).item()
        # Per-token accuracy
        token_correct += torch.sum(preds == y, dim = 0)
    
    model.train()
    # normalize the metrics
    test_loss = test_loss / len(test_loader) 
    test_acc = test_acc / len(test_loader) / config.batch_size / config.context_len
    token_acc = token_correct / len(test_loader) / config.batch_size

    # Create numpy array
    result = np.column_stack((
        np.full(config.context_len, test_loss),
        np.full(config.context_len, test_acc),
        token_acc.cpu().numpy(),
        np.arange(config.context_len)
    ))

    return result


if __name__ == "__main__":
    config = parse_args_fn()
    main(config)