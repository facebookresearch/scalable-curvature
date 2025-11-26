"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import pickle
from contextlib import nullcontext
import argparse

import numpy as np
import pandas as pd

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from utils.gpt import GPTConfig, GPT
from utils.schedules_utils import warmup_stable_decay
import utils.sharpness_func_utils as sharpness_utils 
import utils.sharpness_cupy_utils as sharpness_cupy_utils 
from utils.critical_learning_rate import compute_critical_learning_rate
from utils.loss_functions import CrossEntropyLoss
from utils.data_storage import DataStorageGeneral as DataStorage
from utils.data_loader import get_batch

torch.backends.cuda.matmul.allow_tf32 = True # enable tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # enable tf32 on cudnn

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(ctx, model, loss_fn, eval_steps, data_dir, context_len, batch_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            X, Y = get_batch(data_dir, split, context_len, batch_size, device)
            with ctx:
                logits = model(X)
                loss = loss_fn(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def estimate_critical_lr(ctx, model, loss_fn, optim, eval_steps, data_dir, context_len, batch_size, device):
    lrs_mean = {}
    lrs_std = {}
    steps = {}
    lr_guess = 1e-04

    for split in ['train', 'val']:
        critical_lrs = torch.zeros(eval_steps)
        num_steps = torch.zeros(eval_steps, dtype = torch.int32)
        for k in range(eval_steps):
            X, Y = get_batch(data_dir, split, context_len, batch_size, device)
            (lr_lower, lr_upper), num_iters_critical = compute_critical_learning_rate(ctx = ctx, model = model, loss_fn = loss_fn, optim = optim, batch = (X, Y), lr_guess = lr_guess, tol_power = cfg.crit_lr_tol_power, recompute_grads = True)
            # print(lr_lower, lr_upper, num_iters_critical)
            optim.zero_grad(set_to_none = True)
            # save the critical learning rate
            critical_lr = (lr_upper + lr_lower) / 2.0
            critical_lrs[k] = critical_lr
            lr_guess = critical_lr
            num_steps[k] = num_iters_critical
        # compute the mean and std of the critical learning rates
        lrs_mean[split] = critical_lrs.mean().item()
        lrs_std[split] = critical_lrs.std().item()
        # compute the max number of steps taken to estimate the critical learning rate
        steps[split] = num_steps.max().item()
    model.train()
    return lrs_mean, lrs_std, steps


def estimate_sharpness(ctx, model, loss_fn, optim, eval_steps, data_dir, context_len, batch_size, device, max_iters = 200, sharpness_tol = 1e-06):
    sharpness_lst = []
    pre_sharpness_lst = []

    eigvec = None 
    pre_eigvec = None

    split = 'train'

    for k in range(eval_steps):
        X, Y = get_batch(data_dir, split, context_len, batch_size, device)
        # estimate pre-sharpness
        pre_sharpness, pre_eigvec, num_iters_pre_sharpness = sharpness_cupy_utils.get_pre_sharpness_lobpcg(model, loss_fn, optim, (X, Y), eigvecs = pre_eigvec, tol = cfg.sharpness_tol, max_iters = cfg.max_iters)
        # clear gradients
        optim.zero_grad(set_to_none = True)

        print(f'Pre-sharpness: {pre_sharpness:.4e}, num_iters_pre_sharpness: {num_iters_pre_sharpness}')

        if num_iters_pre_sharpness < max_iters:
            pre_sharpness_lst.append(pre_sharpness)
        
        # estimate sharpness
        sharpness, eigvec, num_iters_sharpness = sharpness_cupy_utils.get_sharpness_lobpcg(model, loss_fn, (X, Y), eigvecs = eigvec, tol = cfg.sharpness_tol, max_iters = cfg.max_iters)
        print(f'Sharpness: {sharpness:.4e}, num_iters_sharpness: {num_iters_sharpness}')
        # clear gradients
        optim.zero_grad(set_to_none = True)

        if num_iters_sharpness < max_iters:
            sharpness_lst.append(sharpness)

    # if no sharpness was estimated, return 0
    pre_sharpness_converged = False
    pre_sharpness_std = 0.0
    pre_sharpness_mean = pre_sharpness
    if len(pre_sharpness_lst) > 1:
        pre_sharpness_mean = np.mean(pre_sharpness_lst)
        pre_sharpness_converged = True
    if len(pre_sharpness_lst) > 1:
        pre_sharpness_std = np.std(pre_sharpness_lst)

    sharpness_mean = sharpness
    sharpness_std = 0.0
    sharpness_converged = False        
    if len(sharpness_lst) > 0:
        sharpness_mean = np.mean(sharpness_lst)
        sharpness_converged = True
    
    if len(sharpness_lst) > 1:
        sharpness_std = np.std(sharpness_lst)
    
    return pre_sharpness_mean, pre_sharpness_std, pre_sharpness_converged, sharpness_mean, sharpness_std, sharpness_converged
    
def get_ckpt_filename(cfg, num_steps):
    base_filename = (
        f'{cfg.dataset_name}_'
        f'v{cfg.vocab_size}_'
        f'{cfg.model_name}_'
        f'var{cfg.init_var:0.1f}_'
        f'd{cfg.num_layers}_'
        f'h{cfg.num_heads}_'
        f'n{cfg.embd_dim}_'
        f'c{cfg.context_len}_'
        f'{cfg.optim_name}_'
        f'Tw{cfg.warmup_steps}_'
        f'r{cfg.warmup_exponent}_'
        f'Ts{cfg.stable_steps}_'
        f'{cfg.decay_schedule_name}_'
        f'p{cfg.decay_exponent}_'
        f'T{num_steps}_'
        f'ga{cfg.gradient_accumulation_steps}_'
        f'lr{cfg.lr_peak:.0e}_'
        f'lr{cfg.lr_min_factor}_'
        f'wd{cfg.weight_decay}_'
        f'bs{cfg.batch_size}_'
        f'b{cfg.beta1}_'
        f'b{cfg.beta2}_'
        f'eps{cfg.eps}_'
        f'gc{cfg.grad_clip}'
    )
    return base_filename

def get_base_filename(cfg, num_steps):
    add_filename = (
        f'tol{cfg.crit_lr_tol_power}_'
        f'rg{cfg.recompute_grads}_'
        f'k{cfg.topk}_'
        f'n{cfg.max_iters}_'
        f'tol{cfg.sharpness_tol}'
    )
    get_ckpt_filename_base = get_ckpt_filename(cfg, num_steps)
    base_filename = f'{get_ckpt_filename_base}_{add_filename}'
    return base_filename


def create_train_state(cfg, device, ckpt_steps):
    " Creates the model, optimizer, and loss function for training. It does not move the model to the device or compile it. "
    # create model
    gpt_conf = GPTConfig(
        context_len = cfg.context_len,
        vocab_size = cfg.vocab_size,
        num_layers = cfg.num_layers,
        num_heads = cfg.num_heads,
        embd_dim = cfg.embd_dim,
        bias = cfg.use_bias,
        init_var = cfg.init_var,
        use_flash = cfg.use_flash
    )
    
    model = GPT(gpt_conf)

    # create optimizer
    optim = model.configure_optimizers(
        learning_rate = cfg.lr_init, 
        betas = (cfg.beta1, cfg.beta2),
        eps = cfg.eps,
        weight_decay = cfg.weight_decay, 
        device_type = cfg.device_type
    )
    
    loss_fn = CrossEntropyLoss()
    
    return model, loss_fn, optim

def evaluate_ckpts_on_datasets(cfg, device):
    
    ckpt_steps_lst = [cfg.ckpt_interval*i  for i in range(0, cfg.num_steps//cfg.ckpt_interval + 1)]
    print(f'Evaluating checkpoints at steps: {ckpt_steps_lst}')

    datasets = ['fineweb', 'wikitext', 'alpaca', 'gsm8k']
        
    # 'pre_sharpness', 'pre_num_iters_power', 'sharpness', 'num_iters_power'
    eval_results = DataStorage(columns = ['ckpt_step', 'lr_step', 'train_loss', 'val_loss', 'dataset'])
    sharpness_results = DataStorage(columns = ['ckpt_step', 'lr_step', 'pre_sharpness', 'pre_sharpness_std', 'pre_sharpnes_converged', 'sharpness', 'sharpness_std', 'sharpness_converged', 'dataset'])
    forward_results = DataStorage(columns = ['ckpt_step', 'lr_step', 'critical_lr_train_mean', 'critical_lr_train_std', 'critical_lr_val_mean', 'critical_lr_val_std', 'num_iters_train', 'num_iters_val', 'dataset'])
        
    for ckpt_steps in ckpt_steps_lst:
        
        model, loss_fn, train_optim = create_train_state(cfg, device, ckpt_steps) # move model to device and compile it later

        downstream_optim = model.configure_optimizers(
            learning_rate = cfg.lr_init, 
            betas = (cfg.beta1, cfg.beta2),
            eps = cfg.eps,
            weight_decay = cfg.weight_decay, 
            device_type = cfg.device_type
        )

        lr_step = warmup_stable_decay(step = ckpt_steps, init_value = cfg.lr_init, peak_value = cfg.lr_peak, min_value = cfg.lr_min, num_steps = cfg.num_steps, warmup_steps = cfg.warmup_steps, stable_steps = cfg.stable_steps, warmup_exponent = cfg.warmup_exponent, decay_schedule_name = cfg.decay_schedule_name, decay_exponent = cfg.decay_exponent)

        ckpt_filename = get_ckpt_filename(cfg, ckpt_steps)
        ckpt_path = os.path.join(cfg.ckpt_dir, f'{ckpt_filename}.ckpt')

        if os.path.exists(ckpt_path):
            print(f'Loading checkpoint from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location = device, weights_only = False)
            
            # load weights from the model
            state_dict = ckpt['model']
            train_steps = ckpt['step']
            
            # remove the unwanted prefix
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
            # load the state dict and optimizer state
            model.load_state_dict(state_dict)
            train_optim.load_state_dict(ckpt['optim'])
        else:
            print(f'Checkpoint {ckpt_path} does not exist, skipping...')
            continue
    
        num_params, embd_params = model.get_num_params()

        print(f'Loaded checkpoint at step {ckpt_steps} with {num_params} parameters, {embd_params} embedding parameters')

        # free up memory
        ckpt = None

        # push model to gpu
        model.to(device)

        # move optimizer states to gpu as well
        for state in train_optim.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    
        # compile the model
        if cfg.compile:
            print("compiling the model... (takes a ~minute)")
            model = torch.compile(model)

        # autocast for mixed precision forward pass
        ctx = nullcontext() if cfg.device_type == 'cpu' else torch.amp.autocast(device_type = cfg.device_type, dtype = cfg.ptdtype)

        # create random eigenvectors for sharpness estimation
        num_params = len(torch.nn.utils.parameters_to_vector(model.parameters()))
        eigvec = None
        pre_eigvec = None

        ## evaluate loss, critical LR and sharpness on each dataset

        for dset in datasets:

            optim = train_optim if dset == 'fineweb' else downstream_optim # use the dummy optimizer if recomputing gradients is not needed

            print(f'Evaluating on ckpt at step: {ckpt_steps} on dataset: {dset}')
            data_dir = os.path.join('data', dset)    
            cfg.max_iters = 1000
            cfg.sharpness_tol = 1e-09
            # estimate sharpness
            batch = get_batch(data_dir, 'train', cfg.context_len, cfg.batch_size, device) # get a batch for sharpness estimation
            
            # evaluate the model on the dataset
            losses = estimate_loss(ctx, model, loss_fn, cfg.eval_steps, data_dir, cfg.context_len, cfg.batch_size, device)
            eval_results.add_entry(ckpt_step = ckpt_steps, lr_step = lr_step, train_loss = losses['train'].item(), val_loss = losses['val'].item(), dataset = dset)

            print(f'Losses on ckpt at step: {ckpt_steps} on dataset: {dset} - train: {losses['train'].item():.4f}, val: {losses['val'].item():.4f}')

            # estimate critical learning rate
            lrs_mean, lrs_std, iters = estimate_critical_lr(ctx, model, loss_fn, optim, cfg.forward_eval_steps, data_dir, cfg.context_len, cfg.batch_size, device)
            forward_results.add_entry(ckpt_step = ckpt_steps, lr_step = lr_step, critical_lr_train_mean = lrs_mean['train'], critical_lr_train_std = lrs_std['train'], critical_lr_val_mean = lrs_mean['val'], critical_lr_val_std = lrs_std['val'], num_iters_train = iters['train'], num_iters_val = iters['val'], dataset = dset)

            # clear gradients
            optim.zero_grad(set_to_none = True)

            print(f'Critical learning rate on ckpt at step: {ckpt_steps} on dataset: {dset} - train: {lrs_mean['train']:.2e}, val: {lrs_mean['val']:.2e}, num_iters_train: {iters['train']}, num_iters_val: {iters['val']}')

            pre_sharpness, pre_sharpness_std, pre_sharpness_converged, sharpness, sharpness_std, sharpness_converged = estimate_sharpness(ctx, model, loss_fn, optim, cfg.sharpness_eval_steps, data_dir, cfg.context_len, cfg.batch_size, device, max_iters = cfg.max_iters, sharpness_tol = cfg.sharpness_tol)
            sharpness_results.add_entry(ckpt_step = ckpt_steps, lr_step = lr_step, pre_sharpness = pre_sharpness, pre_sharpness_std = pre_sharpness_std, pre_sharpnes_converged = pre_sharpness_converged, sharpness = sharpness, sharpness_std = sharpness_std, sharpness_converged = sharpness_converged, dataset = dset)

            # clear gradients
            optim.zero_grad(set_to_none = True)

            print(f'Sharpness on ckpt at step: {ckpt_steps} on dataset: {dset} - pre-sharpness: {pre_sharpness:.4e}, pre_sharpness_std {pre_sharpness_std:0.4f}, pre-sharpness converged: {pre_sharpness_converged}, sharpness: {sharpness:.4e}, sharpness_std: {sharpness_std:0.4f}, sharpness converged: {sharpness_converged}')

            
        # training data results
        eval_results.save_to_csv(cfg.evals_path)
        forward_results.save_to_csv(cfg.forward_path)
        sharpness_results.save_to_csv(cfg.sharpness_path)

    return 

### CONFIG ###

parser = argparse.ArgumentParser(description = 'Experiment parameters')
parser.add_argument('--cluster', type = str, default = 'zaratan')
parser.add_argument('--dtype', type = str, default = 'float32') # 'float32', 'bfloat16', or 'float16'

### dataset 
parser.add_argument('--dataset_name', type = str, default = 'fineweb')
parser.add_argument('--vocab_size', type = int, default = 50304)
# GPU + DDP

### model
parser.add_argument('--model_name', type = str, default = 'gpt')
parser.add_argument('--ckpt_steps', type = int, default = 0)
parser.add_argument('--init_var', type = float, default = 1.0)
parser.add_argument('--num_layers', type = int, default = 12)
parser.add_argument('--num_heads', type = int, default = 12)
parser.add_argument('--head_dim', type = int, default = 64)
parser.add_argument('--embd_dim', type = int, default = 768)
parser.add_argument('--bias', type = str, default = 'False')
parser.add_argument('--context_len', type = int, default = 1024)
parser.add_argument('--compile', type = lambda x: x.lower() == 'true', default = True)
parser.add_argument('--use_flash', type = lambda x: x.lower() == 'true', default = False)

### optimization
# AdamW optimizer
parser.add_argument('--optim_name', type = str, default = 'AdamW')
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--lr_peak', type = float, default = 3e-04)
parser.add_argument('--lr_min_factor', type = lambda x: float('inf') if x.lower() == 'inf' else float(x), default = float('inf'))
parser.add_argument('--beta1', type = float, default = 0.9)
parser.add_argument('--beta2', type = float, default = 0.95)
parser.add_argument('--eps', type = float, default = 1e-08)
parser.add_argument('--weight_decay', type = float, default = 0.1)
# training time; batch_size; and other things
parser.add_argument('--num_steps', type = int, default = 5_000)
parser.add_argument('--warmup_steps', type = int, default = 500)
parser.add_argument('--warmup_exponent', type = float, default = 1.0)
parser.add_argument('--stable_steps', type = int, default = 0)
parser.add_argument('--decay_schedule_name', type = str, default = 'polynomial') # 'cosine' or 'polynomial'
parser.add_argument('--decay_exponent', type = float, default = 1.0)
parser.add_argument('--gradient_accumulation_steps', type = int, default = 128)
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--grad_clip', type = float, default = 0.0)
### evaluation
parser.add_argument('--results_dir', type = str, default = 'eval_results')
parser.add_argument('--log_interval', type = int, default = 1)
parser.add_argument('--eval_interval', type = int, default = 100)
parser.add_argument('--eval_steps', type = int, default = 200)
parser.add_argument('--forward_eval_steps', type = int, default = 3)
parser.add_argument('--sharpness_eval_steps', type = int, default = 1)
parser.add_argument('--verbose', type = bool, default = False)
# sharpness estimation
parser.add_argument('--sharpness_interval', type = int, default = 100)
parser.add_argument('--topk', type = int, default = 1)
parser.add_argument('--max_iters', type = int, default = 500) # tolerance for convergence of the power method
parser.add_argument('--sharpness_tol', type = float, default = 1e-11) # tolerance for convergence of lobpcg method
# critical learning rate estimation
parser.add_argument('--crit_lr_tol_power', type = int, default = 4)
parser.add_argument('--recompute_grads', type = lambda x: x.lower() == 'true', default = True)
# checkpointing
parser.add_argument('--ckpt_dir', type = str, default = '/fsx-maui/dayal/checkpoints/pretrain_gpt')
parser.add_argument('--ckpt_interval', type = int, default = 500)


cfg = parser.parse_args()

cfg.use_bias = True if cfg.bias == 'True' else False
cfg.dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
cfg.embd_dim = cfg.head_dim * cfg.num_heads
cfg.lr_min = cfg.lr_peak / cfg.lr_min_factor

# DDP settings
device = 'cuda'
backend = 'nccl'
cfg.ddp = int(os.environ.get('RANK', -1)) != -1 # check if this is a DDP run

if cfg.ddp:
    init_process_group(backend = backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    cfg.master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert cfg.gradient_accumulation_steps % ddp_world_size == 0
    cfg.gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    cfg.master_process = True
    seed_offset = 0
    ddp_world_size = 1

torch.manual_seed(1337 + seed_offset)

cfg.device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
cfg.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.dtype]

### Data loader ###
cfg.data_dir = os.path.join('data', cfg.dataset_name)
print(f"data_dir: {cfg.data_dir}")

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(cfg.data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Save results
if cfg.master_process:
    os.makedirs(cfg.results_dir, exist_ok=True)

# Create a descriptive name based on the model and training parameters
base_filename = get_base_filename(cfg, cfg.num_steps)

sharpness_filename = f'sharpness_{base_filename}.csv'
cfg.sharpness_path = os.path.join(cfg.results_dir, sharpness_filename)

evals_filename = f'eval_{base_filename}.csv'
cfg.evals_path = os.path.join(cfg.results_dir, evals_filename)

forward_filename = f'forward_{base_filename}.csv'
cfg.forward_path = os.path.join(cfg.results_dir, forward_filename)

## Train, evaluate and save checkpoints ###
evaluate_ckpts_on_datasets(cfg, device)

