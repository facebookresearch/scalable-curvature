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
import utils.sharpness_cupy_utils as sharpness_utils 
from utils.critical_learning_rate import compute_critical_learning_rate
from utils.loss_functions import CrossEntropyLoss
from utils.data_storage import DataStorage
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
    ckpt_filename = get_ckpt_filename(cfg, num_steps)
    base_filename = f'{ckpt_filename}_{add_filename}'
    return base_filename

def save_checkpoint(model, optim, step, config, ckpt_dir = '/fsx-maui/dayal/checkpoints/gpt'):

    """
    model: has to be the raw model; not the ddp wrapped model
    """

    ckpt_name = get_base_filename(config, step)

    checkpoint = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'step': step,
        'config': config,
    }
    print(f"saving checkpoint to {ckpt_dir}")
    ckpt_name = f'{get_ckpt_filename(config, step)}.ckpt'
    torch.save(checkpoint, os.path.join(ckpt_dir, ckpt_name))

    return

def get_last_checkpoint(cfg, device):
    save_steps = [cfg.ckpt_interval*i  for i in range(0, cfg.num_steps//cfg.ckpt_interval + 1)]

    for step in save_steps[::-1]:
        ckpt_name = f'{get_ckpt_filename(cfg, step)}.ckpt'
        ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            print(f'Checkpoint found at: {ckpt_path}')
            return torch.load(ckpt_path, map_location = device, weights_only = False)
    print(f'No checkpoint found at {ckpt_path}')
    return None

def create_train_state(cfg, device):

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

def train_and_evaluate(cfg, device):

    ### CREATE TRAIN STATE AND LOAD CHECKPOINT ###
    load_checkpoint = cfg.load_checkpoint
    init_steps = 0
    last_ckpt = None

    ### TRAIN STATE ###
    # create the model, loss function and optimizer
    model, loss_fn, optim = create_train_state(cfg, device)

    num_params, embd_params = model.get_num_params()
    print("number of parameters: %.2fM" % (num_params/1e6,))
    print("number of embd params: %.2fM" % (embd_params/1e6,))

    ### LOAD CHECKPOINT ###
    # Load the last checkpoint if it exists
    if load_checkpoint:
        last_ckpt = get_last_checkpoint(cfg, device)

        if last_ckpt is not None:
            # find out the training step from the checkpoint
            init_steps = last_ckpt['step']

    ### DATA CONTAINERS ###
    train_results = DataStorage(columns = ['step', 'lr_step', 'loss_step'], num_params = num_params, embd_params = embd_params)
    forward_results = DataStorage(columns = ['step', 'lr_step', 'lr_lower', 'lr_upper', 'num_iters_critical'], num_params = num_params, embd_params = embd_params)
    sharpness_results = DataStorage(columns = ['step', 'lr_step', 'pre_sharpness', 'pre_num_iters', 'sharpness', 'num_iters'], num_params = num_params, embd_params = embd_params)
    eval_results = DataStorage(columns = ['step', 'lr_step', 'train_loss', 'val_loss'], num_params = num_params, embd_params = embd_params)
    
    # Load existing data if it exists 
    if last_ckpt is not None:
        train_data_exists = train_results.load_from_csv(cfg.train_path, num_steps = init_steps) # load previous results if any
        forward_data_exists = forward_results.load_from_csv(cfg.forward_path, num_steps = init_steps) # load previous results if any
        sharpness_data_exists = sharpness_results.load_from_csv(cfg.sharpness_path, num_steps = init_steps) # load previous results if any
        eval_data_exists = eval_results.load_from_csv(cfg.evals_path, num_steps = init_steps) # load previous results if any
        # we will only load the checkpoint if all data exists
        load_checkpoint = train_data_exists and forward_data_exists and sharpness_data_exists and eval_data_exists        

    # load the last checkpoint if (1) it exists and (2) the data exists
    if load_checkpoint and last_ckpt is not None:
        print(f"Loading from checkpoint at step {init_steps}...")
        # load model and optimizer state dicts
        state_dict = last_ckpt['model']
        init_steps = last_ckpt['step']
        # remove the unwanted prefix
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        # load model and optimizer state dicts
        model.load_state_dict(state_dict)
        optim.load_state_dict(last_ckpt['optim'])
    else:
        print("Either no checkpoint found or data does not exist, starting from scratch.")
        init_steps = 0
    
    # free up memory
    last_ckpt = None

    # push model to gpu
    model.to(device)

    # move optimizer states to the same device
    for state in optim.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    
    # compile the model
    if cfg.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # wrap model into DDP container
    if cfg.ddp:
        model = DDP(model, device_ids = [ddp_local_rank])
    
    raw_model = model.module if cfg.ddp else model # unwrap DDP container for saving checkpoints
    
    ### TRAINING LOOP SETUP ###
    lr_guess = cfg.lr_peak
    lr_step = warmup_stable_decay(step = max(init_steps-1, 0), init_value = cfg.lr_init, peak_value = cfg.lr_peak, min_value = cfg.lr_min, num_steps = cfg.num_steps, warmup_steps = cfg.warmup_steps, stable_steps = cfg.stable_steps, warmup_exponent = cfg.warmup_exponent, decay_schedule_name = cfg.decay_schedule_name, decay_exponent = cfg.decay_exponent)
     # autocast for mexied precision forward pass
    ctx = nullcontext() if cfg.device_type == 'cpu' else torch.amp.autocast(device_type = cfg.device_type, dtype = cfg.ptdtype)
    ### Data loader ###
    data_dir = os.path.join('data', cfg.dataset_name)
    print(f"data_dir: {data_dir}")

    X, Y = get_batch(data_dir, 'train', cfg.context_len, cfg.batch_size, device)
    # initial eigenvec
    eigvec = None
    pre_eigvec = None
    crit_thresh = (2 + 2 * cfg.beta1) / (1 - cfg.beta1)

    print(f'Recompute Grads: {cfg.recompute_grads}')

    for step in range(init_steps, cfg.num_steps+1):

        ### SHARPNESS AND CRITICAL LR ESTIMATION ###
        
        # critical learning rate estimation; using the batch sampled
        (lr_lower, lr_upper), num_iters_critical = compute_critical_learning_rate(ctx = ctx, model = model, loss_fn = loss_fn, optim = optim, batch = (X, Y), lr_guess = lr_guess, tol_power = cfg.crit_lr_tol_power, recompute_grads = cfg.recompute_grads)
        lr_guess = lr_lower

        # For learning rate range results
        forward_results.add_entry(step = step, lr_step = lr_step, lr_lower = lr_lower, lr_upper = lr_upper, num_iters_critical = num_iters_critical)
        print(f'LR Range: {lr_lower:0.1e}, {lr_upper:0.1e} computed in {num_iters_critical} steps')
            
        # flush the gradients before sharpness computation to save memory; only saves memory for sharpness and not pre conditioned sharpness
        optim.zero_grad(set_to_none = True)

        if step % cfg.sharpness_interval == 0 and cfg.master_process:
            # eigenvalue estimation using lobpcg method
            pre_sharpness_step, pre_eigvec, num_iters_pre_sharpness = sharpness_utils.get_pre_sharpness_lobpcg(model, loss_fn, optim, (X, Y), eigvecs = pre_eigvec, tol = cfg.sharpness_tol, max_iters = cfg.max_iters)
            print(f'Pre-sharpness: {pre_sharpness_step:0.4f} computed in {num_iters_pre_sharpness} steps, critical LR: {crit_thresh/pre_sharpness_step:0.1e}')
            # eigenvalue estimation using lobpcg method
            sharpness_step, eigvec, num_iters_sharpness = sharpness_utils.get_sharpness_lobpcg(model, loss_fn, (X, Y), eigvecs = eigvec, tol = cfg.sharpness_tol, max_iters = cfg.max_iters)
            print(f'Sharpness: {sharpness_step:0.4f} computed in {num_iters_sharpness} steps')
            
            # For sharpness results
            sharpness_results.add_entry(step = step, lr_step = lr_step, pre_sharpness = pre_sharpness_step, pre_num_iters = num_iters_pre_sharpness, sharpness = sharpness_step, num_iters = num_iters_sharpness)

            # flush the gradients; I dont think I need it here but just to be safe
            optim.zero_grad(set_to_none = True)
        
        ### CHECKPOINTING ###
        if step % cfg.ckpt_interval == 0 and cfg.master_process:
            save_checkpoint(model = raw_model, optim = optim, step = step, config = cfg, ckpt_dir = cfg.ckpt_dir)
            
        ### EVALUATE ##
        if step % cfg.eval_interval == 0 and cfg.master_process:
            losses = estimate_loss(ctx, model, loss_fn, cfg.eval_steps, data_dir, cfg.context_len, cfg.batch_size, device)
            eval_results.add_entry(step = step, lr_step = lr_step, train_loss = losses['train'].item(), val_loss = losses['val'].item())

            # save the results at every interval
            train_results.save_to_csv(cfg.train_path)
            forward_results.save_to_csv(cfg.forward_path)
            eval_results.save_to_csv(cfg.evals_path)
            sharpness_results.save_to_csv(cfg.sharpness_path)

        ### TRAIN STEP ###
        # set gradients to None
        optim.zero_grad(set_to_none = True)

        cosine_step = step - cfg.warmup_steps
        # get the learning rate
        lr_step = warmup_stable_decay(step = step, init_value = cfg.lr_init, peak_value = cfg.lr_peak, min_value = cfg.lr_min, num_steps = cfg.num_steps, warmup_steps = cfg.warmup_steps, stable_steps = cfg.stable_steps, warmup_exponent = cfg.warmup_exponent, decay_schedule_name = cfg.decay_schedule_name, decay_exponent = cfg.decay_exponent)
        
        # update the learning rate
        for param_group in optim.param_groups:
            param_group['lr'] = lr_step * param_group.get('lr_scale', 1.0)

        for micro_step in range(cfg.gradient_accumulation_steps):
            if cfg.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                model.require_backward_grad_sync = (micro_step == cfg.gradient_accumulation_steps - 1)
            with ctx:
                logits = model(X)
                loss = loss_fn(logits, Y)
                loss = loss / cfg.gradient_accumulation_steps # scale the loss to account for gradient accumulation
            
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch(data_dir, 'train', cfg.context_len, cfg.batch_size, device)
            # backward pass
            loss.backward()
        
        # clip the gradient
        if cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        # dont flush the gradients yet, we may use it for critical learning rate estimation

        ### Loss storage ##
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * cfg.gradient_accumulation_steps
        print(f"step: {step}, lr: {lr_step:0.1e}, loss: {lossf:.4f}")
        train_results.add_entry(step=step, lr_step=lr_step, loss_step=lossf)

        ### Optimizer step ###
        optim.step()
        # dont free up gradients; they will be utilized for critical LR estimation
        
    # training data results
    train_results.save_to_csv(cfg.train_path)
    forward_results.save_to_csv(cfg.forward_path)
    eval_results.save_to_csv(cfg.evals_path)
    sharpness_results.save_to_csv(cfg.sharpness_path)

    # destroy the DDP process group
    if cfg.ddp:
        destroy_process_group()
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
parser.add_argument('--lr_peak', type = float, default = 1e-04)
parser.add_argument('--lr_min_factor', type = lambda x: float('inf') if x.lower() == 'inf' else float(x), default = float('inf'))
parser.add_argument('--beta1', type = float, default = 0.9)
parser.add_argument('--beta2', type = float, default = 0.95)
parser.add_argument('--eps', type = float, default = 1e-08)
parser.add_argument('--weight_decay', type = float, default = 0.0)
# training time; batch_size; and other things
parser.add_argument('--num_steps', type = int, default = 10_000)
parser.add_argument('--warmup_steps', type = int, default = 1_000)
parser.add_argument('--warmup_exponent', type = float, default = 1.0)
parser.add_argument('--stable_steps', type = int, default = 0)
parser.add_argument('--decay_schedule_name', type = str, default = 'cosine') # 'cosine' or 'polynomial'
parser.add_argument('--decay_exponent', type = float, default = 1.0)
parser.add_argument('--gradient_accumulation_steps', type = int, default = 40)
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--grad_clip', type = float, default = 0.0)
### evaluation
parser.add_argument('--results_dir', type = str, default = 'gpt_results')
parser.add_argument('--log_interval', type = int, default = 1)
parser.add_argument('--eval_interval', type = int, default = 50)
parser.add_argument('--eval_steps', type = int, default = 200)
parser.add_argument('--verbose', type = bool, default = False)
# sharpness estimation
parser.add_argument('--sharpness_interval', type = int, default = 10)
parser.add_argument('--topk', type = int, default = 1)
parser.add_argument('--max_iters', type = int, default = 200) # tolerance for convergence of the power method
parser.add_argument('--sharpness_tol', type = float, default = 1e-11) # tolerance for convergence of the power method
# critical learning rate estimation
parser.add_argument('--crit_lr_tol_power', type = int, default = 4)
parser.add_argument('--recompute_grads', type = lambda x: x.lower() == 'true', default = True)
# checkpointing
parser.add_argument('--load_checkpoint', type = lambda x: x.lower() == 'true', default = True)
parser.add_argument('--ckpt_dir', type = str, default = '/fsx-maui/dayal/checkpoints/pretrain_gpt')
parser.add_argument('--ckpt_interval', type = int, default = 500) # save a checkpoint every 500 steps


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

if cfg.master_process:
    os.makedirs(cfg.results_dir, exist_ok=True)

# Create a descriptive name based on the model and training parameters
base_filename = get_base_filename(cfg, cfg.num_steps)

train_filename = f'train_{base_filename}.csv'
cfg.train_path = os.path.join(cfg.results_dir, train_filename)

evals_filename = f'eval_{base_filename}.csv'
cfg.evals_path = os.path.join(cfg.results_dir, evals_filename)

forward_filename = f'forward_{base_filename}.csv'
cfg.forward_path = os.path.join(cfg.results_dir, forward_filename)

sharpness_filename = f'sharpness_{base_filename}.csv'
cfg.sharpness_path = os.path.join(cfg.results_dir, sharpness_filename)

## Train, evaluate and save checkpoints ###
train_and_evaluate(cfg, device)

