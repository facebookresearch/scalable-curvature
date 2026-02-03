"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python demo_nanogpt.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 demo_nanogpt.py
"""

import os
import pickle
from contextlib import nullcontext
import argparse
import time
import importlib

import numpy as np
import pandas as pd

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from utils.schedules_utils import warmup_stable_decay
from utils.loss_functions import CrossEntropyLoss
from utils.data_storage import DataStorage
from utils.data_loader import get_batch
from utils.optimizers import AdamW
from utils.measure_critical_lr import measure_critical_lr

def print_gpu_memory(stage_name):
    """Simple GPU memory usage print"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"{stage_name}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

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

def get_base_filename(cfg, num_steps):
    base_filename = (
        f'{cfg.dataset_name}_'
        f'v{cfg.vocab_size}_'
        f'I{cfg.seed}_'
        f'{cfg.model_name}_'
        f'{cfg.abc}_'
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
        f'lr{cfg.lr_peak:.1e}_'
        f'lr{cfg.lr_min_factor}_'
        f'wd{cfg.weight_decay}_'
        f'bs{cfg.batch_size}_'
        f'b{cfg.beta1}_'
        f'b{cfg.beta2}_'
        f'eps{cfg.eps}_'
        f'gc{cfg.grad_clip}'
    )
    return base_filename


def save_checkpoint(model, optim, step, config, ckpt_dir = 'checkpoints/gpt_mup', save_last_only = False):
    """
    NOTE: model has to be the raw model; not the ddp wrapped model
    """

    ckpt_name = get_base_filename(config, step)

    checkpoint = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'step': step,
        'config': config,
    }
    # create the checkpoint directory if it does not exist
    if config.master_process:
        os.makedirs(config.ckpt_dir, exist_ok = True)

        # Delete previous checkpoint if it exists
        if save_last_only:
            if step >= config.ckpt_interval:  # Only try to delete if this isn't the first checkpoint
                prev_step = step - config.ckpt_interval
                prev_ckpt_name = f'{get_base_filename(config, prev_step)}.ckpt'
                prev_ckpt_path = os.path.join(config.ckpt_dir, prev_ckpt_name)
                if os.path.exists(prev_ckpt_path):
                    os.remove(prev_ckpt_path)
            
        # save the checkpoint
        ckpt_name = f'{get_base_filename(config, step)}.ckpt'
        ckpt_filename = os.path.join(config.ckpt_dir, ckpt_name)
        print(f"saving checkpoint to {ckpt_filename}")
        torch.save(checkpoint, ckpt_filename)
    return

def get_last_checkpoint(cfg, device):
    # checkpoints are saved every 10% of the training steps
    save_steps = [cfg.ckpt_interval*i  for i in range(1, cfg.num_steps//cfg.ckpt_interval + 1)]
    
    for step in save_steps[::-1]:
        ckpt_name = f'{get_base_filename(cfg, step)}.ckpt'
        ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            if cfg.master_process:
                print(f'Checkpoint found at: {ckpt_path}')
            return torch.load(ckpt_path, map_location = device, weights_only = False)
        
    if cfg.master_process:
        print(f'No checkpoint found in {cfg.ckpt_dir}')
    return None

def create_train_state(cfg, device):

    """ Create a train state with the model, optimizer, and loss function. 
    NOTE: It does not compile the model or pushes it to the device. """

    # Dynamic import based on abc parameter
    model_module_name = f'utils.gpt_{cfg.abc}'
    try:
        model_module = importlib.import_module(model_module_name)
        GPTConfig = model_module.GPTConfig
        GPT = model_module.GPT
    except ImportError as e:
        raise ImportError(f"Could not import {model_module_name}. Make sure utils/gpt_{cfg.abc}.py exists. Error: {e}")

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
    
    if cfg.verbose and cfg.master_process:
        for name, param in model.named_parameters():
            mean = param.mean().item()
            var = param.var().item()
            if 'scale' in name or 'bias' in name:
                print(f'Parameter: {name}, mean: {mean:.4f}, var: {var:.4f}')
            else:
                print(f'Parameter: {name}, mean: {mean:.4f}, var: {var:.4f}, 1/var: {1.0/var:0.4f}')
    
    # Custom Optimizer
    optim = AdamW(
        model.parameters(),
        lr = cfg.lr_init, 
        betas = (cfg.beta1, cfg.beta2), 
        weight_decay = cfg.weight_decay, 
        eps = cfg.eps,
    ) 

    # loss function
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
    if cfg.master_process:
        print("number of parameters: %.2fM" % (num_params/1e6,))
        print("number of embd params: %.2fM" % (embd_params/1e6,))

    ### LOAD CHECKPOINT ###
    # Load the last checkpoint if it exists
    if load_checkpoint:
        last_ckpt = get_last_checkpoint(cfg, 'cpu')

        if last_ckpt is not None:
            # find out the training step from the checkpoint
            init_steps = last_ckpt['step']

    ### DATA CONTAINERS ###
    if cfg.master_process:
        train_results = DataStorage(columns=['step', 'lr_step', 'loss_step', 'critical_lr', 'critical_sharpness', 'num_iters'])
        eval_results = DataStorage(columns=['step', 'lr_step', 'train_loss', 'val_loss'])

    # Load existing data if it exists 
    if last_ckpt is not None and cfg.master_process:
        train_data_exists = train_results.load_from_csv(cfg.train_path, num_steps=init_steps)
        eval_data_exists = eval_results.load_from_csv(cfg.evals_path, num_steps=init_steps)
        # we will only load the checkpoint if all data exists
        load_checkpoint = train_data_exists and eval_data_exists

    # load the last checkpoint if (1) it exists and (2) the data exists
    if load_checkpoint and last_ckpt is not None:
        print(f"Loading from checkpoint at step {init_steps}...")
        # load model and optimizer state dicts
        init_steps = last_ckpt['step']
        # remove the unwanted prefix
        unwanted_prefix = '_orig_mod.'
        for k,v in list(last_ckpt['model'].items()):
            if k.startswith(unwanted_prefix):
                last_ckpt['model'][k[len(unwanted_prefix):]] = last_ckpt['model'].pop(k)
        # load model and optimizer state dicts
        model.load_state_dict(last_ckpt['model'])
        del last_ckpt['model']
        optim.load_state_dict(last_ckpt['optim'])
    else:
        if cfg.master_process:
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
        if cfg.master_process:
            print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    # wrap model into DDP container
    if cfg.ddp:
        model = DDP(model, device_ids = [ddp_local_rank])
    
    raw_model = model.module if cfg.ddp else model # unwrap DDP container for saving checkpoints
    
    ### TRAINING LOOP SETUP ###
    # lr_step is the learning rate at the current step
    torch.cuda.empty_cache() # clear the cache before starting the training loop
    lr_step = warmup_stable_decay(step = max(init_steps-1, 0), init_value = cfg.lr_init, peak_value = cfg.lr_peak, min_value = cfg.lr_min, num_steps = cfg.num_steps, warmup_steps = cfg.warmup_steps, stable_steps = cfg.stable_steps, warmup_exponent = cfg.warmup_exponent, decay_schedule_name = cfg.decay_schedule_name, decay_exponent = cfg.decay_exponent)

    # autocast for mexied precision forward pass
    ctx = nullcontext() if cfg.device_type == 'cpu' else torch.amp.autocast(device_type = cfg.device_type, dtype = cfg.ptdtype)
    ### Data loader ###
    data_dir = os.path.join(cfg.data_dir, cfg.dataset_name)
    if cfg.master_process:
        print(f"data_dir: {data_dir}")

    X, Y = get_batch(data_dir, 'train', cfg.context_len, cfg.batch_size, device)
    
    current_lr_guess = 1e-3

    for step in range(init_steps, cfg.num_steps + 1):

        ### CHECKPOINTING ###
        if step % cfg.ckpt_interval == 0 and cfg.save_checkpoint and cfg.master_process:
            save_checkpoint(model = raw_model, optim = optim, step = step, config = cfg, ckpt_dir = cfg.ckpt_dir, save_last_only = cfg.save_last_only)
            
        ### EVALUATE ##
        if step % cfg.eval_interval == 0:
            losses = estimate_loss(ctx, model, loss_fn, cfg.eval_steps, data_dir, cfg.context_len, cfg.batch_size, device)
            if cfg.master_process:
                print(f"step: {step}, lr: {lr_step:0.1e}, train loss: {losses['train'].item():.4f}, val loss: {losses['val'].item():.4f}")
                eval_results.add_entry({
                    'step': step, 
                    'lr_step': lr_step, 
                    'train_loss': losses['train'].item(), 
                    'val_loss': losses['val'].item(),
                    'num_params': num_params,
                    'embd_params': embd_params
                })

                # save the results at every interval
                train_results.save_to_csv(cfg.train_path)
                eval_results.save_to_csv(cfg.evals_path)

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
        
        ### Loss storage ##
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * cfg.gradient_accumulation_steps

        # --- Measure Critical Curvature ---
        def loss_closure(batch):
            bx, by = batch
            with ctx:
                l_logits = model(bx)
                l_loss = loss_fn(l_logits, by)
            
            if cfg.ddp:
                dist.all_reduce(l_loss, op=dist.ReduceOp.AVG)
                
            return l_loss.item()
        
        lr_lower, lr_upper, num_iters = measure_critical_lr(
            model=model,
            optimizer=optim,
            batch=(X, Y), 
            loss_closure=loss_closure,
            lr_guess=current_lr_guess,
            tol_power=cfg.tol_power
        )
        
        critical_lr = (lr_lower + lr_upper) / 2.0
        critical_sharpness = 2.0 / critical_lr
        
        current_lr_guess = critical_lr

        if cfg.master_process:
            print(f"[Step {step}] LR: {lr_step:.2e} | Loss: {lossf:.4f} | Critical LR: {critical_lr:.2e} | Sharpness: {critical_sharpness:.4f} | Iters: {num_iters}")
            
            train_results.add_entry({
                'step': step, 
                'lr_step': lr_step, 
                'loss_step': lossf, 
                'critical_lr': critical_lr,
                'critical_sharpness': critical_sharpness,
                'num_iters': num_iters,
                'num_params': num_params,
                'embd_params': embd_params
            })
        
        ### Optimizer step ###
        optim.step()
        # flush the gradients as soon as we can, no need for this memory anymore
        optim.zero_grad(set_to_none = True)

    # training data results
    if cfg.master_process:
        print("Training completed.")
        train_results.save_to_csv(cfg.train_path)
        eval_results.save_to_csv(cfg.evals_path)

    # destroy the DDP process group
    if cfg.ddp:
        destroy_process_group()
    return 


### CONFIG ###

parser = argparse.ArgumentParser(description = 'Experiment parameters')
parser.add_argument('--cluster', type = str, default = 'zaratan')

### dataset 
parser.add_argument('--dataset_name', type = str, default = 'fineweb')
parser.add_argument('--data_dir', type = str, default = 'data/')
parser.add_argument('--vocab_size', type = int, default = 50304)

# GPU + DDP
# add seed variable
parser.add_argument('--seed', type = int, default = 1337)
### model
parser.add_argument('--model_name', type = str, default = 'gpt')
parser.add_argument('--abc', type = str, default = 'sp')
parser.add_argument('--init_var', type = float, default = 1.0)
parser.add_argument('--num_layers', type = int, default = 12)
parser.add_argument('--num_heads', type = int, default = 12)
parser.add_argument('--head_dim', type = int, default = 64)
parser.add_argument('--embd_dim', type = int, default = 768)
parser.add_argument('--bias', type = str, default = 'False')
parser.add_argument('--context_len', type = int, default = 1024)
parser.add_argument('--compile', type = lambda x: x.lower() == 'true', default = True)
parser.add_argument('--use_flash', type = lambda x: x.lower() == 'true', default = True)

### optimization
# adamw optimizer
parser.add_argument('--optim_name', type = str, default = 'AdamW')
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--lr_peak', type = float, default = 1e-04)
parser.add_argument('--lr_min_factor', type = lambda x: float('inf') if x.lower() == 'inf' else float(x), default = float('inf'))
parser.add_argument('--beta1', type = float, default = 0.9)
parser.add_argument('--beta2', type = float, default = 0.95)
parser.add_argument('--eps', type = float, default = 1e-08)
parser.add_argument('--weight_decay', type = float, default = 0.1)
# training time; batch_size; and other things
parser.add_argument('--num_steps', type = int, default = 10_000)
parser.add_argument('--warmup_steps', type = int, default = 2000)
parser.add_argument('--warmup_exponent', type = float, default = 1.0)
parser.add_argument('--stable_steps', type = int, default = 6000)
parser.add_argument('--decay_schedule_name', type = str, default = 'polynomial') # 'cosine' or 'polynomial'
parser.add_argument('--decay_exponent', type = float, default = 1.0)
parser.add_argument('--gradient_accumulation_steps', type = int, default = 64)
parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--grad_clip', type = float, default = 1.0)
parser.add_argument('--tol_power', type = int, default = 4)
### evaluation
parser.add_argument('--log_interval', type = int, default = 1)
parser.add_argument('--eval_interval', type = int, default = 200)
parser.add_argument('--eval_steps', type = int, default = 200)
parser.add_argument('--verbose', type = lambda x: x.lower() == 'true', default = False)
# checkpointing
parser.add_argument('--load_checkpoint', type = lambda x: x.lower() == 'true', default = True)
parser.add_argument('--save_checkpoint', type = lambda x: x.lower() == 'true', default = True)
parser.add_argument('--save_last_only', type = lambda x: x.lower() == 'true', default = True)
parser.add_argument('--ckpt_interval', type = int, default = 2000)
parser.add_argument('--ckpt_dir', type = str, default = 'checkpoints/temp')

cfg = parser.parse_args()

cfg.results_dir = f'results/{cfg.model_name}_{cfg.abc}'
# cfg.ckpt_dir = f'checkpoints/{cfg.model_name}_{cfg.abc}'

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

torch.manual_seed(cfg.seed + seed_offset)

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

## Train, evaluate and save checkpoints ###
train_and_evaluate(cfg, device)