import os
import wandb
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader

import utils.models as model_utils
import utils.image_data as data_utils
from utils.schedules_utils import warmup_stable_decay
import utils.sharpness_cupy_utils as sharpness_cupy_utils 
import utils.sharpness_dir_utils as sharpness_dir_utils
import utils.loss_functions as loss_functions
from utils.optimizers import SGD
from utils.critical_lr_cache_utils import compute_critical_learning_rate
torch.set_float32_matmul_precision('high')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set environment variables (optional, but recommended for clarity)
os.environ["WANDB_ENTITY"] = "dayal"
os.environ["WANDB_PROJECT"] = "critical_lr_image"
os.environ["WANDB_MODE"] = "online"  # or "offline" for local logging

def iterate_dataset(dataset: Dataset, batch_size: int):
    """ Generic Dataloader """
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    for (x, y) in loader:
        yield x.to(device), y.to(device)

def compute_loss_and_accuracy(model, loss_fn, dataset, batch_size):

    total_loss = 0
    total_acc = 0
    num_items = 0

    with torch.no_grad():
        for (x, y) in iterate_dataset(dataset, batch_size):

            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item()

            if isinstance(loss_fn, nn.MSELoss) or isinstance(loss_fn, loss_functions.MSELoss):
                # MSE loss has one-hot labels
                acc = (logits.argmax(1) == y.argmax(1)).float().mean() 
            else:
                acc = (logits.argmax(1) == y).float().mean()
            total_acc += acc.item()

            num_items += 1
    
    total_loss = total_loss / num_items
    total_acc = total_acc / num_items

    return total_loss, 100*total_acc
    
def train_and_evaluate(cfg, model, loss_fn, optim, x_train, y_train, x_test, y_test, x_subset, y_subset):

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_subset = TensorDataset(torch.from_numpy(x_subset).float(), torch.from_numpy(y_subset))
    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))

    train_results = list()
    forward_results = list()
    eval_results = list()

    cfg.num_steps_per_epoch = (len(train_dataset) // cfg.batch_size)
    cfg.num_epochs = cfg.num_steps // cfg.num_steps_per_epoch

    lr_guess = cfg.lr_peak
    eigvec = None

    # Initialize wandb
    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        entity=os.environ["WANDB_ENTITY"],
        config=vars(cfg),  # logs all argparse/config values
        name=f"run_{cfg.random_seed}_{cfg.model_name}_{cfg.width}x{cfg.depth}"
    )

    print(f'Training for {cfg.num_epochs} epochs')

    for epoch in range(cfg.num_epochs):

        for batch_idx, batch in enumerate(iterate_dataset(train_dataset, cfg.batch_size)):

            step = epoch * cfg.num_steps_per_epoch + batch_idx
            
            if step % cfg.eval_interval == 0:

                # Evaluation
                train_loss, train_acc = compute_loss_and_accuracy(model, loss_fn, train_subset, cfg.batch_size)
                test_loss, test_acc = compute_loss_and_accuracy(model, loss_fn, test_dataset, cfg.batch_size)

                result = np.array([step, train_loss, train_acc, test_loss, test_acc])
                eval_results.append(result)

                ### Critical LR and sharpness computation ###

                (lr_lower, lr_upper), num_iters_critical = compute_critical_learning_rate(model = model, loss_fn = loss_fn, optim = optim, batch = batch, lr_guess = lr_guess, tol_power = cfg.crit_lr_tol_power, recompute_grads = cfg.recompute_grads)
                lr_guess = lr_lower

                print(f'LR Range: {lr_lower:0.4f}, {lr_upper:0.4f} computed in {num_iters_critical} steps')
                
                # flush the gradients before sharpness computation to save memory
                optim.zero_grad(set_to_none = True)

                # eigenvalue estimation using power method
                sharpness_step, eigvec, num_iters_power = sharpness_cupy_utils.get_sharpness_lobpcg(model = model, loss_fn = loss_fn, batch = batch, max_iters = cfg.max_iters, tol = cfg.sharpness_tol, eigvecs = eigvec)
                print(f'Sharpness: {sharpness_step:0.4f} computed in {num_iters_power} steps, critical LR: {2/sharpness_step:0.4f}')

                # compute directional sharpness
                dir_sharpness_step = sharpness_dir_utils.get_dir_sharpness(model = model, loss_fn = loss_fn, optim = optim, batch = batch)
                print(f'Directional Sharpness: {dir_sharpness_step:0.4f}, critical LR: {2/dir_sharpness_step:0.4f}')

                result = np.asarray([step, sharpness_step, num_iters_power, dir_sharpness_step, lr_lower, lr_upper, num_iters_critical])
                forward_results.append(result)

                # After each evaluation interval
                wandb.log({
                    "step": step,
                    "eval/train_loss": train_loss,
                    "eval/train_acc": train_acc,
                    "eval/test_loss": test_loss,
                    "eval/test_acc": test_acc,
                    "eval/sharpness": sharpness_step,
                    "eval/dir_sharpness": dir_sharpness_step,
                    "eval/lr_lower": lr_lower,
                    "eval/lr_upper": lr_upper
                })
                
                # flush the gradients; I dont think I need it here but just to be safe
                optim.zero_grad(set_to_none = True)
            
                
            ### Training ###

            # Learning rate schedule
            cosine_step = step - cfg.warmup_steps
            # get the learning rate
            lr_step = warmup_stable_decay(
                step = step,
                init_value = cfg.lr_init,
                peak_value = cfg.lr_peak,
                min_value = cfg.lr_min,
                num_steps = cfg.num_steps,
                warmup_steps = cfg.warmup_steps,
                stable_steps = cfg.stable_steps,
                warmup_exponent = cfg.warmup_exponent,
                decay_schedule_name = cfg.decay_schedule_name,
                decay_exponent = cfg.decay_exponent
            )

            for param_group in optim.param_groups:
                param_group['lr'] = lr_step

            # optimizer step
            optim.zero_grad()
            # compute loss
            x, y = batch
            loss = loss_fn(model(x), y)
            loss_step = loss.item()
            loss.backward()
            optim.step()

            print(f'step: {step}, lr: {lr_step:0.4f}, loss: {loss_step:0.4f}')

            result = np.asarray([step, lr_step, loss_step])
            train_results.append(result)

            if np.isnan(loss_step) or np.isinf(loss_step):
                print(f'Loss is NaN or Inf at step {step}. Stopping training.')
                return np.asarray(train_results), np.asarray(eval_results), np.asarray(forward_results)
            
            if step % 100 == 0:
                df_train = pd.DataFrame(train_results, columns = ['step', 'lr_step', 'loss_step'])
                df_train.to_csv(train_path)

                df_eval = pd.DataFrame(eval_results, columns = ['step', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
                df_eval.to_csv(eval_path)

                df_forward = pd.DataFrame(forward_results, columns = ['step', 'sharpness', 'num_iters_power', 'dir_sharpness', 'lr_lower', 'lr_upper', 'num_iters_critical'])
                df_forward.to_csv(forward_path)
    
    wandb.finish()

    return train_results, eval_results, forward_results

LOSS_FUNCTIONS = {'mse': loss_functions.MSELoss(), 'xent': nn.CrossEntropyLoss(reduction = 'mean')}

parser = argparse.ArgumentParser(description = 'Image classification using FCNs')
parser.add_argument('--random_seed', type = int, default = 42)
# dataset
parser.add_argument('--dataset_name', type = str, default = 'cifar-10', choices = data_utils.DATASETS.keys())
parser.add_argument('--data_dir', type = str, default = './datasets/')
parser.add_argument('--in_dim', type = int, default = 32*32*3)
parser.add_argument('--num_classes', type = int, default = 10)

# model hparams
parser.add_argument('--model_name', type = str, default = 'fcn')
parser.add_argument('--width', type = int, default = 512)
parser.add_argument('--depth', type = int, default = 4)
parser.add_argument('--act_name', type = str, default = 'gelu')
parser.add_argument('--varw', type = float, default = 1.0)
# optimization hparams
parser.add_argument('--loss_name', type = str, default = 'mse')
parser.add_argument('--batch_size', type = int, default = 50_000)
parser.add_argument('--num_steps', type = int, default = 10_000)
# optimizer hparams
parser.add_argument('--opt_name', type = str, default = 'sgd')
parser.add_argument('--lr_init', type = float, default = 0.0)
parser.add_argument('--lr_peak', type = float, default = 1.0)
parser.add_argument('--lr_min_factor', type = lambda x: float('inf') if x.lower() == 'inf' else float(x), default = float('inf'))
parser.add_argument('--warmup_steps', type = int, default = 100)
parser.add_argument('--warmup_exponent', type = float, default = 1.0)
parser.add_argument('--stable_steps', type = int, default = 9900)
parser.add_argument('--decay_schedule_name', type = str, default = 'cosine') # 'cosine' or 'polynomial'
parser.add_argument('--decay_exponent', type = float, default = 1.0)
parser.add_argument('--momentum', type = float, default = 0.0)
parser.add_argument('--weight_decay', type = float, default = 0.0)
# sharpness
parser.add_argument('--topk', type = int, default = 1)
parser.add_argument('--lr_guess', type = float, default = 1e-06)
parser.add_argument('--crit_lr_tol_power', type = int, default = 4)
parser.add_argument('--recompute_grads', type = lambda x: x.lower() == 'true', default = True)
parser.add_argument('--max_iters', type = int, default = 200)
parser.add_argument('--sharpness_tol', type = float, default = 1e-11)
# misc
parser.add_argument('--results_dir', type = str, default = 'sgd_dir_sharp_results')
parser.add_argument('--save_model', type = bool, default = False)
parser.add_argument('--eval_interval', type = int, default = 1)
parser.add_argument('--subset_size', type = int, default = 10_000) # for computing the training loss during evaluation

cfg = parser.parse_args()

cfg.lr_min = cfg.lr_peak / cfg.lr_min_factor

# create directories if they do not exist
os.makedirs(cfg.results_dir, exist_ok = True)

# set random seed
torch.manual_seed(cfg.random_seed)

### LOAD DATASET ###
cfg.in_dim, cfg.num_classes = data_utils.DATASET_DIMS['cifar-10']

(x_train, y_train), (x_test, y_test) = data_utils.load_dataset(cfg.dataset_name, cfg.data_dir)
if cfg.loss_name == 'mse':
    # One-hot encode y for MSELoss
    y_train = data_utils._one_hot(y_train, num_classes = cfg.num_classes)
    y_test = data_utils._one_hot(y_test, num_classes = cfg.num_classes)
            
# create a small dataset for Hessian computation
x_subset, y_subset = x_train[:cfg.subset_size], y_train[:cfg.subset_size]


### MODEL AND OPTIMIZER ###

model = model_utils.FCN(in_dim = cfg.in_dim, width = cfg.width, depth = cfg.depth, out_dim = cfg.num_classes, act_name = cfg.act_name)
    
# Initialize weights
model_utils.initialize_weights(model, varw = cfg.varw)
model = model.to(device)

# Print model info
print(f'\nModel Parameters: {model_utils.count_parameters(model)/1e06:0.2f}M')

# loss function
loss_fn = LOSS_FUNCTIONS[cfg.loss_name]

# optimizer
optimizer_grouped_parameters = [
            {
                "params": [
                    (n, p)
                    for n, p in model.named_parameters()
                ],
                "weight_decay": cfg.weight_decay,
            },
        ]
        
optim = SGD(optimizer_grouped_parameters, lr = cfg.lr_init, momentum = cfg.momentum, weight_decay = cfg.weight_decay)

exp_id = f'{cfg.dataset_name}_{cfg.loss_name}_I{cfg.random_seed}_{cfg.model_name}_n{cfg.width}_d{cfg.depth}_{cfg.act_name}_varw{cfg.varw}_B{cfg.batch_size}_opt{cfg.opt_name}_lr{cfg.lr_peak:.0e}_T{cfg.num_steps}_Tw{cfg.warmup_steps}_r{cfg.warmup_exponent}_Ts{cfg.stable_steps}_{cfg.decay_schedule_name}_p{cfg.decay_exponent}_m{cfg.momentum}_wd{cfg.weight_decay}'
train_path = os.path.join(cfg.results_dir, f'train_{exp_id}.csv')
forward_path = os.path.join(cfg.results_dir, f'forward_{exp_id}.csv')
eval_path = os.path.join(cfg.results_dir, f'eval_{exp_id}.csv')


# Training go Brrr.....
train_results, eval_results, forward_results = train_and_evaluate(cfg, model, loss_fn, optim, x_train, y_train, x_test, y_test, x_subset, y_subset)

df_train = pd.DataFrame(train_results, columns = ['step', 'lr_step', 'loss_step'])
df_train.to_csv(train_path)

df_eval = pd.DataFrame(eval_results, columns = ['step', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
df_eval.to_csv(eval_path)

df_forward = pd.DataFrame(forward_results, columns = ['step', 'sharpness', 'num_iters_power', 'dir_sharpness', 'lr_lower', 'lr_upper', 'num_iters_critical'])
df_forward.to_csv(forward_path)