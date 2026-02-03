import numpy as np
import torch
import os

def get_batch(data_dir, split, context_len, batch_size, device):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    # NOTE: use uint16 reduced the dataset size during read; ONLY USED WHEN the vocab size is less than 2^16
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype = np.uint16, mode = 'r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype = np.uint16, mode = 'r')

    # select batch_size (valid) starting points
    ix = torch.randint(len(data) - context_len, (batch_size,))
    # extract inputs
    x = torch.stack([torch.from_numpy((data[i:i+context_len]).astype(np.int64)) for i in ix])
    # extract outputs
    y = torch.stack([torch.from_numpy((data[i+1:i+1+context_len]).astype(np.int64)) for i in ix])

    if 'cuda' in device:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking = True), y.pin_memory().to(device, non_blocking = True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
