# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Tuple
from torchvision.datasets import CIFAR10, CIFAR100

DATASET_DIMS = {
    'cifar-10': (32*32*3, 10), 
    'cifar-100': (32*32*3, 100)
    }

DATASETS = {
    'cifar-10': CIFAR10,
    'cifar-100': CIFAR100
}


def standardize(x_train: np.ndarray, x_test: np.ndarray):
    mean = x_train.mean(0)
    std = x_train.std(0)

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test

def flatten(x: np.ndarray):
    return x.reshape(x.shape[0], -1)

def unflatten(x: np.ndarray, shape: Tuple):
    return x.reshape(x.shape[0], *shape)

def _one_hot(x: np.array, num_classes: int):
    """
    Create a one-hot encoding of x of size num_classes
    """
    return np.array(x[..., None] == np.arange(num_classes), dtype = np.float32)

def load_dataset(dataset_name: str, data_dir: str = './'):

    "Loads the data; currently supporting CIFAR-10 and CIFAR-100"

    train_ds = DATASETS[dataset_name](root = data_dir, download = False, train = True)
    test_ds = DATASETS[dataset_name](root = data_dir, download = False, train = False)

    x_train, y_train = np.asarray(train_ds.data) / 255.0, np.asarray(train_ds.targets)
    x_test, y_test = np.asarray(test_ds.data) / 255.0, np.asarray(test_ds.targets)

    # standardize the dataset
    x_train, x_test = standardize(x_train, x_test)

    return (x_train, y_train), (x_test, y_test)
