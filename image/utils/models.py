import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Sequence, Tuple, Callable
from functools import partial

### Helper function

def count_parameters(model):
    """ Counts parameters in a model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(model, varw = 1.0):
    """
    Initialize model weights using varw/fan_in for hidden layers and 1/fan_in for output layer.    
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            fan_in = m.weight.size(1) # fan_out, fan_in 
            
            is_last_layer = False
            if isinstance(m, nn.Linear) and isinstance(getattr(m, 'out_features', None), int) and m.out_features == model.layers[-1].out_features:    
                is_last_layer = True
            
            # Initialize weights
            if is_last_layer:
                std = math.sqrt(1.0 / fan_in)
            else:
                std = math.sqrt(varw / fan_in)
            
            nn.init.normal_(m.weight, mean = 0.0, std = std)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

############
### FCNs ###
############

class FCN(nn.Module):
    def __init__(self, in_dim, width, depth, out_dim, act_name = 'relu', use_bias = False):
        super(FCN, self).__init__()

        # use bias
        self.use_bias = use_bias
        
        # Set activation function
        self.activation = self._get_activation(act_name)
        
        # Create layers as ModuleList for explicit forward pass
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(in_dim, width, bias = self.use_bias))
        
        # Hidden layers
        for _ in range(depth):
            self.layers.append(nn.Linear(width, width, bias = self.use_bias))
        
        # Output layer
        self.layers.append(nn.Linear(width, out_dim, bias = self.use_bias))
    
    def _get_activation(self, act_name: str):        
        
        activation_dict = {
            'relu': F.relu,
            'gelu': F.gelu,
            'selu': F.selu,
            'elu': F.elu,
            'leakyrelu': F.leaky_relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        
        if isinstance(act_name, str):
            act_name = act_name.lower()
            if act_name not in activation_dict:
                raise ValueError(f'Unsupported activation: {act_name}. Supported activations: {list(activation_dict.keys())}')
            return activation_dict[act_name]
        
        raise ValueError('Activation must be string')
    
    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten the image
        # forward pass        
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)        
        x = self.layers[-1](x)
        return x

############
### CNNs ###
############

class ConvBlock(nn.Module):
    """Convolution block: Convolution followed by activation"""
    def __init__(self, in_channels: int, filters: int, act: Callable = F.relu, kernel_size: Tuple[int, int] = (3, 3), strides: Tuple[int, int] = (1, 1), use_bias: bool = True):
        super().__init__()
        self.act = act
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = filters, kernel_size = kernel_size, stride = strides, padding = 'same', bias = use_bias)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x

class Myrtle(nn.Module):
    """Myrtle CNNs in Standard Parameterization"""
    def __init__(self, in_channels: int, num_filters: int, num_layers: int, num_classes: int, pool_list: Sequence[int], kernel_size: Tuple[int, int] = (3, 3), use_bias: bool = False, act: Callable = F.relu):
        super().__init__()
        self.num_layers = num_layers
        self.pool_list = pool_list
        self.act = act
        
        # Create conv layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = ConvBlock(in_channels = in_channels, filters = num_filters, act = act, kernel_size = kernel_size, use_bias = use_bias)
            self.conv_layers.append(layer)
            
        # Final dense layer
        self.classifier = nn.Linear(num_filters, num_classes, bias = use_bias)

    def forward(self, x):
        # Forward pass through conv layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if i in self.pool_list:
                x = F.avg_pool2d(x, kernel_size = 2, stride = 2)
        
        # Global average pooling
        x = torch.mean(x, dim=(2, 3))
        
        # Final classification layer
        x = self.classifier(x)
        return x

# Pre-configured models
Myrtle5 = partial(Myrtle, pool_list=[1, 2, 3], num_layers=4)
Myrtle7 = partial(Myrtle, pool_list=[1, 3, 5], num_layers=6)
Myrtle10 = partial(Myrtle, pool_list=[2, 5, 8], num_layers=9)

