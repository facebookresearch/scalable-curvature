import torch
import pandas as pd

class ActivationHook:
    """Class to collect activation norms from network layers"""
    
    def __init__(self):
        self.hooks = []
        self.norm_data = []
    
    def register_hooks(self, model, verbose = False):
        """Register hooks to specified layers"""
        for name, module in model.named_modules():

            if not name:
                continue
            hook = module.register_forward_hook(self._get_hook_fn(name))
            self.hooks.append(hook)
                
        return self
    
    def _get_hook_fn(self, name):
        """ Create a hook function for a given layer"""

        def hook_fn(module, input, output):
            
            with torch.no_grad():
                activation = output.detach()
            
                if isinstance(output, torch.Tensor):
                    l1_norm = activation.abs().mean().item()
                    l2_norm = activation.pow(2).mean().sqrt().item()
                    
                    self.norm_data.append({
                        'layer': name,
                        'l1_norm': l1_norm,
                        'l2_norm': l2_norm
                    })
                
        return hook_fn
    
    def get_norms_df(self, step):
        df = pd.DataFrame(self.norm_data)
        df['step'] = step
        return df
    
    def remove(self):
        """ Remove all hooks to free memory """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.norm_data = []

