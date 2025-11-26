"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.scale.shape, weight = self.scale, bias = self.bias, eps = 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.embd_dim % config.num_heads == 0
        self.num_heads = config.num_heads # number of heads
        self.embd_dim = config.embd_dim # embedding dimension
        self.head_dim = self.embd_dim // self.num_heads # head dimension

        # key, query, value projections for all heads, but in a batch
        # separate projections for key, query, value
        # fan_in, fan_out
        self.k_proj = nn.Linear(self.embd_dim, self.num_heads*self.head_dim, bias = config.bias)
        self.q_proj = nn.Linear(self.embd_dim, self.num_heads*self.head_dim, bias = config.bias)
        self.v_proj = nn.Linear(self.embd_dim, self.num_heads*self.head_dim, bias = config.bias)
        # output projection
        self.output_proj = nn.Linear(self.num_heads*self.head_dim, self.embd_dim, bias = config.bias)
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.context_len, config.context_len)).view(1, 1, config.context_len, config.context_len))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embd_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)

        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (batch_size, num_heads, context_len, head_dim)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (batch_size, num_heads, context_len, head_dim)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (batch_size, num_heads, context_len, head_dim)

        ## NOTE: MUP edit start ###
        # 1/d scaling instead of 1/sqrt(d) scaling
        attention_scale = 1.0 / k.size(-1)
        ## MUP edit end ###

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = None, is_causal = True, scale = attention_scale)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * attention_scale
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        # transpose -> (B, T, nh, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        # output projection
        y = self.output_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mlp_up = nn.Linear(config.embd_dim, 4 * config.embd_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.mlp_proj = nn.Linear(4 * config.embd_dim, config.embd_dim, bias=config.bias)

    def forward(self, x):
        x = self.mlp_up(x)
        x = self.gelu(x)
        x = self.mlp_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn_norm = LayerNorm(config.embd_dim, bias = config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = LayerNorm(config.embd_dim, bias = config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

@dataclass
class GPTConfig:
    context_len: int = 1024
    vocab_size: int = 50304 
    num_layers: int = 12
    num_heads: int = 12
    embd_dim: int = 768
    bias: bool = False
    init_var: float = 1.0
    
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_len is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embd_dim),
            wpe = nn.Embedding(config.context_len, config.embd_dim),
            blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            final_norm = LayerNorm(config.embd_dim, bias = config.bias),
        ))

        self.lm_head = nn.Linear(config.embd_dim, config.vocab_size, bias = False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        ### NOTE: mup code edit start
        for name, param in self.named_parameters():
            # Embedding layers
            if 'wte.weight' in name or 'wpe.weight' in name:
                torch.nn.init.normal_(param, mean = 0.0, std = math.sqrt(self.config.init_var) / math.sqrt(self.config.embd_dim))
            # Special projection layers
            elif 'output_proj.weight' in name or 'mlp_proj.weight' in name:
                torch.nn.init.normal_(param, mean = 0.0, std = math.sqrt(self.config.init_var) / math.sqrt(2 * self.config.num_layers * self.config.embd_dim))
            # Regular weights
            elif 'weight' in name:
                torch.nn.init.normal_(param, mean = 0.0, std = math.sqrt(self.config.init_var) / math.sqrt(self.config.embd_dim))
            # All biases initialized to zero
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        ### mup code edit end

        # report number of parameters
        # print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    
    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params, self.transformer.wte.weight.numel()

    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.context_len, f"Cannot forward sequence of length {t}, block size is only {self.config.context_len}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, embd_dim)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, embd_dim)
        x = tok_emb + pos_emb
        ### NOTE: mup code edit start
        x *= math.sqrt(self.config.embd_dim)
        ### mup code end
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.final_norm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            ### NOTE: mup code edit start
            logits *= (1.0 / math.sqrt(self.config.embd_dim))
            ### mup code end
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def muAdamW(self, learning_rate, betas, eps, weight_decay, device_type):
        
        sqrt_width_scaling_params = []
        width_scaling_params = []
        nodecay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                    
            if param.dim() >= 2:
                if name.endswith('wte.weight') or name.endswith('wpe.weight'):
                    # first and last layer with LR = 1 /sqrt(width)
                    sqrt_width_scaling_params.append(param)
                    # print(f'{name}, {param.shape}, sqrt width + WD')
                else:
                    # includes all hidden layers with LR = 1/width
                    width_scaling_params.append(param)
                    # print(f'{name}, {param.shape}, width + WD')
            else:
                # include layernorms, biases; no LR scaling
                nodecay_params.append(param)
                # print(f'{name}, {param.shape}, 1.0 + no WD')
            
        optim_groups = [
                {'params': sqrt_width_scaling_params, 'weight_decay': weight_decay, 'lr_scale': 1.0 / math.sqrt(self.config.embd_dim)},
                {'params': width_scaling_params, 'weight_decay': weight_decay, 'lr_scale': 1.0 / self.config.embd_dim},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr_scale': 1.0}
        ]
        
        ### End muP code ###
        
        # # Create AdamW optimizer and use the fused version if it is available
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        # use_fused = fused_available and device_type == 'cuda'
        # extra_args = dict(fused = True) if use_fused else dict()

        # optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = betas, eps = eps, **extra_args)
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = betas, eps = eps, weight_decay = weight_decay)

        # print(f'Using fused AdamW: {use_fused}')

        return optimizer


    def muSGD(self, learning_rate, beta, weight_decay, device_type):
        
        decay_params = []
        nodecay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                    
            if param.dim() >= 2:
                # decay params for 2D tensors
                decay_params.append(param)
            else:
                # include layernorms, biases; no weight decay
                nodecay_params.append(param)
            
        optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr_scale': 1.0}
        ]
        
        ### End muP code ###
        
        # # Create AdamW optimizer and use the fused version if it is available
        # fused_available = 'fused' in inspect.signature(torch.optim.SGD).parameters
        # use_fused = fused_available and device_type == 'cuda'
        # extra_args = dict(fused = True) if use_fused else dict()

        # optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = betas, eps = eps, **extra_args)
        optimizer = torch.optim.SGD(optim_groups, lr = learning_rate, momentum = beta, weight_decay = weight_decay)

        # print(f'Using fused AdamW: {use_fused}')

        return optimizer



    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at context_len
            idx_cond = idx if idx.size(1) <= self.config.context_len else idx[:, -self.config.context_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

