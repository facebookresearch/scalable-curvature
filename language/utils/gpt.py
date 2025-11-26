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
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, weight = self.weight, bias = self.bias, eps = 1e-5)

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
        self.flash = config.use_flash and hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.context_len, config.context_len))
                                        .view(1, 1, config.context_len, config.context_len))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embd_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)

        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

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
        self.ln_1 = LayerNorm(config.embd_dim, bias = config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.embd_dim, bias = config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    context_len: int = 1024
    vocab_size: int = 50304 
    num_layers: int = 12
    num_heads: int = 12
    embd_dim: int = 768
    bias: bool = False
    init_var: float = 1.0 # not used
    use_flash: bool = False

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_len is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embd_dim),
            wpe = nn.Embedding(config.context_len, config.embd_dim),
            h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = LayerNorm(config.embd_dim, bias = config.bias),
        ))
        self.lm_head = nn.Linear(config.embd_dim, config.vocab_size, bias = False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for name, p in self.named_parameters():
            if 'output_proj.weight' in name or 'mlp_proj.weight' in name:
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
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
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        return logits
    
    def configure_optimizers(self, learning_rate, betas, eps, weight_decay, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = betas, eps = eps, weight_decay = weight_decay)

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

