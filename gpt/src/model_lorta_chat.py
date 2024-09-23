#  ------------------------------------------------------------------------------------------
#  Modified model_nola.py to use LorTa (Low-Rank Tensor Adapters)
#  ------------------------------------------------------------------------------------------

import logging
import math
import os
from collections import OrderedDict 
import copy
import math
from torch import Tensor
import torch.nn.init as init

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parameter import Parameter
import numpy as np 

import lortalib as lorta

# Remove the custom LoRA and NOLA implementations
# We'll use LorTaLinear from lortalib

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# Other activation functions and helper classes remain the same

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale

        self.layer_index = config.layer_index

        # Use LorTaLinear for the c_attn layer
        self.c_attn = lorta.LorTaLinear(
            nx,
            n_state * 3,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            num_heads=config.num_heads,
            num_modules=config.num_modules-1,  # Q, K, V
            fan_in_fan_out=True,
        )

        self.c_proj = lorta.LorTaLinear(
            n_state,
            nx,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            num_heads=config.num_heads,
            num_layers=config.n_layer,
            num_modules=1,  # Output projection
            fan_in_fan_out=True,
        )

        self.config = config
        
    def _attn(self, q, k, v, len_kv=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)

        if len_kv is not None:
            _len = torch.arange(k.size(-1), device=k.device)
            _input_msk =  _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10) 

        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1).contiguous()
        else:
            return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, x, layer_past=None, len_past=None):
        hidden_states = x

        x = self.c_attn(
            x,
            head_index=0,  # Adjust as needed
            layer_index=self.layer_index,
            module_index=0,  # Adjust as needed for Q, K, V
        )

        query, key, value = x.split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        len_kv = None

        if layer_past is not None:
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1

                _batch = torch.arange(0, key.shape[0], dtype=torch.long, device=key.device)

                past_key, past_value = layer_past[0], layer_past[1]

                past_key[_batch,:,len_past,:] = key.squeeze(-1)
                past_value[_batch,:,len_past,:] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        present = torch.stack((key.transpose(-2, -1), value))
        a = self._attn(query, key, value, len_kv=len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(
            a,
            head_index=0,
            layer_index=self.layer_index,
            module_index=0,
        )
        return a, present

class MLP(nn.Module):
    def __init__(self, n_state, config):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.layer_index = config.layer_index

        self.c_fc = lorta.LorTaLinear(
            nx,
            n_state,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            num_heads=1,
            num_layers=config.n_layer,
            num_modules=1,
            fan_in_fan_out=False,
        )
        self.c_proj = lorta.LorTaLinear(
            n_state,
            nx,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            num_heads=1,
            num_layers=config.n_layer,
            num_modules=1,
            fan_in_fan_out=False,
        )
        self.act = gelu

    def forward(self, x):
        h = self.act(
            self.c_fc(
                x,
                head_index=0,
                layer_index=self.layer_index,
                module_index=0,
            )
        )
        h2 = self.c_proj(
            h,
            head_index=0,
            layer_index=self.layer_index,
            module_index=0,
        )
        return h2

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False, layer_index=0):
        super(Block, self).__init__()
        nx = config.n_embd
        self.layer_index = layer_index
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, len_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past, len_past=len_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)

        self.h = nn.ModuleList()
        for layer_index in range(config.n_layer):
            block_config = copy.deepcopy(config)
            block_config.layer_index = layer_index
            block = Block(config.n_ctx, block_config, scale=True, layer_index=layer_index)
            self.h.append(block)

        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.config = config

    def forward(
        self, 
        input_ids, 
        position_ids=None, 
        token_type_ids=None, 
        past=None, 
        len_past=None
    ):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        elif len_past is None:
            past_length = past[0][0].size(-2)

        if position_ids is None and len_past is None:
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length, 
                dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        elif len_past is not None:
            position_ids = (len_past).unsqueeze(1)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)     

        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past=layer_past, len_past=len_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents

# The rest of the code remains largely unchanged, including GPT2LMHead and GPT2LMModel classes.

class GPT2Config(object):
    def __init__(
        self,
        vocab_size_or_config_json_file=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        fix_dropout=0.0,
        lora_qv=False, 
        lora_mlp=False, 
        use_nola=False,
        lora_rank=8, 
        lora_alpha=1.0, 
        nola_num_basis=1024,
        qnola=False, 
        qbits=2,
    ):
        self.vocab_size = vocab_size_or_config_json_file
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.fix_dropout = fix_dropout
        
        self.lora_qv=lora_qv 
        self.lora_mlp=lora_mlp 
        self.lora_rank=lora_rank 
        self.lora_alpha=lora_alpha 
        self.use_nola=use_nola
        self.nola_num_basis = nola_num_basis
        self.qnola=qnola
        self.qbits=qbits

class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, hidden_state):
        # Truncated Language modeling logits (we remove the last token)
        # h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits