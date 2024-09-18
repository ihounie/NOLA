#  ------------------------------------------------------------------------------------------
#  Implementing LorTa (Low-Rank Tensor Adapters)
#  ------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LorTaLayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
        num_heads: int,
        num_layers: int,
        num_modules: int,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_modules = num_modules  # e.g., 4 for Q, K, V, O

class LorTaLinear(nn.Linear, LorTaLayer):
    # LorTa implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        num_heads: int = 1,
        num_layers: int = 1,
        num_modules: int = 1,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LorTaLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            num_heads=num_heads,
            num_layers=num_layers,
            num_modules=num_modules,
        )
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'A'):
            # Initialize A and B
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
            nn.init.normal_(self.C_H)
            nn.init.normal_(self.C_L)
            nn.init.normal_(self.C_M)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        # You can implement weight merging here if needed

    def forward(self, x: torch.Tensor, head_index: int = 0, layer_index: int = 0, module_index: int = 0):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0:
            # Compute the combined scaling vector c
            c = self.C_H[head_index, :] * self.C_L[layer_index, :] * self.C_M[module_index, :]  # shape (r,)
            # Apply the scaling to A
            scaled_A = self.A * c.unsqueeze(1)  # shape (r, in_features)
            # Compute the weight update
            delta_W = self.B @ scaled_A  # (r, out_features) @ (r, in_features) -> (out_features, in_features)
            delta_W = delta_W * self.scaling
            delta_W = T(delta_W)
            # Compute the output
            result = F.linear(x, T(self.weight) + delta_W, bias=self.bias)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
