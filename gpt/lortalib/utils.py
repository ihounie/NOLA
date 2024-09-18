#  ------------------------------------------------------------------------------------------
#  Utilities for LorTa (Low-Rank Tensor Adapters)
#  ------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

from typing import Dict

from .layers import LorTaLayer, LorTaLinear

def mark_only_lorta_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if not any(name in n for name in ['A', 'B', 'C_H', 'C_L', 'C_M']):
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LorTaLinear) and \
               hasattr(m, 'bias') and \
               m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError

def lorta_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    lorta_param_names = ['A', 'B', 'C_H', 'C_L', 'C_M']
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if any(name in k for name in lorta_param_names)}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if any(name in k for name in lorta_param_names) or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if any(name in k for name in lorta_param_names):
                to_return[k] = my_state_dict[k]
                bias_name = k.split('.A')[0] + '.bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
