from functools import partial

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import act_dict, register_act


class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


def _safe_register_act(name, module):
    if name not in act_dict:
        register_act(name, module)


_safe_register_act("swish", partial(SWISH, inplace=cfg.mem.inplace))
_safe_register_act("lrelu_03", partial(nn.LeakyReLU, 0.3, inplace=cfg.mem.inplace))
_safe_register_act("gelu", nn.GELU)
_safe_register_act("elu", partial(nn.ELU, inplace=cfg.mem.inplace))
_safe_register_act("leakyrelu", partial(nn.LeakyReLU, 0.01, inplace=cfg.mem.inplace))
_safe_register_act("leaky_relu", partial(nn.LeakyReLU, 0.01, inplace=cfg.mem.inplace))
