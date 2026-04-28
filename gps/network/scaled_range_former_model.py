import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network

from gps.network.grit_model import GritTransformer, FeatureEncoder


@register_network("ScaledRangeFormerModel")
class ScaledRangeFormerModel(GritTransformer):
    """Unified network shell for ScaledRangeFormer layer variants.

    Reuses the GRIT-style encoder and masked pretraining pipeline while allowing
    `gt.layer_type` to switch between Shared / Mixed / Sequential / Bypass layer
    variants under one network name.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__(dim_in=dim_in, dim_out=dim_out)


@register_network("ScaledRangeFormerPretrainModel")
class ScaledRangeFormerPretrainModel(torch.nn.Module):
    """Thin wrapper that makes the intended pretrain network explicit.

    Functionally this is equivalent to `ScaledRangeFormerModel`; the separate
    registration name is useful for experiment bookkeeping.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.model = ScaledRangeFormerModel(dim_in=dim_in, dim_out=dim_out)

    def forward(self, batch):
        return self.model(batch)
