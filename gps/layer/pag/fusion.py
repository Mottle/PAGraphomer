import torch
from torch import nn
from torch_geometric.nn import GCNConv, Sequential


class AFF(nn.Module):
    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dense = nn.Sequential(nn.Linear(2 * channels, channels), nn.Sigmoid())

    def forward(self, x, y):
        concated = torch.cat([x, y], dim=-1)
        alpha = self.dense(concated)
        out = x * alpha + y * (1 - alpha)

        return out


class IAFF(nn.Module):
    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense_alpha = nn.Sequential(
            nn.Linear(2 * channels, channels), nn.Sigmoid()
        )
        self.dense_beta = nn.Sequential(nn.Linear(2 * channels, channels), nn.Sigmoid())

    def forward(self, x, y):
        concated = torch.cat([x, y], dim=-1)
        alpha = self.dense_alpha(concated)
        concated_beta = torch.cat([x * alpha, y * (1 - alpha)], dim=-1)
        beta = self.dense_beta(concated_beta)
        out = x * beta + y * (1 - beta)
        return out


# class GAFF(nn.Module):
#     def __init__(self, channels: int, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.backbone = Sequential(
#             "x, edge_index, batch", [GCNConv(channels, channels), nn.LeakyReLU()]
#         )
#         self.backbone_linear = nn.Linear(channels, channels)

#     def forward(self, xs, ys, x, y, batch):
#         xs = self.backbone(xs, batch)
#         ys = self.backbone(ys, batch)
