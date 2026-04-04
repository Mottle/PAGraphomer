import torch
from torch_geometric.data import Data


def test_model_forward():
    from rum.models import RUMModel

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    g = Data(edge_index=edge_index, num_nodes=6)
    model = RUMModel(
        in_features=16,
        out_features=8,
        hidden_features=12,
        depth=2,
        num_samples=2,
        length=3,
        self_supervise=False,
    )
    h = torch.ones(6, 16)
    out, loss = model(g, h)
    assert out.shape == (2, 6, 8)
    assert loss >= 0.0
