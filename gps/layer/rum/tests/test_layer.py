import torch
from torch_geometric.data import Data


def test_layer_forward():
    from rum.layers import RUMLayer

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    g = Data(edge_index=edge_index, num_nodes=6)
    layer = RUMLayer(
        in_features=16,
        out_features=8,
        original_features=16,
        num_samples=2,
        length=3,
        self_supervise=False,
    )
    h = torch.ones(6, 16)
    out, loss = layer(g, h, h)
    assert out.shape == (2, 6, 8)
    assert loss >= 0.0


def test_layer_masks_invalid_edge_ids():
    from rum.layers import RUMLayer

    class CaptureRNN(torch.nn.Module):
        def __init__(
            self,
            input_size,
            hidden_size,
            bidirectional=False,
            num_layers=1,
            **kwargs,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.bidirectional = bidirectional
            self.num_layers = num_layers
            self.last_input = None

        def forward(self, input, h_0):
            self.last_input = input.detach().clone()
            out_dim = self.hidden_size * (2 if self.bidirectional else 1)
            if input.shape[-1] >= out_dim:
                output = input[..., :out_dim]
            else:
                padding = torch.zeros(
                    *input.shape[:-1],
                    out_dim - input.shape[-1],
                    device=input.device,
                    dtype=input.dtype,
                )
                output = torch.cat([input, padding], dim=-1)
            h_n = torch.zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                *input.shape[:-2],
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )
            return output, h_n

    def fixed_walk(g, num_samples, length, subsample=None):
        walks = torch.tensor(
            [[[0, -1, -1], [1, -1, -1], [2, -1, -1]]],
            dtype=torch.long,
        )
        eids = torch.full((1, 3, 2), -1, dtype=torch.long)
        return walks, eids

    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    g = Data(edge_index=edge_index, num_nodes=3)
    layer = RUMLayer(
        in_features=4,
        out_features=2,
        original_features=4,
        num_samples=1,
        length=3,
        edge_features=1,
        degrees=False,
        self_supervise=False,
        dropout=0.0,
        rnn=CaptureRNN,
        random_walk=fixed_walk,
    )
    h = torch.randn(3, 4)
    e = torch.tensor([[0.0], [10.0]], dtype=torch.float)
    out, loss = layer(g, h, h, e=e)
    assert out.shape == (1, 3, 2)
    assert loss == 0.0
    edge_slots = layer.rnn.last_input[..., 1::2, :]
    assert torch.allclose(edge_slots, torch.zeros_like(edge_slots))


def test_layer_self_supervise_with_mismatched_dims():
    from rum.layers import RUMLayer

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]],
        dtype=torch.long,
    )
    g = Data(edge_index=edge_index, num_nodes=6)
    layer = RUMLayer(
        in_features=16,
        out_features=8,
        original_features=16,
        num_samples=2,
        length=3,
        self_supervise=True,
    )
    layer.train()
    h = torch.randn(6, 16)
    out, loss = layer(g, h, h)
    assert out.shape == (2, 6, 8)
    assert torch.is_tensor(loss)
    assert torch.isfinite(loss)


def test_layer_edge_features_mismatch_raises():
    from rum.layers import RUMLayer

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    g = Data(edge_index=edge_index, num_nodes=2)
    layer = RUMLayer(
        in_features=4,
        out_features=4,
        original_features=4,
        num_samples=1,
        length=3,
        edge_features=0,
        self_supervise=False,
    )
    h = torch.randn(2, 4)
    e = torch.randn(2, 1)
    try:
        layer(g, h, h, e=e)
    except ValueError as exc:
        assert "edge_features" in str(exc)
    else:
        raise AssertionError("Expected ValueError for edge feature mismatch.")
