# Agent Guidelines for GPS Repository

This file provides guidelines for AI coding agents working in this repository.

## Project Overview

GPS (Graph Positional Encoding with Self-Attention) is a Graph Neural Network framework built on PyTorch Geometric. It implements various positional encodings (RWSE, LapPE, etc.) combined with transformer-style attention for graph representation learning.

## Build & Run Commands

### Environment Setup

```bash
# Install dependencies with pixi
pixi install

# Activate environment
pixi shell
```

### Running Experiments

```bash
# Train with a config file
pixi run python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml wandb.use=False

# Run with multiple seeds
pixi run python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml --repeat 5 wandb.use=False
```

### Running Tests

```bash
# Run all unit tests
python -m pytest unittests/

# Run all unit tests with unittest
python -m unittest discover -s unittests

# Run a single test file
python -m unittest unittests.test_eigvecs

# Run a specific test
python -m unittest unittests.test_eigvecs.TestEigvecsNormalization.test_L1
```

## Code Style Guidelines

### Formatting

- **Use Black** for code formatting (version >= 26.1.0)
- Maximum line length: 88 characters (Black default)
- Run `black .` before committing

### Import Order

1. Standard library imports
2. Third-party imports (torch, numpy, sklearn, etc.)
3. Local/graphgps imports

```python
# Good example
import os
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error

import graphgps
from graphgps.logger import CustomLogger
from graphgps.metrics_ogb import some_metric
```

### Naming Conventions

- **Variables/functions**: `snake_case` (e.g., `def get_feature_encoder()`)
- **Classes**: `PascalCase` (e.g., `class GPSLayer(nn.Module)`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_HIDDEN_DIM = 256`)
- **Private methods**: prefix with underscore (e.g., `_compute_embedding()`)

### Type Hints

- Add type hints for function parameters and return values when beneficial
- Use `Optional[X]` instead of `X | None` for Python 3.13 compatibility
- Common types: `List`, `Dict`, `Tuple`, `Optional`, `Union`

```python
def process_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    ...
```

### Error Handling

- Use specific exceptions instead of bare `except:`
- Include meaningful error messages
- Validate inputs at function boundaries

```python
# Good
def load_config(cfg_path: str) -> dict:
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    ...

# Avoid
def load_config(cfg_path):
    try:
        ...
    except:
        pass
```

### PyTorch Conventions

- Use `nn.Module` for neural network layers
- Call `super().__init__()` in `__init__` methods
- Use `self.register_buffer()` for non-learnable tensors
- Use `self.register_parameter()` for learnable parameters

```python
class CustomLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.register_buffer('scale', torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x)) * self.scale
```

### Logging

- Use the logging module for important information
- Use appropriate log levels: `logging.debug()`, `logging.info()`, `logging.warning()`, `logging.error()`
- Include context in log messages

```python
logger = logging.getLogger(__name__)

logger.info(f"Loaded dataset '{dataset_name}' with {num_graphs} graphs")
logger.warning(f"Expected undirected edges, got {num_directed} directed edges")
```

### Documentation

- Use docstrings for public functions and classes
- Follow Google or NumPy docstring style
- Include Args, Returns, and Raises sections for complex functions

```python
def compute_positional_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    dim_pe: int,
) -> torch.Tensor:
    """Compute random walk structural encoding for graph nodes.

    Args:
        edge_index: Graph edge indices (2, num_edges)
        num_nodes: Number of nodes in the graph
        dim_pe: Dimension of positional encoding

    Returns:
        Positional encoding tensor of shape (num_nodes, dim_pe)
    """
    ...
```

### Git Conventions

- Make focused, atomic commits
- Use meaningful commit messages
- Do not commit large files, generated outputs, or sensitive data
- Add relevant file patterns to `.gitignore`

### Performance Considerations

- Use `torch.no_grad()` for inference
- Use `@torch.jit.script` for performance-critical functions
- Prefer in-place operations when safe (e.g., `x.relu_()` vs `F.relu(x)`)
- Use `torch.jit.script` for model inference when possible

### Testing Guidelines

- Place tests in `unittests/` directory
- Use `unittest.TestCase` class structure
- Test edge cases and error conditions
- Use `torch.testing.assert_close()` for tensor comparisons
- Use `np.testing.assert_array_almost_equal()` for numpy arrays

```python
class TestMyFunction(unittest.TestCase):
    def test_basic_case(self):
        result = my_function(input_data)
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result, expected)

    def test_error_case(self):
        with self.assertRaises(ValueError):
            my_function(invalid_input)
```

## Common Patterns

### Config-Based Training

The project uses YACS configuration system. Configs are stored in YAML files under `configs/`. Key config sections:

- `dataset`: Data loading and preprocessing
- `model`: Model architecture
- `train`: Training loop settings
- `optim`: Optimizer and scheduler

### Positional Encodings

This project implements multiple positional encodings:
- `RWSE`: Random Walk Structural Encoding
- `LapPE`: Laplacian Positional Encoding
- `EquivStableLapPE`: Equivariant Stable Laplacian PE
- `SignNet`: Signnet positional encoding

### Model Architecture

The main model is `GPSModel` combining:
- Node/edge encoders
- Multiple `GPSLayer` layers (GNN + attention)
- Graph head for final prediction
