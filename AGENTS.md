# GPS Repository - Agent Guidelines

**Generated:** 2026-04-04  
**Commit:** unknown (HEAD)  
**Branch:** unknown

## OVERVIEW

GPS (Graph Positional Encoding with Self-Attention) is a PyTorch Geometric-based GNN framework implementing multiple positional encodings (RWSE, LapPE, EquivStableLapPE, SignNet) with transformer-style attention for graph representation learning.

**Core Stack**: PyTorch + PyTorch Geometric + GraphGym + pixi (not conda/pip)

**Key Models**: GPS, PAG, OTFormer, SAN, GatedGCN, GINE, Graphormer

## STRUCTURE

```
./
├── main.py                 # Entry point (GraphGym extension)
├── gps/                    # Main package (modular, auto-imports)
├── configs/                # YAML configs by model type (GPS, SAN, PAG, OTFormer, etc.)
├── run/                    # SLURM batch scripts (HPC-focused)
├── tests/                  # Integration tests + configs
├── unittests/              # Unit tests (unittest framework)
├── datasets/               # Local dataset storage (raw + processed)
└── results/                # Experiment outputs
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| **Train model** | `main.py` | Extends `torch_geometric.graphgym` |
| **Model architecture** | `gps/network/` | Uses `@register_network` decorators |
| **GNN layers** | `gps/layer/` | GPSLayer, PAGLayer, RUM, OTFormerLayer, etc. |
| **Data loading** | `gps/loader/` | Master loader with 15+ dataset formats |
| **Positional encodings** | `gps/encoder/`, `gps/transform/posenc_stats.py` | RWSE, LapPE, SignNet, etc. |
| **Configs** | `configs/{model_type}/` | Organized by architecture (GPS/, SAN/, PAG/, OTFormer/, etc.) |
| **Training loops** | `gps/train/custom_train.py` | 5 registered trainers via `@register_train` |
| **OTFormer pretraining** | `gps/train/otformer_pretrain.py` | Custom pretraining loop for OTFormer |
| **Utilities** | `gps/utils.py` | Core utilities (negate_edge_index, flatten_dict, etc.) |
| **Testing** | `unittests/` | Uses unittest (not pytest) |

## CONVENTIONS (Deviations from Standard)

**1. Dual Test Directories**
- `tests/` = Integration tests + shell scripts
- `unittests/` = Unit tests (uses unittest framework)
二次元构造: 大多数项目只有一个test目录

**2. Config Organization by Model Type**
- Standard: Group by dataset/task (zinc/, ogbg-molhiv/)
- This project: Group by architecture (GPS/, SAN/, PAG/, OTFormer/, GatedGCN/, Graphormer/, GINE/)

**3. Layer Subdirectory Pattern**
- `gps/layer/` has flat files + nested subdirs (`rum/`, `pag/`)
- RUM submodule has empty `__init__.py` (standalone import pattern)
- PAG submodule lacks `__init__.py` (implicit namespace package)

**4. Example Files Everywhere**
- Each submodule has `example.py` (e.g., `gps/layer/example.py`, `gps/network/example.py`)
- Unusual pattern - most projects centralize examples

**5. Pixi for Environment Management**
- Uses `pixi.toml` instead of conda/pip/poetry
- No `setup.py`, `pyproject.toml`, or `requirements.txt`

**6. GraphGym Extension Pattern**
- Imports from `torch_geometric.graphgym.*`
- Uses `@register_*` decorators extensively
- Custom training modes via `cfg.train.mode`

**7. Module Auto-Discovery**
- All subpackages use glob pattern in `__init__.py`:
  ```python
  modules = glob.glob(join(dirname(__file__), "*.py"))
  __all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")]
  ```
- New files automatically imported without modifying `__init__.py`

## ANTI-PATTERNS (THIS PROJECT)

**DO NOT:**
- Use `X | None` type hints - use `Optional[X]` (Python 3.13 compatibility)
- Add `__init__.py` to `gps/layer/pag/` (relies on implicit namespace package)
- Export from `gps/layer/rum/__init__.py` (intentionally empty)
- Use `pytest` fixtures - uses `unittest.TestCase` with `setUpClass`/`tearDownClass`
- Write configs in standard location - use `configs/{model_type}/`

**CRITICAL WARNING** (`gps/config/defaults_config.py` lines 9-12):
> "At the time of writing, the order in which custom config-setting functions like this one are executed is random... Therefore never reset here config options that are custom added, only change those that exist in core GraphGym."

## UNIQUE STYLES

**1. Layer Type String Format**
- Configs use: `layer_type: None+Transformer` or `layer_type: GINE+Transformer`
- Parsed as: `local_gnn_type, global_model_type = macro_gps_layer_type.split("+")`

**2. PAG Configuration**
- Custom section under `pag:` (not `model:`)
- Nested: `pag.layer_defaults.macro`, `pag.layer_defaults.local`, etc.

**3. OTFormer Configuration**
- Custom section under `otformer:` (not `model:`)
- Dual sub-sections: `otformer.motif:` for Sinkhorn OT settings, `otformer.pretrain:` for pretraining tasks
- Pretraining modes: `joint`, `atom_only`, `motif_only`, `edge_only`, `no_ot`

**4. RUM Test Import Pattern**
- Tests use: `from rum.layers import RUMLayer` (not `gps.layer.rum.layers`)
- Works because tests run from within `rum/` directory context

**5. Chinese Comments in PAG**
- `path_attention.py` has Chinese comments (e.g., "计算注意力权重的熵极小化损失")

**6. BigBird Block-Sparse Attention**
- `bigbird_layer.py` (1932 lines) - 5-part attention computation
- Each query block (first, second, middle, second-last, last) uses different patterns

## COMMANDS

```bash
# Environment
pixi install        # Install dependencies
pixi shell          # Activate environment

# Training
pixi run python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml wandb.use=False
pixi run python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml --repeat 5 wandb.use=False

# Testing
python -m pytest unittests/
python -m unittest discover -s unittests
python -m unittest unittests.test_eigvecs

# Batch execution (HPC/SLURM)
./run/run_experiments.sh
```

## NOTES

**Complexity Hotspots:**
1. `bigbird_layer.py` (1932 lines) - Google Research BigBird port
2. `performer_layer.py` (796 lines) - Performer linear attention
3. `master_loader.py` (698 lines) - 15+ dataset format handling
4. `pcqm4mv2_contact.py` (560 lines) - Link prediction with custom negative sampling

**Hidden Patterns:**
- RUM submodule has empty `__init__.py` - tests import via `from rum.layers`
- PAG uses implicit namespace package (no `__init__.py`)
- Custom training modes: `standard` vs `custom` in `cfg.train.mode`

**Cross-Module Dependencies:**
- `gps/network/pag_model.py` imports from `gps.layer.pag.fusion`, `gps.layer.pag_layer`, etc.
- `gps/train/custom_train.py` uses `gps.utils`, `gps.loss`, `gps.logger`
- All modules auto-discovered via glob pattern in respective `__init__.py`

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

### OTFormer Architecture

OTFormer (Optimal Transport Transformer) is a specialized model using Sinkhorn optimal transport for motif matching:

**Core Components**:
- **Sinkhorn Algorithm**: Entropy-regularized optimal transport for matching paths to motifs
- **OTMotifMemory**: Learnable motif memory matched via OT to random walk paths
- **OTFormerBlock**: Dual-track transformer processing node and pair representations
- **RUM Model**: Random Walk Model for path feature extraction
- **Recycling Mechanism**: Iterative refinement of representations

**Pretraining Tasks**:
1. **Masked Atom Prediction**: Predict masked atom types (molecular graphs)
2. **Motif Prediction**: Predict motif membership from OT assignments
3. **Edge Denoising**: Distinguish true edges from noise (perturbed edges)
4. **OT Prior Loss**: Regularize transport matrix to minimize transport cost

**Integration**:
- Follows same registration pattern as GPSModel, PAGModel
- Uses shared `FeatureEncoder` from `gps_model.py`
- Custom config section: `otformer:` with `motif:` and `pretrain:` sub-sections
- Located in `configs/OTFormer/` directory
