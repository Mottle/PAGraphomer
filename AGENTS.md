# GPS Repository - Agent Guidelines

**Generated:** 2026-05-01  **Commit:** cd5e095  **Branch:** master

## OVERVIEW

GPS (Graph Positional Encoding with Self-Attention) is a PyTorch Geometric-based GNN framework with multiple positional encodings and transformer-style attention for graph representation learning.

**Core Stack**: PyTorch + PyTorch Geometric + GraphGym + pixi

**Key Models**: GPS, PAG, OTFormer, SAN, GatedGCN, GINE, Graphormer, **GatedDeltaNet**, **ScaledRangeFormer**, **GRIT**

## STRUCTURE

```
./
├── main.py                 # Entry point (GraphGym extension)
├── gps/                    # Main package (modular, auto-imports)
│   ├── network/            # Model architectures
│   ├── layer/              # GNN layers
│   ├── encoder/            # Node/edge/PE encoders
│   ├── loader/             # Data loading
│   ├── train/              # Training loops
│   └── transform/          # Graph transforms + PE stats
├── configs/                # YAML configs by model type
├── run/                    # SLURM batch scripts
├── tests/                  # Integration tests
├── unittests/              # Unit tests (unittest framework)
├── datasets/               # Local dataset storage
└── results/                # Experiment outputs
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Train model | `main.py` | Extends `torch_geometric.graphgym` |
| Model architecture | `gps/network/` | `@register_network` decorators |
| GNN layers | `gps/layer/` | GPSLayer, PAGLayer, RUM, OTFormerLayer, etc. |
| Data loading | `gps/loader/` | Master loader with 15+ dataset formats |
| Positional encodings | `gps/encoder/`, `gps/transform/posenc_stats.py` | RWSE, LapPE, SignNet, RRWP, SPD |
| Configs | `configs/{model_type}/` | Organized by architecture |
| Training loops | `gps/train/custom_train.py` | 5 registered trainers |
| OTFormer pretraining | `gps/train/otformer_pretrain.py` | Custom pretraining loop |
| Utilities | `gps/utils.py` | Core utilities |
| Testing | `unittests/` | Uses unittest (not pytest) |

## CONVENTIONS

**1. Dual Test Directories**
- `tests/` = Integration tests + shell scripts
- `unittests/` = Unit tests (unittest framework)

**2. Config Organization by Model Type**
- Standard: Group by dataset/task
- This project: Group by architecture (GPS/, SAN/, PAG/, OTFormer/, GatedDeltaNet/, ScaledRangeFormer/, GRIT/, GatedGCN/, Graphormer/, GINE/)

**3. Layer Subdirectory Pattern**
- `gps/layer/rum/` has empty `__init__.py` (standalone import: `from rum.layers`)
- `gps/layer/pag/` lacks `__init__.py` (implicit namespace package)

**4. Pixi for Environment Management**
- Uses `pixi.toml` instead of conda/pip/poetry
- Commands: `pixi install`, `pixi shell`, `pixi run python main.py ...`

**5. GraphGym Extension Pattern**
- Uses `@register_*` decorators: `@register_network`, `@register_train`, `@register_loader`, `@register_config`
- Custom training modes: `cfg.train.mode` (`standard` vs `custom`)

**6. Module Auto-Discovery**
- All subpackages use identical glob pattern in `__init__.py`:
  ```python
  from os.path import dirname, basename, isfile, join
  import glob
  modules = glob.glob(join(dirname(__file__), "*.py"))
  __all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")]
  ```

**7. Layer Type String Format**
- Configs use: `layer_type: {local_gnn}+{global_model}`
- Examples: `GINE+Transformer`, `None+Transformer`, `GatedGCN+BigBird`
- Parsed as: `local_gnn_type, global_model_type = macro_gps_layer_type.split("+")`

**8. Custom Config Sections**
- **PAG**: `pag:` section (not under `model:`), nested: `pag.layer_defaults.macro`, `pag.layer_defaults.local`
- **OTFormer**: `otformer:` section with `otformer.motif:` (Sinkhorn OT) and `otformer.pretrain:` (pretraining tasks)
- **GatedDeltaNet**: `gt:` section with `dual_fla`, `perm_ensemble`, `perm_lambda`, `perm_mode`, `perm_pct`
- **ScaledRangeFormer**: `layer_type` variants: `Shared`, `Mixed`, `Sequential`, `Bypass`
- **GRIT**: `layer_type: GRIT` + `posenc_RRWP` for relative random walk positional encoding

## ANTI-PATTERNS

- Use `X | None` type hints - use `Optional[X]` (Python 3.13 compatibility)
- Add `__init__.py` to `gps/layer/pag/` (relies on implicit namespace package)
- Export from `gps/layer/rum/__init__.py` (intentionally empty)
- Use `pytest` fixtures - uses `unittest.TestCase` with `setUpClass`/`tearDownClass`
- Write configs in standard location - use `configs/{model_type}/`

**CRITICAL WARNING** (`gps/config/defaults_config.py` lines 9-12):
> "At the time of writing, the order in which custom config-setting functions like this one are executed is random... Therefore never reset here config options that are custom added, only change those that exist in core GraphGym."

## COMMANDS

```bash
# Environment
pixi install        # Install dependencies
pixi shell          # Activate environment

# Training
pixi run python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml wandb.use=False

# Testing
python -m unittest discover -s unittests
python -m unittest unittests.test_eigvecs

# Batch execution
./run/run_experiments.sh
```

## KEY MODELS

### GPSModel
- Core model combining local GNN + global attention
- Node/edge encoders + GPSLayer stack + graph head
- Configs in `configs/GPS/`

### PAGModel
- Path Attention Graph extending GPSModel
- Uses PAGLayer with RUMModel and path attention
- Custom `pag:` config section
- Configs in `configs/PAG/`

### OTFormerModel
- Optimal Transport Transformer with Sinkhorn OT for motif matching
- Pretraining tasks: masked atom, motif prediction, edge denoising, OT prior loss
- Custom `otformer:` config section
- Configs in `configs/OTFormer/`

### GatedDeltaNet (GDN)
- Fast Linear Attention with gated delta mechanism from `fla.layers`
- V5: Dual FLA (global + structural) with edge-to-node aggregation + neighbor mean
- Permutation ensemble for data augmentation (edge-safe partial permutation)
- Custom `gt:` config parameters: `dual_fla`, `perm_ensemble`, `perm_lambda`, `perm_mode`, `perm_pct`
- Configs in `configs/GatedDeltaNet/`

### ScaledRangeFormer
- Multi-scale graph transformer with distance masking
- Layer variants: Shared, Mixed, Sequential, Bypass
- Configs in `configs/ScaledRangeFormer/`

### GRIT
- Graph Rewiring with Iterative Transformer
- RRWP (Relative Random Walk Positional Encoding) integration
- Supports pretraining and finetuning
- Configs in `configs/GRIT/`

## COMPLEXITY HOTSPOTS

| File | Lines | Description |
|------|-------|-------------|
| `gps/layer/bigbird_layer.py` | 1932 | Google Research BigBird port, 5-part attention |
| `gps/layer/performer_layer.py` | 796 | Performer linear attention (FAVOR+) |
| `gps/network/grit_model.py` | 795 | GRIT transformer with RRWP encoding |
| `gps/layer/scaled_range_former_layer.py` | 766 | Multi-scale graph transformer |
| `gps/loader/master_loader.py` | 698 | 15+ dataset format handling |
| `gps/network/otformer_model.py` | 596 | OTFormer with Sinkhorn OT |
| `gps/loader/dataset/pcqm4mv2_contact.py` | 556 | Link prediction with negative sampling |
| `gps/transform/posenc_stats.py` | 439 | Centralized PE statistics computation |
| `gps/train/custom_train.py` | 407 | Custom training loop for PAG/OTFormer |
| `gps/encoder/signnet_pos_encoder.py` | 387 | Sign-invariant eigenvector encoder |

## HIDDEN PATTERNS

- RUM submodule: empty `__init__.py`, tests import via `from rum.layers`
- PAG: implicit namespace package (no `__init__.py`)
- Custom training modes: `standard` vs `custom` in `cfg.train.mode`
- GatedDeltaNet uses `FLA_GatedDeltaNet` from external `fla.layers` package
- GDN V5 dual-FLA: global FLA + structural FLA with edge-to-node aggregation
- Permutation ensemble: partial permutation with edge-safe node remapping
- ScaledRangeFormerModel inherits from `gps.network.grit_model.GritTransformer`

## CODE STYLE

### Formatting
- **Use Black** (version >= 26.1.0), max line length 88
- Run `black .` before committing

### Import Order
1. Standard library imports
2. Third-party imports (torch, numpy, sklearn)
3. Local/graphgps imports

### Naming Conventions
- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: prefix with underscore

### Type Hints
- Use `Optional[X]` instead of `X | None` (Python 3.13 compatibility)
- Common types: `List`, `Dict`, `Tuple`, `Optional`, `Union`

### PyTorch Conventions
- Use `nn.Module` for layers, call `super().__init__()`
- Use `self.register_buffer()` for non-learnable tensors
- Use `self.register_parameter()` for learnable parameters

### Error Handling
- Use specific exceptions with meaningful messages
- Validate inputs at function boundaries

### Testing
- Place tests in `unittests/` directory
- Use `unittest.TestCase` class structure
- Use `torch.testing.assert_close()` for tensor comparisons
- Use `np.testing.assert_array_almost_equal()` for numpy arrays

### Git
- Make focused, atomic commits
- Use meaningful commit messages
- Do not commit large files, generated outputs, or secrets
- Add relevant patterns to `.gitignore`

### Performance
- Use `torch.no_grad()` for inference
- Prefer in-place operations when safe
- Use `@torch.jit.script` for performance-critical functions
