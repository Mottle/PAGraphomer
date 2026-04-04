# GPS Template

A PyTorch Geometric-based template for Graph Neural Networks with positional encodings and transformer-style attention.

Based on [GraphGPS](https://github.com/rampasek/GraphGPS).

## Features

- Multiple positional encodings: RWSE, LapPE, EquivStableLapPE, SignNet
- Various GNN layers: GatedGCN, GINE, PNA
- Global attention mechanisms: Transformer, Performer, BigBird
- Built on PyTorch Geometric and GraphGym

## Requirements

- Python 3.13+
- CUDA 12.9+ (for GPU training)
- See `pixi.toml` for full dependencies

## Quick Start

### Environment Setup

```bash
# Install dependencies
pixi install

# Activate environment
pixi shell
```

### Training

```bash
# Train with config file
pixi run python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml wandb.use=False

# Run with multiple seeds
pixi run python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml --repeat 5 wandb.use=False
```

## Project Structure

```
graphgps/
├── config/          # Configuration definitions
├── encoder/         # Node/edge encoders
├── layer/           # GNN layers (GPSLayer, etc.)
├── head/            # Graph pooling heads
├── loader/          # Data loaders
├── logger/          # Logging utilities
├── loss/            # Loss functions
├── network/         # Model architectures
├── optimizer/       # Optimizers and schedulers
├── pooling/         # Graph pooling
├── stage/           # Model stages
├── train/           # Training loops
└── transform/      # Graph transforms
```

## Adding New Components

- **Positional Encodings**: Add to `graphgps/transform/`
- **GNN Layers**: Add to `graphgps/layer/`
- **Graph Heads**: Add to `graphgps/head/`
- **Encoders**: Add to `graphgps/encoder/`

## Testing

```bash
# Run all unit tests
python -m pytest unittests/

# Run with unittest
python -m unittest discover -s unittests

# Run specific test
python -m unittest unittests.test_eigvecs
```

## Configuration

Configs are YAML files in `configs/`. Key sections:

- `dataset`: Data loading and preprocessing
- `model`: Model architecture
- `train`: Training settings
- `optim`: Optimizer and scheduler

## Citation

If you use this template, please cite the original GraphGPS paper:

```bibtex
@article{rampasek2022GPS,
  title={{Recipe for a General, Powerful, Scalable Graph Transformer}}, 
  author={Ladislav Rampášek and Mikhail Galkin and Vijay Prakash Dwivedi and Anh Tuan Luu and Guy Wolf and Dominique Beaini},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```
