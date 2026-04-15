from torch_geometric.graphgym.register import register_config


@register_config("dataset_cfg")
def dataset_cfg(cfg):
    """Dataset-specific config options."""

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # infer-link parameters (e.g., edge prediction task)
    cfg.dataset.infer_link_label = "None"

    # Optional external CSV directory for custom scaffold split logic.
    cfg.dataset.external_smiles_csv = ""

    # Molecule dataset backend for ogbg-mol* tasks.
    # - 'ogb': default PygGraphPropPredDataset loader
    # - 'motil_csv': build dataset directly from copied MotiL CSV files
    cfg.dataset.molecule_loader = "ogb"
