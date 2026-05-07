import argparse
import logging
import os
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

_original_torch_load = torch.load


def _patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(
        f, map_location=map_location, pickle_module=pickle_module, **kwargs
    )


torch.load = _patched_torch_load


def smooth_motif_by_neighbors(motif_dist, edge_index, mode="blend", alpha=0.5, iters=3):
    K = motif_dist.size(-1)
    device = motif_dist.device
    src, dst = edge_index[0], edge_index[1]
    adj = defaultdict(list)
    for i, j in zip(src.tolist(), dst.tolist()):
        adj[i].append(j)
        adj[j].append(i)

    if mode == "vote":
        cur_id = motif_dist.argmax(dim=-1)
        for _ in range(iters):
            new_id = cur_id.clone()
            for nid, neighbors in adj.items():
                if not neighbors:
                    continue
                nb = torch.tensor(neighbors, device=device)
                neighbor_ids = cur_id[nb]
                counts = torch.zeros(K, device=device)
                counts.scatter_add_(0, neighbor_ids, torch.ones_like(neighbor_ids, dtype=torch.float))
                winner = counts.argmax()
                new_id[nid] = winner
            cur_id = new_id
        return F.one_hot(cur_id, num_classes=K).float()

    cur = motif_dist.clone()
    for _ in range(iters):
        neighbor_sum = torch.zeros_like(cur)
        neighbor_count = torch.zeros(cur.size(0), 1, device=device)
        for nid, neighbors in adj.items():
            if neighbors:
                nb = torch.tensor(neighbors, device=device)
                neighbor_sum[nid] = cur[nb].mean(dim=0)
                neighbor_count[nid] = 1
        valid = neighbor_count.squeeze(-1) > 0
        cur[valid] = (1 - alpha) * cur[valid] + alpha * neighbor_sum[valid]
    return cur


def get_colored_nodes(motif_id, edge_index):
    """Return set of node indices whose motif matches at least one neighbor."""
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    adj = defaultdict(set)
    for i, j in zip(src, dst):
        adj[i].add(j)
        adj[j].add(i)
    colored = set()
    for nid, neighbors in adj.items():
        if any(motif_id[nid] == motif_id[nb] for nb in neighbors):
            colored.add(nid)
    return colored


from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.loader import DataLoader as PYGDataloader

import gps
from gps.finetuning import load_pretrained_model_cfg

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


MOTIF_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#e6194b", "#3cb44b",
]


def get_smiles_list():
    import pandas as pd
    name = cfg.dataset.name
    root = cfg.dataset.dir
    mapping_dir = os.path.join(root, name.replace("-", "_"), "mapping")
    for fname in ["mol.csv.gz", "mol.csv"]:
        csv_path = os.path.join(mapping_dir, fname)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            col = "smiles" if "smiles" in df.columns else df.columns[0]
            return df[col].tolist()
    return None


def plot_molecule_with_motifs(smiles, motif_id, save_path, colored_nodes=None, img_size=800):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  [WARN] Invalid SMILES")
        return
    n_atoms = mol.GetNumAtoms()
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass
    h = int(img_size * 0.75)
    drawer = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(img_size, h)
    highlight_atoms = []
    highlight_colors = {}
    for i in range(n_atoms):
        if colored_nodes is not None and i not in colored_nodes:
            continue
        m = motif_id[i].item() if torch.is_tensor(motif_id) else motif_id[i]
        hex_color = MOTIF_PALETTE[m % len(MOTIF_PALETTE)]
        r = int(hex_color[1:3], 16) / 255
        g = int(hex_color[3:5], 16) / 255
        b = int(hex_color[5:7], 16) / 255
        highlight_atoms.append(i)
        highlight_colors[i] = (r, g, b)
    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms,
                        highlightAtomColors=highlight_colors)
    drawer.FinishDrawing()
    with open(save_path, "wb") as f:
        f.write(drawer.GetDrawingText())
    print(f"  Saved: {save_path}")


def plot_motif_histogram(motif_hist, save_path, title=None, img_size=800):
    K = motif_hist.size(-1)
    fig, ax = plt.subplots(figsize=(img_size / 100, img_size * 0.3 / 100))
    x = np.arange(K)
    colors = [MOTIF_PALETTE[k % len(MOTIF_PALETTE)] for k in range(K)]
    ax.bar(x, motif_hist.numpy(), color=colors)
    ax.set_xlabel("Motif ID")
    ax.set_ylabel("Soft Count")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_transport_heatmap(transport, save_path, title=None):
    w = transport.mean(dim=1).numpy()
    fig, ax = plt.subplots(figsize=(8, max(4, w.shape[0] * 0.3)))
    im = ax.imshow(w, aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("Motif ID")
    ax.set_ylabel("Node Index")
    plt.colorbar(im, ax=ax, label="Assignment Weight")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_motif_memory_tsne(motif_vectors, save_path):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [WARN] sklearn not available, skipping t-SNE")
        return
    K = motif_vectors.size(0)
    emb = TSNE(n_components=2, random_state=0, perplexity=min(30, K - 1)).fit_transform(
        motif_vectors.numpy()
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = [MOTIF_PALETTE[k % len(MOTIF_PALETTE)] for k in range(K)]
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=30)
    for k in range(K):
        ax.annotate(str(k), emb[k], fontsize=5, alpha=0.7)
    ax.set_title("Motif Memory t-SNE")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="viz_output")
    parser.add_argument("--graph_ids", type=int, nargs="+", default=None)
    parser.add_argument("--max_graphs", type=int, default=10)
    parser.add_argument("--smooth", action="store_true",
                        help="Smooth motif assignments via neighbor propagation (reduces fragmentation)")
    parser.add_argument("--smooth_mode", type=str, default="blend", choices=["blend", "vote"],
                        help="Smoothing mode: 'blend' (soft avg) or 'vote' (neighbor majority)")
    parser.add_argument("--smooth_alpha", type=float, default=0.5,
                        help="Neighbor smoothing blend (0=none, 1=only neighbors)")
    parser.add_argument("--smooth_iters", type=int, default=3,
                        help="Neighbor propagation iterations")
    parser.add_argument("--smooth_prune", action="store_true",
                        help="Remove coloring for atoms whose motif differs from all neighbors")
    parser.add_argument("--smooth_min_size", type=int, default=1,
                        help="Minimum connected-component size for motif coloring (default 1=off)")
    parser.add_argument("--img_size", type=int, default=800,
                        help="Image size in pixels (default 800, results in img_size x round(img_size*0.75))")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("[*] Loading config...")
    set_cfg(cfg)
    load_cfg(cfg, argparse.Namespace(cfg_file=args.cfg, opts=[]))
    auto_select_device()

    if cfg.pretrained.dir or cfg.pretrained.weights_path:
        load_pretrained_model_cfg(cfg)

    print(f"[*] Loading dataset: {cfg.dataset.name}")
    from ogb.graphproppred import PygGraphPropPredDataset
    full_dataset = PygGraphPropPredDataset(name=cfg.dataset.name, root=cfg.dataset.dir)

    cfg.share.dim_out = full_dataset.num_tasks
    if cfg.dataset.task_type == "classification" and cfg.share.dim_out == 2:
        cfg.share.dim_out = 1
    cfg.share.dim_in = full_dataset.num_features

    smiles_list = get_smiles_list()
    if smiles_list:
        print(f"  SMILES list loaded: {len(smiles_list)} entries")
    else:
        print("  [WARN] No SMILES list found, molecule coloring will be skipped")
    K = cfg.otformer.motif.memory_size
    print(f"  Full dataset: {len(full_dataset)} graphs, Motif memory K={K}")

    print("[*] Creating model...")
    model = create_model(to_device=False)
    print(f"[*] Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    device = torch.device(cfg.accelerator)
    model.to(device)

    graph_ids = args.graph_ids
    if graph_ids is None:
        graph_ids = list(range(min(args.max_graphs, len(full_dataset))))
    else:
        graph_ids = [g for g in graph_ids if g < len(full_dataset)]

    print(f"[*] Running inference on {len(graph_ids)} graphs...")
    sub_loader = PYGDataloader(
        [full_dataset[int(i)] for i in graph_ids],
        batch_size=min(24, len(graph_ids)),
        shuffle=False,
    )

    all_motif_hists = []
    per_batch_data = []

    for batch in sub_loader:
        batch = batch.to(device)
        with torch.no_grad():
            pred, true = model(batch)
        aux = batch.otformer_aux
        transport = aux.get("transport")
        if transport is None:
            print("[!] No transport matrix (RUM/OT disabled)")
            return
        motif_dist = transport.mean(dim=1)
        if args.smooth:
            motif_dist = smooth_motif_by_neighbors(
                motif_dist, batch.edge_index,
                mode=args.smooth_mode,
                alpha=args.smooth_alpha, iters=args.smooth_iters,
            )
            print(f"  [Smooth] mode={args.smooth_mode}, alpha={args.smooth_alpha}, iters={args.smooth_iters})")
        motif_id = motif_dist.argmax(dim=-1)
        motif_hist = aux["motif_hist_graph"]
        all_motif_hists.append(motif_hist.cpu())
        per_batch_data.append({
            "motif_id": motif_id.cpu(),
            "motif_dist": motif_dist.cpu(),
            "transport": transport.cpu(),
            "batch_vec": batch.batch.cpu(),
            "edge_index": batch.edge_index.cpu(),
        })

    motif_hists = torch.cat(all_motif_hists, dim=0)
    motif_vectors = model.model.ot_memory.memory.detach().cpu()

    print("[*] Generating visualizations...")
    global_idx = 0
    for bi, data in enumerate(per_batch_data):
        bv = data["batch_vec"]
        n_graphs = bv.max().item() + 1
        for gi in range(n_graphs):
            if global_idx >= len(graph_ids):
                break
            gid = graph_ids[global_idx]
            mask = bv == gi
            prefix = f"graph_{gid}"

            smiles = smiles_list[gid] if smiles_list and gid < len(smiles_list) else None
            motif_ids_g = data["motif_id"][mask]
            mask_indices = torch.where(mask)[0]

            colored_nodes = None
            if args.smooth_prune or args.smooth_min_size > 1:
                ei = data["edge_index"]
                mask_indices_list = mask_indices.tolist()
                mask_set = set(mask_indices_list)
                motif_local = {int(idx): int(motif_ids_g[j].item())
                               for j, idx in enumerate(mask_indices_list)}
                local_adj = defaultdict(set)
                for u, v in zip(ei[0].tolist(), ei[1].tolist()):
                    if u in mask_set and v in mask_set:
                        local_adj[u].add(v)
                        local_adj[v].add(u)

            if args.smooth_prune:
                colored_nodes = set()
                for nid, neighbors in local_adj.items():
                    if any(motif_local.get(nid) == motif_local.get(nb) for nb in neighbors):
                        colored_nodes.add(nid)
                global_to_local = {g: l for l, g in enumerate(mask_indices_list)}
                colored_nodes = {global_to_local[n] for n in colored_nodes if n in global_to_local}

            if args.smooth_min_size > 1:
                motif_to_nodes = defaultdict(list)
                for nid in mask_indices_list:
                    motif_to_nodes[motif_local[nid]].append(nid)
                keep_global = set()
                for mid, nodes in motif_to_nodes.items():
                    visited = set()
                    for seed in nodes:
                        if seed in visited:
                            continue
                        stack = [seed]
                        component = set()
                        while stack:
                            n = stack.pop()
                            if n in visited:
                                continue
                            visited.add(n)
                            component.add(n)
                            for nb in local_adj.get(n, set()):
                                if motif_local.get(nb) == mid and nb not in visited:
                                    stack.append(nb)
                        if len(component) >= args.smooth_min_size:
                            keep_global.update(component)
                global_to_local = {g: l for l, g in enumerate(mask_indices_list)}
                colored_nodes = {global_to_local[n] for n in keep_global if n in global_to_local}

            if smiles:
                plot_molecule_with_motifs(
                    smiles, motif_ids_g,
                    os.path.join(args.out, f"{prefix}_motif.png"),
                    colored_nodes=colored_nodes,
                    img_size=args.img_size,
                )

            plot_motif_histogram(
                motif_hists[global_idx],
                os.path.join(args.out, f"{prefix}_hist.png"),
                title=f"Graph {gid} Motif Histogram",
                img_size=args.img_size,
            )

            if global_idx == 0:
                plot_transport_heatmap(
                    data["transport"],
                    os.path.join(args.out, "transport_heatmap.png"),
                )

            global_idx += 1

    plot_motif_memory_tsne(
        motif_vectors,
        os.path.join(args.out, "motif_tsne.png"),
    )

    print(f"[*] Done! Outputs in {args.out}/")


if __name__ == "__main__":
    main()
