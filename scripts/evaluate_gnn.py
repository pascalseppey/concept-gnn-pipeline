#!/usr/bin/env python3
"""Evaluate trained GNN checkpoints with inversion metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.train_gnn import FusionGNN, load_dataset  # type: ignore


def load_checkpoint(path: Path, in_channels: int, global_dim: int, num_classes: int) -> FusionGNN:
    checkpoint = torch.load(path, map_location="cpu")
    model = FusionGNN(in_channels, global_dim, hidden=256, num_classes=num_classes)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate(model: FusionGNN, loader: DataLoader, device: torch.device, topk: int) -> Dict[str, float]:
    total = 0
    exact = 0
    topk_hits = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            total += batch.num_graphs
            exact += int((preds == batch.y.view(-1)).sum())
            top_vals, top_idx = probs.topk(min(topk, probs.shape[1]), dim=-1)
            for target, candidates in zip(batch.y.view(-1), top_idx):
                if target.item() in candidates.cpu().tolist():
                    topk_hits += 1
    return {
        "exact_acc": exact / total if total else 0.0,
        "topk_acc": topk_hits / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    samples, command_map = load_dataset(args.dataset)
    loader = DataLoader(samples, batch_size=args.batch_size)
    in_channels = samples[0].x.shape[1]
    global_dim = samples[0].global_feats.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, in_channels, global_dim, len(command_map)).to(device)

    metrics = evaluate(model, loader, device, args.topk)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
