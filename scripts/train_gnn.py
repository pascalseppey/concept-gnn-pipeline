#!/usr/bin/env python3
"""Training script for patch-based GNN with coverage-aware dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from tqdm import tqdm


def load_dataset(path: Path) -> Tuple[List[Data], Dict[str, int]]:
    with path.open() as f:
        records = [json.loads(line) for line in f]

    edge_index = _build_edge_index()
    command_map: Dict[str, int] = {}
    samples: List[Data] = []

    for rec in records:
        signature = json.dumps(rec["commands"], sort_keys=True)
        if signature not in command_map:
            command_map[signature] = len(command_map)
        label = command_map[signature]
        patch_feats = torch.tensor(rec["patch_features"], dtype=torch.float32)
        data = Data(x=patch_feats, edge_index=edge_index)
        data.y = torch.tensor([label], dtype=torch.long)

        # global metrics vector
        metrics = rec["metrics"]
        global_vec = torch.tensor(
            [
                metrics["density"],
                metrics["binary_entropy"],
                metrics["permutation_entropy"],
                metrics["lz_ratio"],
                metrics["mutual_info_local"],
                metrics["fft_anisotropy"],
                metrics["fft_radial_energy"],
                metrics["components"],
                metrics["mean_component_area"],
                metrics["hamming_from_base"],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        data.global_feats = global_vec
        samples.append(data)
    return samples, command_map


def _build_edge_index() -> torch.Tensor:
    patch_dim = 16
    size = 256
    steps = size // patch_dim
    edges = []
    for r in range(steps):
        for c in range(steps):
            idx = r * steps + c
            if r + 1 < steps:
                edges.append((idx, (r + 1) * steps + c))
                edges.append(((r + 1) * steps + c, idx))
            if c + 1 < steps:
                edges.append((idx, r * steps + (c + 1)))
                edges.append((r * steps + (c + 1), idx))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


class FusionGNN(nn.Module):
    def __init__(self, in_channels: int, global_dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden, heads=4, concat=False, dropout=0.1)
        self.conv2 = TransformerConv(hidden, hidden, heads=4, concat=False, dropout=0.1)
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        patch_emb = global_mean_pool(x, batch)
        global_feats = data.global_feats
        global_emb = self.global_mlp(global_feats)
        fused = torch.cat([patch_emb, global_emb], dim=-1)
        return self.head(fused)


def split_dataset(samples: List[Data], train_ratio: float = 0.8) -> Tuple[List[Data], List[Data]]:
    n = len(samples)
    split = int(n * train_ratio)
    indices = np.random.permutation(n)
    train_idx = indices[:split]
    test_idx = indices[split:]
    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in test_idx]
    return train, test


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=-1)
            correct += int((preds == batch.y.view(-1)).sum())
            total += batch.num_graphs
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN with metric-aware dataset.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML configuration file")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to dataset JSONL")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Epoch interval for saving checkpoints")
    parser.add_argument("--log-dir", type=Path, default=Path("analysis/train_logs"))
    args = parser.parse_args()

    if args.config is not None:
        with args.config.open() as f:
            cfg = yaml.safe_load(f) or {}
        if args.dataset is None:
            args.dataset = Path(cfg.get("dataset", "")) if cfg.get("dataset") else None
        args.epochs = cfg.get("epochs", args.epochs)
        args.batch_size = cfg.get("batch_size", args.batch_size)
        args.lr = cfg.get("learning_rate", args.lr)
        args.weight_decay = cfg.get("weight_decay", args.weight_decay)
        args.checkpoint_every = cfg.get("checkpoint_every", args.checkpoint_every)
        if cfg.get("log_dir"):
            args.log_dir = Path(cfg["log_dir"])

    if args.dataset is None:
        parser.error("--dataset must be specified either via CLI or config file")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    samples, command_map = load_dataset(args.dataset)
    train_samples, valid_samples = split_dataset(samples)

    train_loader = DataLoader(train_samples, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_samples, batch_size=args.batch_size)

    in_channels = train_samples[0].x.shape[1]
    global_dim = train_samples[0].global_feats.shape[-1]
    num_classes = len(command_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionGNN(in_channels, global_dim, hidden=256, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = args.log_dir / "metrics_log.jsonl"
    with log_path.open("w") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_graphs = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}", leave=False)
            for batch in progress:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = F.cross_entropy(logits, batch.y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
                total_graphs += batch.num_graphs
                progress.set_postfix(loss=loss.item())
            avg_loss = total_loss / max(total_graphs, 1)
            train_acc = evaluate(model, train_loader, device)
            val_acc = evaluate(model, valid_loader, device)
            record = {"epoch": epoch, "train_loss": avg_loss, "train_acc": train_acc, "val_acc": val_acc}
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | train acc {train_acc:.3f} | val acc {val_acc:.3f}")

            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                ckpt_path = args.log_dir / f"ckpt_epoch{epoch:02d}.pt"
                torch.save({"model": model.state_dict(), "command_map": command_map}, ckpt_path)


if __name__ == "__main__":
    main()
