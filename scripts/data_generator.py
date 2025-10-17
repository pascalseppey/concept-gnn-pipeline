#!/usr/bin/env python3
"""Dataset generator that enforces metric coverage for effect chains."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bijective_pipeline import Command, EFFECTS  # type: ignore
from src.metrics import compute_patch_features, summarize_metrics  # type: ignore


Shape = Tuple[int, int]
MATRIX_SIZE = 256


def alternating_columns(shape: Shape) -> np.ndarray:
    rows, cols = shape
    pattern = np.fromiter((1 if c % 2 == 0 else 0 for c in range(cols)), dtype=np.uint8)
    return np.tile(pattern, (rows, 1))


def apply_commands(mat: np.ndarray, commands: Iterable[Command]) -> np.ndarray:
    out = mat.copy()
    for cmd in commands:
        fn, _ = EFFECTS[cmd.name]
        out = fn(out, cmd.params)
    return out


def enumerate_sequence_pool(config: Dict[str, Iterable[int]] | None = None) -> List[List[Command]]:
    cfg = config or {}
    roll_shifts = cfg.get("roll_shifts", [-64, -32, 0, 32, 64])
    xor_seeds = cfg.get("xor_seeds", [11, 29, 47, 83])
    perm_rows_seeds = cfg.get("perm_rows_seeds", [3, 7, 13, 19])
    perm_cols_seeds = cfg.get("perm_cols_seeds", [5, 17, 23, 31])
    block_sizes = cfg.get("block_sizes", [8, 16, 32, 64])
    block_seeds = cfg.get("block_seeds", [2, 5, 11])

    pool: List[List[Command]] = [[]]
    roll_params = [Command("roll", {"axis": axis, "shift": shift}) for axis in (0, 1) for shift in roll_shifts]
    xor_params = [Command("xor_mask", {"seed": seed}) for seed in xor_seeds]
    perm_rows = [Command("permute_rows", {"seed": seed}) for seed in perm_rows_seeds]
    perm_cols = [Command("permute_cols", {"seed": seed}) for seed in perm_cols_seeds]
    block_params = [Command("block_shuffle", {"block": block, "seed": seed}) for block in block_sizes for seed in block_seeds]
    single = roll_params + xor_params + perm_rows + perm_cols + block_params
    pool.extend([[cmd] for cmd in single])

    def product(a: Sequence[Command], b: Sequence[Command]) -> Iterable[List[Command]]:
        for cmd1 in a:
            for cmd2 in b:
                yield [cmd1, cmd2]

    pair_sets = [roll_params, xor_params, perm_rows, perm_cols, block_params]
    for first in pair_sets:
        for second in pair_sets:
            pool.extend(product(first, second))

    # representative triples
    for cmd1 in roll_params[:3]:
        for cmd2 in perm_rows[:3]:
            for cmd3 in block_params[:3]:
                pool.append([cmd1, cmd2, cmd3])
    for cmd1 in xor_params[:3]:
        for cmd2 in perm_cols[:3]:
            for cmd3 in block_params[:3]:
                pool.append([cmd1, cmd2, cmd3])
    return pool


def metric_bins(metrics: Dict[str, float], bin_config: Dict[str, List[float]]) -> Tuple[int, ...]:
    coords = []
    for key, edges in bin_config.items():
        value = metrics[key]
        idx = 0
        while idx < len(edges) and value >= edges[idx]:
            idx += 1
        coords.append(idx)
    return tuple(coords)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset with metric coverage.")
    parser.add_argument("--config", type=Path, default=None, help="YAML configuration file")
    parser.add_argument("--output", type=Path, default=Path("analysis/dataset.jsonl"), help="Output JSONL file")
    parser.add_argument("--max-samples", type=int, default=2000, help="Total samples to keep")
    parser.add_argument("--patch-variant", choices=["spectral", "correlation", "hybrid"], default="spectral")
    parser.add_argument("--coverage-threshold", type=float, default=0.3, help="Target ratio of samples reserved for coverage (0-1)")
    parser.add_argument("--min-per-bin", type=int, default=3, help="Minimum samples per bin before rejecting")
    args = parser.parse_args()

    cfg: Dict[str, Iterable[int]] = {}
    if args.config is not None:
        with args.config.open() as f:
            raw_cfg = yaml.safe_load(f) or {}
        args.max_samples = raw_cfg.get("max_samples", args.max_samples)
        args.coverage_threshold = raw_cfg.get("coverage_threshold", args.coverage_threshold)
        args.min_per_bin = raw_cfg.get("min_per_bin", args.min_per_bin)
        args.patch_variant = raw_cfg.get("patch_variant", args.patch_variant)
        cfg = {
            "roll_shifts": raw_cfg.get("roll_shifts"),
            "xor_seeds": raw_cfg.get("xor_seeds"),
            "perm_rows_seeds": raw_cfg.get("perm_rows_seeds"),
            "perm_cols_seeds": raw_cfg.get("perm_cols_seeds"),
            "block_sizes": raw_cfg.get("block_sizes"),
            "block_seeds": raw_cfg.get("block_seeds"),
        }

    base = alternating_columns((MATRIX_SIZE, MATRIX_SIZE))
    sequences = enumerate_sequence_pool(cfg)

    # bins for key metrics
    bin_config = {
        "binary_entropy": [0.4, 0.7, 0.9, 1.0],
        "fft_anisotropy": [-0.5, -0.2, 0.2, 0.5, 0.8],
        "mutual_info_local": [0.02, 0.05, 0.1, 0.2],
        "hamming_from_base": [5000, 15000, 25000, 40000],
    }

    num_bins = int(np.prod([len(v) + 1 for v in bin_config.values()]))
    target_per_bin = max(args.min_per_bin, int(math.ceil(args.max_samples * args.coverage_threshold / num_bins)))
    bin_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    kept = 0

    with args.output.open("w") as f:
        iterator = tqdm(enumerate(sequences), total=len(sequences), desc="Generating dataset")
        for seq_id, commands in iterator:
            transformed = apply_commands(base, commands)
            metrics = summarize_metrics(transformed, base)
            bin_id = metric_bins(metrics, bin_config)
            if bin_counts[bin_id] >= target_per_bin:
                continue
            patch_features = compute_patch_features(transformed, patch_size=16, variant=args.patch_variant)
            record = {
                "sequence_id": seq_id,
                "commands": [{"name": cmd.name, **cmd.params} for cmd in commands],
                "metrics": metrics,
                "patch_features": patch_features.tolist(),
                "matrix_bits": transformed.flatten().tolist(),
            }
            f.write(json.dumps(record) + "\n")
            bin_counts[bin_id] += 1
            kept += 1
            if kept >= args.max_samples:
                break

    filled_bins = sum(1 for count in bin_counts.values() if count > 0)
    print(f"Wrote {kept} samples to {args.output}")
    print(f"Bins filled: {filled_bins}/{num_bins} (target per bin {target_per_bin})")
    print("Coverage summary (non-empty bins):")
    for bin_id, count in sorted(bin_counts.items()):
        if count:
            print(f"  bin {bin_id}: {count}")


if __name__ == "__main__":
    main()
