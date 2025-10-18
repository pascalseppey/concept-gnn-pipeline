#!/usr/bin/env python3
"""Dataset generator with adaptive coverage and multi-pass support."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import compute_patch_features, summarize_metrics  # type: ignore
from src.effects import EffectConfig, generate_sequence_pool  # type: ignore

Shape = Tuple[int, int]
MATRIX_SIZE = 256
BASE_MATRIX: np.ndarray | None = None


def alternating_columns(shape: Shape) -> np.ndarray:
    rows, cols = shape
    pattern = np.fromiter((1 if c % 2 == 0 else 0 for c in range(cols)), dtype=np.uint8)
    return np.tile(pattern, (rows, 1))


def init_worker() -> None:
    global BASE_MATRIX
    if BASE_MATRIX is None:
        BASE_MATRIX = alternating_columns((MATRIX_SIZE, MATRIX_SIZE))


def apply_serialized_commands(serialized: List[Dict[str, int]]) -> np.ndarray:
    from src.bijective_pipeline import Command, EFFECTS  # import local pour workers

    assert BASE_MATRIX is not None
    out = BASE_MATRIX.copy()
    for entry in serialized:
        name = entry["name"]
        params = {k: v for k, v in entry.items() if k != "name"}
        cmd = Command(name=name, params=params)
        fn, _ = EFFECTS[cmd.name]
        out = fn(out, cmd.params)
    return out


def metric_bins(metrics: Dict[str, float], bin_config: Dict[str, List[float]]) -> Tuple[int, ...]:
    coords = []
    for key, edges in bin_config.items():
        value = metrics[key]
        idx = 0
        while idx < len(edges) and value >= edges[idx]:
            idx += 1
        coords.append(idx)
    return tuple(coords)


def compute_metrics_task(args: Tuple[int, List[Dict[str, int]]]) -> Tuple[int, List[Dict[str, int]], Dict[str, float], np.ndarray]:
    idx, serialized = args
    mat = apply_serialized_commands(serialized)
    metrics = summarize_metrics(mat, BASE_MATRIX)  # type: ignore[arg-type]
    return idx, serialized, metrics, mat


def adaptive_threshold(target_counts: Dict[Tuple[int, ...], int], attempts: int) -> int:
    base_threshold = 1
    if attempts > 3:
        base_threshold = 2
    if attempts > 6:
        base_threshold = 3
    return base_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset with metric coverage.")
    parser.add_argument("--config", type=Path, default=None, help="YAML configuration file")
    parser.add_argument("--output", type=Path, default=Path("analysis/dataset.jsonl"), help="Output JSONL file")
    parser.add_argument("--max-samples", type=int, default=2000, help="Total samples to keep")
    parser.add_argument("--patch-variant", choices=["spectral", "correlation", "hybrid"], default="spectral")
    parser.add_argument("--coverage-threshold", type=float, default=0.3, help="Target ratio of samples reserved for coverage (0-1)")
    parser.add_argument("--min-per-bin", type=int, default=3, help="Minimum samples per bin before rejection")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--passes", type=int, default=1, help="Number of adaptive passes")
    args = parser.parse_args()

    effect_cfg = EffectConfig(
        roll_shifts=[-64, -32, 0, 32, 64],
        xor_seeds=[11, 29, 47, 83],
        perm_rows_seeds=[3, 7, 13, 19],
        perm_cols_seeds=[5, 17, 23, 31],
        block_sizes=[8, 16, 32, 64],
        block_seeds=[2, 5, 11],
    )

    bin_config_path = ROOT / "config" / "bins.yml"
    with bin_config_path.open() as f:
        bin_config = yaml.safe_load(f)

    if args.config is not None:
        with args.config.open() as f:
            raw_cfg = yaml.safe_load(f) or {}
        args.max_samples = raw_cfg.get("max_samples", args.max_samples)
        args.coverage_threshold = raw_cfg.get("coverage_threshold", args.coverage_threshold)
        args.min_per_bin = raw_cfg.get("min_per_bin", args.min_per_bin)
        args.patch_variant = raw_cfg.get("patch_variant", args.patch_variant)
        effect_cfg = EffectConfig(
            roll_shifts=raw_cfg.get("roll_shifts", effect_cfg.roll_shifts),
            xor_seeds=raw_cfg.get("xor_seeds", effect_cfg.xor_seeds),
            perm_rows_seeds=raw_cfg.get("perm_rows_seeds", effect_cfg.perm_rows_seeds),
            perm_cols_seeds=raw_cfg.get("perm_cols_seeds", effect_cfg.perm_cols_seeds),
            block_sizes=raw_cfg.get("block_sizes", effect_cfg.block_sizes),
            block_seeds=raw_cfg.get("block_seeds", effect_cfg.block_seeds),
            max_depth=raw_cfg.get("max_depth", effect_cfg.max_depth),
        )

    global BASE_MATRIX
    BASE_MATRIX = alternating_columns((MATRIX_SIZE, MATRIX_SIZE))
    sequences = generate_sequence_pool(effect_cfg)

    num_bins = int(np.prod([len(v) + 1 for v in bin_config.values()]))
    target_per_bin = max(args.min_per_bin, math.ceil(args.max_samples * args.coverage_threshold / num_bins))

    bin_counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    kept = 0
    passes = args.passes

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for pass_idx in range(1, passes + 1):
        print(f"=== Pass {pass_idx}/{passes} ===")
        target = adaptive_threshold(bin_counts, pass_idx)

        tasks = list(enumerate(sequences))
        results: List[Tuple[int, List[Dict[str, int]], Dict[str, float], np.ndarray]] = []

        with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker) as executor:
            iterator = tqdm(executor.map(compute_metrics_task, tasks), total=len(tasks), desc="Compute metrics")
            for res in iterator:
                results.append(res)

        results.sort(key=lambda x: x[0])

        with output_path.open("a") as f:
            for seq_id, serialized, metrics, mat in tqdm(results, desc="Selecting samples"):
                bin_id = metric_bins(metrics, bin_config)
                if bin_counts[bin_id] >= max(target_per_bin, target):
                    continue
                patch_features = compute_patch_features(mat, patch_size=16, variant=args.patch_variant)
                record = {
                    "sequence_id": seq_id,
                    "commands": serialized,
                    "metrics": metrics,
                    "patch_features": patch_features.tolist(),
                    "matrix_bits": mat.flatten().tolist(),
                }
                f.write(json.dumps(record) + "\n")
                bin_counts[bin_id] += 1
                kept += 1
                if kept >= args.max_samples:
                    break

        if kept >= args.max_samples:
            print("Quota global atteint")
            break

    filled_bins = sum(1 for count in bin_counts.values() if count > 0)
    print(f"Wrote {kept} samples to {output_path}")
    print(f"Bins filled: {filled_bins}/{num_bins} (target per bin {target_per_bin})")
    print("Coverage summary (non-empty bins):")
    for bin_id, count in sorted(bin_counts.items()):
        if count:
            print(f"  bin {bin_id}: {count}")


if __name__ == "__main__":
    init_worker()
    main()
