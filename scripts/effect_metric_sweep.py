#!/usr/bin/env python3
"""Deterministic sweep of effect chains with detailed metric logging."""

from __future__ import annotations

import csv
import math
import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
from tqdm import tqdm

from src.bijective_pipeline import Command, EFFECTS  # type: ignore


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


def binary_entropy(mat: np.ndarray) -> float:
    p = mat.mean()
    if p in (0.0, 1.0):
        return 0.0
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


def permutation_entropy(mat: np.ndarray, order: int = 3) -> float:
    data = mat.flatten()
    n = len(data)
    if n < order:
        return 0.0
    counts: Dict[Tuple[int, ...], int] = {}
    for i in range(n - order + 1):
        window = tuple(data[i : i + order])
        counts[window] = counts.get(window, 0) + 1
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return float(-sum(p * math.log2(p) for p in probs))


def lz_compression_ratio(mat: np.ndarray) -> float:
    import zlib

    data = mat.astype(np.uint8).flatten()
    compressed = zlib.compress(data.tobytes(), level=9)
    return len(compressed) / len(data)


def mutual_information_local(mat: np.ndarray) -> float:
    # consider horizontal and vertical neighbor pairs
    pairs: Dict[Tuple[int, int], int] = {}
    total = 0
    rows, cols = mat.shape
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                pairs[(int(mat[r, c]), int(mat[r, c + 1]))] = pairs.get(
                    (int(mat[r, c]), int(mat[r, c + 1])), 0
                ) + 1
                total += 1
            if r + 1 < rows:
                pairs[(int(mat[r, c]), int(mat[r + 1, c]))] = pairs.get(
                    (int(mat[r, c]), int(mat[r + 1, c])), 0
                ) + 1
                total += 1
    if total == 0:
        return 0.0
    p_xy = {k: v / total for k, v in pairs.items()}
    counts_x = {k: 0.0 for k in (0, 1)}
    counts_y = {k: 0.0 for k in (0, 1)}
    for (x, y), p in p_xy.items():
        counts_x[x] += p
        counts_y[y] += p
    mi = 0.0
    for (x, y), p in p_xy.items():
        if counts_x[x] > 0 and counts_y[y] > 0:
            mi += p * math.log2(p / (counts_x[x] * counts_y[y] + 1e-12) + 1e-12)
    return mi


def fft_features(mat: np.ndarray) -> Tuple[float, float]:
    spectrum = np.fft.fft2(mat.astype(np.float32))
    mag = np.abs(spectrum)
    mag_shifted = np.fft.fftshift(mag)
    rows, cols = mat.shape
    cy, cx = rows // 2, cols // 2
    y = np.arange(rows) - cy
    x = np.arange(cols) - cx
    xv, yv = np.meshgrid(x, y)
    radius = np.sqrt(xv**2 + yv**2)
    max_r = int(radius.max())
    radial = np.zeros(max_r + 1, dtype=np.float32)
    counts = np.zeros(max_r + 1, dtype=np.float32)
    r_idx = radius.astype(int)
    for r in range(max_r + 1):
        mask = r_idx == r
        counts[r] = mask.sum()
        if counts[r] > 0:
            radial[r] = mag_shifted[mask].mean()
    # anisotropy
    freq = np.fft.fftfreq(rows)
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    energy_x = (np.abs(fx) * mag).sum()
    energy_y = (np.abs(fy) * mag).sum()
    anisotropy = (energy_x - energy_y) / (energy_x + energy_y + 1e-12)
    radial_energy = float(radial.mean())
    return anisotropy, radial_energy


def connected_components(mat: np.ndarray) -> Tuple[int, float]:
    rows, cols = mat.shape
    visited = np.zeros_like(mat, dtype=bool)
    total_area = 0
    count = 0
    for r in range(rows):
        for c in range(cols):
            if mat[r, c] == 1 and not visited[r, c]:
                count += 1
                stack = [(r, c)]
                area = 0
                visited[r, c] = True
                while stack:
                    y, x = stack.pop()
                    area += 1
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= ny < rows and 0 <= nx < cols and mat[ny, nx] == 1 and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                total_area += area
    mean_area = total_area / count if count else 0.0
    return count, mean_area


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def enumerate_sequences() -> List[List[Command]]:
    sequences: List[List[Command]] = [[]]
    roll_params = [Command("roll", {"axis": axis, "shift": shift}) for axis in (0, 1) for shift in (-64, -32, 0, 32, 64)]
    xor_params = [Command("xor_mask", {"seed": seed}) for seed in (11, 29, 47, 83)]
    perm_rows = [Command("permute_rows", {"seed": seed}) for seed in (3, 7, 13, 19)]
    perm_cols = [Command("permute_cols", {"seed": seed}) for seed in (5, 17, 23, 31)]
    block_params = [
        Command("block_shuffle", {"block": block, "seed": seed})
        for block in (8, 16, 32, 64)
        for seed in (2, 5, 11)
    ]
    single_effects = roll_params + xor_params + perm_rows + perm_cols + block_params
    sequences.extend([[cmd] for cmd in single_effects])

    # deterministic pairs
    pair_bases = [roll_params, xor_params, perm_rows, perm_cols, block_params]
    for first in pair_bases:
        for second in pair_bases:
            for cmd1 in first:
                for cmd2 in second:
                    sequences.append([cmd1, cmd2])

    # triple combinations limited to representative subset
    for cmd1 in roll_params[:3]:
        for cmd2 in perm_rows[:3]:
            for cmd3 in block_params[:3]:
                sequences.append([cmd1, cmd2, cmd3])
    for cmd1 in xor_params[:3]:
        for cmd2 in perm_cols[:3]:
            for cmd3 in block_params[:3]:
                sequences.append([cmd1, cmd2, cmd3])
    return sequences


def compute_metrics(base: np.ndarray, transformed: np.ndarray, commands: Sequence[Command]) -> Dict[str, float | int | str]:
    density = transformed.mean()
    percent_black = float(density * 100.0)
    percent_white = 100.0 - percent_black
    metrics: Dict[str, float | int | str] = {
        "num_commands": len(commands),
        "command_names": " > ".join(cmd.name for cmd in commands) if commands else "identity",
        "density": density,
        "percent_black": percent_black,
        "percent_white": percent_white,
        "binary_entropy": binary_entropy(transformed),
        "permutation_entropy": permutation_entropy(transformed),
        "lz_ratio": lz_compression_ratio(transformed),
        "mutual_info_local": mutual_information_local(transformed),
    }
    anisotropy, radial_energy = fft_features(transformed)
    components, mean_area = connected_components(transformed)
    metrics.update(
        {
            "fft_anisotropy": anisotropy,
            "fft_radial_energy": radial_energy,
            "components": components,
            "mean_component_area": mean_area,
            "hamming_from_base": hamming_distance(base, transformed),
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep effect chains and log metrics.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/effect_metric_sweep.csv"),
        help="Chemin du CSV de sortie (créera les dossiers).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optionnel : réservé pour compatibilité (pas utilisé pour l'instant).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = alternating_columns((MATRIX_SIZE, MATRIX_SIZE))
    sequences = enumerate_sequences()

    csv_path = args.output
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sequence_id",
        "num_commands",
        "command_names",
        "command_signature",
        "density",
        "percent_black",
        "percent_white",
        "binary_entropy",
        "permutation_entropy",
        "lz_ratio",
        "mutual_info_local",
        "fft_anisotropy",
        "fft_radial_energy",
        "components",
        "mean_component_area",
        "hamming_from_base",
    ]

    rows = []
    iterator = tqdm(enumerate(sequences), total=len(sequences), desc="Sweeping effects")
    for idx, commands in iterator:
        transformed = apply_commands(base, commands)
        metrics = compute_metrics(base, transformed, commands)
        row = {"sequence_id": idx, "command_signature": repr([{"name": cmd.name, **cmd.params} for cmd in commands])}
        row.update(metrics)
        rows.append(row)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {csv_path}")
    # basic sanity: print ranges
    densities = [row["density"] for row in rows]
    entropies = [row["binary_entropy"] for row in rows]
    hamming = [row["hamming_from_base"] for row in rows]
    print(f"Density range: {min(densities):.3f} – {max(densities):.3f}")
    print(f"Binary entropy range: {min(entropies):.3f} – {max(entropies):.3f}")
    print(f"Hamming distance range: {min(hamming)} – {max(hamming)}")


if __name__ == "__main__":
    main()
