#!/usr/bin/env python3
"""Reusable metric utilities for binary 256x256 matrices."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import numpy as np


SUPPORTED_PATCH_VARIANTS = {"spectral", "correlation", "hybrid"}


def binary_entropy(mat: np.ndarray) -> float:
    p = float(mat.mean())
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def permutation_entropy(mat: np.ndarray, order: int = 3) -> float:
    flat = mat.flatten()
    n = len(flat)
    if n < order:
        return 0.0
    counts: Dict[Tuple[int, ...], int] = {}
    for idx in range(n - order + 1):
        window = tuple(int(x) for x in flat[idx : idx + order])
        counts[window] = counts.get(window, 0) + 1
    total = float(sum(counts.values()))
    probs = [c / total for c in counts.values()]
    return float(-sum(p * math.log2(p) for p in probs if p > 0))


def lz_compression_ratio(mat: np.ndarray) -> float:
    import zlib

    data = mat.astype(np.uint8).flatten()
    compressed = zlib.compress(data.tobytes(), level=9)
    return len(compressed) / max(len(data), 1)


def mutual_information_local(mat: np.ndarray) -> float:
    rows, cols = mat.shape
    pairs: Dict[Tuple[int, int], int] = {}
    total = 0
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                k = (int(mat[r, c]), int(mat[r, c + 1]))
                pairs[k] = pairs.get(k, 0) + 1
                total += 1
            if r + 1 < rows:
                k = (int(mat[r, c]), int(mat[r + 1, c]))
                pairs[k] = pairs.get(k, 0) + 1
                total += 1
    if total == 0:
        return 0.0
    p_xy = {k: v / total for k, v in pairs.items()}
    p_x = {0: 0.0, 1: 0.0}
    p_y = {0: 0.0, 1: 0.0}
    for (x, y), p in p_xy.items():
        p_x[x] += p
        p_y[y] += p
    mi = 0.0
    for (x, y), p in p_xy.items():
        if p_x[x] > 0 and p_y[y] > 0 and p > 0:
            mi += p * math.log2(p / (p_x[x] * p_y[y]) + 1e-12)
    return mi


def fft_anisotropy_and_radial(mat: np.ndarray) -> Tuple[float, float]:
    spectrum = np.fft.fft2(mat.astype(np.float32))
    mag = np.abs(spectrum)
    freq = np.fft.fftfreq(mat.shape[0])
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    energy_x = float((np.abs(fx) * mag).sum())
    energy_y = float((np.abs(fy) * mag).sum())
    anisotropy = (energy_x - energy_y) / (energy_x + energy_y + 1e-12)
    # radial profile
    mag_shifted = np.fft.fftshift(mag)
    rows, cols = mat.shape
    cy, cx = rows // 2, cols // 2
    y = np.arange(rows) - cy
    x = np.arange(cols) - cx
    xv, yv = np.meshgrid(x, y)
    radius = np.sqrt(xv**2 + yv**2).astype(int)
    max_r = radius.max()
    radial = np.zeros(max_r + 1, dtype=np.float32)
    counts = np.zeros(max_r + 1, dtype=np.float32)
    for r in range(max_r + 1):
        mask = radius == r
        counts[r] = mask.sum()
        if counts[r] > 0:
            radial[r] = float(mag_shifted[mask].mean())
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
                        if 0 <= ny < rows and 0 <= nx < cols:
                            if mat[ny, nx] == 1 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
                total_area += area
    mean_area = total_area / count if count else 0.0
    return count, mean_area


def compute_global_metrics(mat: np.ndarray) -> Dict[str, float]:
    density = float(mat.mean())
    entropy = binary_entropy(mat)
    perm_ent = permutation_entropy(mat)
    lz_ratio = lz_compression_ratio(mat)
    mi = mutual_information_local(mat)
    anisotropy, radial_energy = fft_anisotropy_and_radial(mat)
    components, mean_area = connected_components(mat)
    return {
        "density": density,
        "percent_black": density * 100.0,
        "percent_white": (1.0 - density) * 100.0,
        "binary_entropy": entropy,
        "permutation_entropy": perm_ent,
        "lz_ratio": lz_ratio,
        "mutual_info_local": mi,
        "fft_anisotropy": anisotropy,
        "fft_radial_energy": radial_energy,
        "components": float(components),
        "mean_component_area": mean_area,
    }


def _spectral_entropy(patch: np.ndarray) -> float:
    spectrum = np.fft.fft2(patch.astype(np.float32))
    mag = np.abs(spectrum).reshape(-1)
    mag = mag / (mag.sum() + 1e-9)
    mag = mag[mag > 0]
    return float(-(mag * np.log(mag + 1e-9)).sum())


def _spectral_anisotropy(patch: np.ndarray) -> float:
    spectrum = np.fft.fft2(patch.astype(np.float32))
    mag = np.abs(spectrum)
    freq = np.fft.fftfreq(patch.shape[0])
    fy, fx = np.meshgrid(freq, freq, indexing="ij")
    energy_x = float((np.abs(fx) * mag).sum())
    energy_y = float((np.abs(fy) * mag).sum())
    return (energy_x - energy_y) / (energy_x + energy_y + 1e-9)


def _edge_density(patch: np.ndarray) -> float:
    horiz = np.count_nonzero(patch[:, 1:] != patch[:, :-1])
    vert = np.count_nonzero(patch[1:, :] != patch[:-1, :])
    return float((horiz + vert) / (2 * patch.shape[0] * (patch.shape[1] - 1) + 1e-9))


def _autocorr(patch: np.ndarray, axis: int) -> float:
    arr = patch.astype(np.float32)
    mean = arr.mean()
    arr -= mean
    shifted = np.roll(arr, -1, axis=axis)
    num = float((arr * shifted).sum())
    den = float((arr * arr).sum()) + 1e-9
    return num / den


def _morans_i(patch: np.ndarray) -> float:
    arr = patch.astype(np.float32)
    mean = arr.mean()
    diff = arr - mean
    total = 0.0
    w = 0.0
    rows, cols = patch.shape
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                total += diff[r, c] * diff[r + 1, c]
                w += 1
            if c + 1 < cols:
                total += diff[r, c] * diff[r, c + 1]
                w += 1
    den = float((diff * diff).sum()) + 1e-9
    n = rows * cols
    return (n / (w + 1e-9)) * (total / den)


@lru_cache(maxsize=1)
def patch_coordinates(patch_size: int, grid_size: int) -> List[Tuple[float, float]]:
    coords = []
    steps = grid_size // patch_size
    for r in range(steps):
        for c in range(steps):
            coords.append(((r + 0.5) / steps, (c + 0.5) / steps))
    return coords


def compute_patch_features(mat: np.ndarray, patch_size: int = 16, variant: str = "spectral") -> np.ndarray:
    if variant not in SUPPORTED_PATCH_VARIANTS:
        raise ValueError(f"variant must be one of {SUPPORTED_PATCH_VARIANTS}")
    rows, cols = mat.shape
    assert rows % patch_size == 0 and cols % patch_size == 0
    coords = patch_coordinates(patch_size, rows)
    features: List[List[float]] = []
    idx = 0
    for r in range(0, rows, patch_size):
        for c in range(0, cols, patch_size):
            patch = mat[r : r + patch_size, c : c + patch_size]
            density = float(patch.mean())
            coord_r, coord_c = coords[idx]
            idx += 1
            if variant == "spectral":
                features.append(
                    [
                        density,
                        _spectral_entropy(patch),
                        _spectral_anisotropy(patch),
                        _edge_density(patch),
                        coord_r,
                        coord_c,
                    ]
                )
            elif variant == "correlation":
                features.append(
                    [
                        density,
                        _autocorr(patch, axis=1),
                        _autocorr(patch, axis=0),
                        _morans_i(patch),
                        _edge_density(patch),
                        coord_r,
                        coord_c,
                    ]
                )
            else:  # hybrid
                features.append(
                    [
                        density,
                        _spectral_entropy(patch),
                        _spectral_anisotropy(patch),
                        _autocorr(patch, axis=0),
                        _autocorr(patch, axis=1),
                        _morans_i(patch),
                        _edge_density(patch),
                        coord_r,
                        coord_c,
                    ]
                )
    return np.array(features, dtype=np.float32)


def compute_hamming(base: np.ndarray, mat: np.ndarray) -> int:
    return int(np.count_nonzero(base != mat))


def summarize_metrics(mat: np.ndarray, base: np.ndarray) -> Dict[str, float]:
    summary = compute_global_metrics(mat)
    summary["hamming_from_base"] = float(compute_hamming(base, mat))
    return summary
