#!/usr/bin/env python3
"""Analyse stored metrics and plot distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_metrics(path: Path, limit: int | None = None) -> pd.DataFrame:
    records = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            rec = json.loads(line)
            metrics = rec.get("metrics", rec)
            metrics["sequence_id"] = rec.get("sequence_id", idx)
            records.append(metrics)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse metrics distributions from dataset")
    parser.add_argument("--input", type=Path, required=True, help="JSONL or CSV")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/plots"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.input.suffix == ".csv":
        df = pd.read_csv(args.input)
    else:
        df = load_metrics(args.input, args.limit)

    print(df.describe())

    for column in ["fft_anisotropy", "mutual_info_local", "hamming_from_base", "binary_entropy"]:
        if column in df.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[column], bins=40)
            plt.title(f"Distribution de {column}")
            plt.tight_layout()
            plt.savefig(args.output_dir / f"hist_{column}.png")
            plt.close()


if __name__ == "__main__":
    main()
