"""Utility helpers to enumerate effect chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.bijective_pipeline import Command


@dataclass
class EffectConfig:
    roll_shifts: Iterable[int]
    xor_seeds: Iterable[int]
    perm_rows_seeds: Iterable[int]
    perm_cols_seeds: Iterable[int]
    block_sizes: Iterable[int]
    block_seeds: Iterable[int]
    max_depth: int = 3


def serialize(cmd: Command) -> Dict[str, int]:
    entry = {"name": cmd.name}
    entry.update(cmd.params)
    return entry


def generate_sequence_pool(cfg: EffectConfig) -> List[List[Dict[str, int]]]:
    pool: List[List[Dict[str, int]]] = [[]]

    roll_params = [
        serialize(Command("roll", {"axis": axis, "shift": shift}))
        for axis in (0, 1)
        for shift in cfg.roll_shifts
    ]
    xor_params = [serialize(Command("xor_mask", {"seed": seed})) for seed in cfg.xor_seeds]
    perm_rows = [serialize(Command("permute_rows", {"seed": seed})) for seed in cfg.perm_rows_seeds]
    perm_cols = [serialize(Command("permute_cols", {"seed": seed})) for seed in cfg.perm_cols_seeds]
    block_params = [
        serialize(Command("block_shuffle", {"block": block, "seed": seed}))
        for block in cfg.block_sizes
        for seed in cfg.block_seeds
    ]

    families = [roll_params, xor_params, perm_rows, perm_cols, block_params]

    pool.extend([[cmd] for family in families for cmd in family])

    def product(a: List[Dict[str, int]], b: List[Dict[str, int]]) -> Iterable[List[Dict[str, int]]]:
        for cmd1 in a:
            for cmd2 in b:
                yield [cmd1, cmd2]

    for family_a in families:
        for family_b in families:
            pool.extend(product(family_a, family_b))

    if cfg.max_depth >= 3:
        for cmd1 in roll_params[:4]:
            for cmd2 in perm_rows[:4]:
                for cmd3 in block_params[:4]:
                    pool.append([cmd1, cmd2, cmd3])
        for cmd1 in xor_params[:4]:
            for cmd2 in perm_cols[:4]:
                for cmd3 in block_params[:4]:
                    pool.append([cmd1, cmd2, cmd3])
    if cfg.max_depth >= 4:
        for cmd1 in xor_params[:3]:
            for cmd2 in roll_params[:3]:
                for cmd3 in perm_rows[:3]:
                    for cmd4 in block_params[:3]:
                        pool.append([cmd1, cmd2, cmd3, cmd4])
    return pool
