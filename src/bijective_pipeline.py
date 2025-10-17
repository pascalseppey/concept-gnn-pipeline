"""Bijective pipeline with explicit command logging and inversion helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np


Matrix = np.ndarray


EffectFn = Callable[[Matrix, Dict[str, int]], Matrix]


def _roll(mat: Matrix, params: Dict[str, int]) -> Matrix:
    axis = params["axis"]
    shift = params["shift"]
    return np.roll(mat, shift=shift, axis=axis)


def _xor_mask(mat: Matrix, params: Dict[str, int]) -> Matrix:
    seed = params["seed"]
    rng = np.random.default_rng(seed)
    mask = rng.integers(0, 2, size=mat.shape, dtype=np.uint8)
    return np.bitwise_xor(mat, mask)


def _permute_rows(mat: Matrix, params: Dict[str, int]) -> Matrix:
    seed = params["seed"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(mat.shape[0])
    return mat[order]


def _permute_rows_inv(mat: Matrix, params: Dict[str, int]) -> Matrix:
    seed = params["seed"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(mat.shape[0])
    inv = np.argsort(order)
    return mat[inv]


def _permute_cols(mat: Matrix, params: Dict[str, int]) -> Matrix:
    seed = params["seed"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(mat.shape[1])
    return mat[:, order]


def _permute_cols_inv(mat: Matrix, params: Dict[str, int]) -> Matrix:
    seed = params["seed"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(mat.shape[1])
    inv = np.argsort(order)
    return mat[:, inv]


def _block_shuffle(mat: Matrix, params: Dict[str, int]) -> Matrix:
    block = params["block"]
    seed = params["seed"]
    h, w = mat.shape
    if h % block or w % block:
        raise ValueError("Block size must divide matrix dimensions")
    rng = np.random.default_rng(seed)
    reshaped = mat.reshape(h // block, block, w // block, block)
    flattened = reshaped.swapaxes(1, 2).reshape(-1, block, block)
    order = rng.permutation(flattened.shape[0])
    shuffled = flattened[order]
    return shuffled.reshape(h // block, w // block, block, block).swapaxes(1, 2).reshape(h, w)


def _block_shuffle_inv(mat: Matrix, params: Dict[str, int]) -> Matrix:
    block = params["block"]
    seed = params["seed"]
    h, w = mat.shape
    rng = np.random.default_rng(seed)
    if h % block or w % block:
        raise ValueError("Block size must divide matrix dimensions")
    reshaped = mat.reshape(h // block, block, w // block, block)
    flattened = reshaped.swapaxes(1, 2).reshape(-1, block, block)
    order = rng.permutation(flattened.shape[0])
    inv = np.argsort(order)
    restored = flattened[inv]
    return restored.reshape(h // block, w // block, block, block).swapaxes(1, 2).reshape(h, w)


EFFECTS: Dict[str, Tuple[EffectFn, EffectFn]] = {
    "roll": (_roll, lambda mat, params: _roll(mat, {"axis": params["axis"], "shift": -params["shift"]})),
    "xor_mask": (_xor_mask, _xor_mask),
    "permute_rows": (_permute_rows, _permute_rows_inv),
    "permute_cols": (_permute_cols, _permute_cols_inv),
    "block_shuffle": (_block_shuffle, _block_shuffle_inv),
}


@dataclass
class Command:
    name: str
    params: Dict[str, int]


@dataclass
class PipelineMetadata:
    shape: Tuple[int, int]
    commands: List[Command] = field(default_factory=list)


class BijectivePipeline:
    """Accumulates bijective effects and logs inverse commands."""

    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape
        self.commands: List[Command] = []

    def apply(self, mat: Matrix, command: Command) -> Matrix:
        fn, _ = EFFECTS[command.name]
        result = fn(mat, command.params)
        self.commands.append(command)
        return result

    def run(self, mat: Matrix) -> Tuple[Matrix, PipelineMetadata]:
        if mat.shape != self.shape:
            raise ValueError(f"Expected matrix shape {self.shape}, got {mat.shape}")
        out = mat.copy()
        for cmd in self.commands:
            fn, _ = EFFECTS[cmd.name]
            out = fn(out, cmd.params)
        metadata = PipelineMetadata(shape=self.shape, commands=list(self.commands))
        return out, metadata

    def clear(self) -> None:
        self.commands.clear()

    @staticmethod
    def invert(noise: Matrix, metadata: PipelineMetadata) -> Matrix:
        mat = noise.copy()
        for cmd in reversed(metadata.commands):
            _, inv_fn = EFFECTS[cmd.name]
            mat = inv_fn(mat, cmd.params)
        return mat


def serialize_metadata(metadata: PipelineMetadata) -> Dict[str, object]:
    return {
        "shape": metadata.shape,
        "commands": [
            {"name": cmd.name, "params": cmd.params} for cmd in metadata.commands
        ],
    }


def deserialize_metadata(payload: Dict[str, object]) -> PipelineMetadata:
    shape = tuple(payload["shape"])  # type: ignore[arg-type]
    commands = [Command(name=cmd["name"], params=dict(cmd["params"])) for cmd in payload["commands"]]
    return PipelineMetadata(shape=shape, commands=commands)
