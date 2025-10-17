import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import metrics  # type: ignore

def test_binary_entropy_identity():
    mat = np.zeros((256, 256), dtype=np.uint8)
    assert metrics.binary_entropy(mat) == 0.0


def test_hamming():
    base = np.zeros((256, 256), dtype=np.uint8)
    other = base.copy()
    other[0, 0] = 1
    assert metrics.compute_hamming(base, other) == 1
