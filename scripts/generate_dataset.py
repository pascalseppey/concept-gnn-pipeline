#!/usr/bin/env python3
"""Compatibility wrapper to generate datasets with coverage enforcement."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_generator import main  # type: ignore


if __name__ == "__main__":
    main()
