import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.generate_dataset import metric_bins  # type: ignore


def test_metric_bins_simple(tmp_path):
    metrics = {"binary_entropy": 0.8, "fft_anisotropy": 0.3, "mutual_info_local": 0.05, "hamming_from_base": 12000}
    bin_config = {
        "binary_entropy": [0.4, 0.7, 0.9, 1.0],
        "fft_anisotropy": [-0.5, -0.2, 0.2, 0.5, 0.8],
        "mutual_info_local": [0.02, 0.05, 0.1, 0.2],
        "hamming_from_base": [5000, 15000, 25000, 40000],
    }
    bin_id = metric_bins(metrics, bin_config)
    assert len(bin_id) == len(bin_config)
