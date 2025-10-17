import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.train_gnn import FusionGNN  # type: ignore


def test_fusion_gnn_forward():
    model = FusionGNN(in_channels=6, global_dim=10, hidden=32, num_classes=4)
    x = torch.randn(8, 6)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    data.global_feats = torch.randn(1, 10)
    out = model(data)
    assert out.shape == (1, 4)
