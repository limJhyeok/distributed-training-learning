import torch.nn as nn
import torch


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(in_features, 2 * in_features)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(2 * in_features, out_features)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.out_proj(self.relu(self.in_proj(x)))
