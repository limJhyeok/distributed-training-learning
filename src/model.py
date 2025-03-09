import torch.nn as nn
import torch


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc = nn.Linear(5, 5, bias=False)

    def forward(self, x: torch.tensor):
        return self.fc(x)
