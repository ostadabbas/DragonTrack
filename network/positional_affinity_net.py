import torch
from torch import nn

class PositionalAffinityNet(nn.Module):
    def __init__(self):
        super(PositionalAffinityNet, self).__init__()
        # Adjust the input size based on your positional edge features
        self.mlp = nn.Sequential(
            nn.Linear(4, 1),  # Assuming 4 features for positional differences
            nn.ReLU()
        )

    def forward(self, inputs):
        return self.mlp(inputs)
