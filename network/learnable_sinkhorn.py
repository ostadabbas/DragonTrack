import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableSinkhorn(nn.Module):
    def __init__(self, max_iter=10):
        super(LearnableSinkhorn, self).__init__()
        self.max_iter = max_iter
        # Learnable parameters to adjust row and column sums
        self.row_weight = nn.Parameter(torch.ones(1))
        self.col_weight = nn.Parameter(torch.ones(1))

    def forward(self, matrix):
        for _ in range(self.max_iter):
            # Row normalization
            row_sum = torch.sum(matrix, dim=1, keepdim=True)
            matrix = matrix / (row_sum * self.row_weight)

            # Column normalization
            col_sum = torch.sum(matrix, dim=0, keepdim=True)
            matrix = matrix / (col_sum * self.col_weight)

        return matrix
