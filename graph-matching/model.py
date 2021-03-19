import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MatchingGCN(nn.Module):
    def __init__(self, in_channels):
        super(MatchingGCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 32)
        self.linear = nn.Linear(32, 1)

    def forward(self, inp):
        x = self.conv1(inp[0].x, inp[0].edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, inp.edge_index)
        x = self.linear(x)
        return x
