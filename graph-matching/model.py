import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing


class Encoder(MessagePassing):
    def __init__(self):
        super().__init__()
        in_node_channels = 2
        out_node_channels = 2
        hidden_node_features = 16
        self.node_mlp = MLP(in_node_channels, out_node_channels, hidden_node_features)

        in_edge_channels = 2
        out_edge_channels = 2
        hidden_edge_channels = 16
        self.edge_mlp = MLP(in_edge_channels, out_edge_channels, hidden_edge_channels)

    def forward(self, data):
        # TODO steps
        # take mlp of nodes and do message passing from current layer to next

        pass


class MLP(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, hidden_features: int
    ) -> None:
        self.layer1 = nn.Linear(in_channels, hidden_features)
        self.layer2 = nn.Linear(hidden_features, out_channels)

    def forward(self, data):
        x = self.layer1(data)
        x = F.relu(x)

        x = self.layer2(x)
        x = F.relu(x)
        return x


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
