# models/gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

class GraphSAGEEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        x = F.relu(self.lin1(x))
        return torch.sigmoid(self.lin2(x))

class GNNLinkPredictor(nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.encoder = GraphSAGEEncoder(hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr='sum')
        self.predictor = LinkPredictor(hidden_channels)

    def forward(self, data, edge_label_index):
        x_dict = self.encoder(data.x_dict, data.edge_index_dict)
        investor_emb = x_dict['investor'][edge_label_index[0]]
        fund_emb = x_dict['fund'][edge_label_index[1]]
        return self.predictor(investor_emb, fund_emb)
