import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import torch.nn as nn

class MultiTaskGNN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()

        self.dropout = dropout

        # edge_dim=1 because edge_attr has shape [num_edges, 1] (STRING weight)
        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=1
        )

        self.conv2 = GATv2Conv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            dropout=dropout,
            edge_dim=1
        )

        # one regression head per disease
        self.out_ad  = nn.Linear(hidden_channels, 1)
        self.out_pd  = nn.Linear(hidden_channels, 1)
        self.out_ftd = nn.Linear(hidden_channels, 1)
        self.out_als = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # dropout on node features
        x = F.dropout(x, p=self.dropout, training=self.training)

        # first GAT layer with edge weights
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # second GAT layer (these embeddings are what we'll cluster)
        shared_embeddings = self.conv2(x, edge_index, edge_attr=edge_attr)

        # disease-specific predictions
        pred_ad  = self.out_ad(shared_embeddings)
        pred_pd  = self.out_pd(shared_embeddings)
        pred_ftd = self.out_ftd(shared_embeddings)
        pred_als = self.out_als(shared_embeddings)

        # embeddings go to UMAP + HDBSCAN for module discovery
        return pred_ad, pred_pd, pred_ftd, pred_als, shared_embeddings
