import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import torch.nn as nn

class MultiTaskGNN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()

        self.dropout = dropout

        # first GAT layer, using multi-head attention
        # heads parameter lets us use multiple attention mechanisms
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)

        # second layer, input size needs to be hidden*heads because of concat
    
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)

        # separate output layer for each disease
       
        self.out_ad = nn.Linear(hidden_channels, 1)
        self.out_pd = nn.Linear(hidden_channels, 1)
        self.out_ftd = nn.Linear(hidden_channels, 1)
        self.out_als = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # dropout on input
        x = F.dropout(x, p=self.dropout, training=self.training)

        # first GAT layer + activation
        x = self.conv1(x, edge_index)
        x = F.elu(x)  

        x = F.dropout(x, p=self.dropout, training=self.training)

        # second layer gives us the embeddings we'll use
        shared_embeddings = self.conv2(x, edge_index)

        # get predictions for each disease from the shared embeddings
        pred_ad = self.out_ad(shared_embeddings)
        pred_pd = self.out_pd(shared_embeddings)
        pred_ftd = self.out_ftd(shared_embeddings)
        pred_als = self.out_als(shared_embeddings)

        # return all predictions plus embeddings (need embeddings for clustering later)
        return pred_ad, pred_pd, pred_ftd, pred_als, shared_embeddings