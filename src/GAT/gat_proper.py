"""
Graph attention network for protein-protein interaction networks.
implements true learned attention over protein neighbors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GATConv, global_mean_pool



class gat_model(nn.Module):
    """
    graph attention network with two GAT layers.
    processes protein-level features and learns attention weights over neighbors.
    
    architecture:
        input: [num_proteins, 1] protein abundances for a patient
        gat1: 1 -> 32 with 8 attention heads -> 256
        gat2: 256 -> 32 with 1 attention head -> 32
        output: classifier layer 32 -> 1
    """
    
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1, num_heads=8, dropout=0.5):
        super(gat_model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # first gat layer: learn attention over neighbors
        # input_dim -> hidden_dim with multiple heads
        self.gat1 = GATConv(
            input_dim,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            add_self_loops=False,
            concat=True
        )
        
        # second gat layer: refine representations
        # hidden_dim*num_heads -> hidden_dim with single head
        self.gat2 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=1,
            dropout=dropout,
            add_self_loops=False,
            concat=False
        )
        
        # output classification layer
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        """
        forward pass through gat layers.
        
        args:
            x: [num_nodes, input_dim] node features
            edge_index: [2, num_edges] edge indices
            edge_weight: [num_edges] optional edge weights
            batch: [num_nodes] batch assignment for graph pooling
        
        returns:
            logits: [num_graphs, 1] patient-level predictions
        """
        
        # ensure edge_index is correct dtype
        if edge_index.dtype != torch.int64:
            edge_index = edge_index.to(torch.int64)
        
        # normalize edge weights 
        if edge_weight is not None:
            if edge_weight.dim() > 1:
                edge_weight = edge_weight.squeeze()
            min_weight = edge_weight.min()
            max_weight = edge_weight.max()
            edge_weight = (edge_weight - min_weight) / (max_weight - min_weight + 1e-8)
        
        # first gat layer with attention
        x = self.gat1(x, edge_index, edge_weight)
        x = f.elu(x)
        x = f.dropout(x, p=0.5, training=self.training)
        
        # second gat layer
        x = self.gat2(x, edge_index, edge_weight)
        x = f.elu(x)
        
        # if batch provided, aggregate to graph level
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # if single graph, take mean across all nodes
            x = x.mean(dim=0, keepdim=True)
        
        # output layer
        logits = self.classifier(x)
        
        return logits