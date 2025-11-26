import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import torch.nn as nn

class MultiTaskGNN(torch.nn.Module):
    """
    Deep Residual GAT with Input Projection and Multi-Task Heads.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5, num_layers=3):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        
        # 1. Input Projection
        # Project raw features (e.g. 8 dim) to high-dim latent space immediately.
        # This allows the GNN to work with richer representations.
        self.embedding_dim = hidden_channels * heads
        self.input_proj = nn.Linear(in_channels, self.embedding_dim)
        self.input_bn = nn.BatchNorm1d(self.embedding_dim)

        # 2. Deep GAT Layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            # We keep dimensions constant (embedding_dim) throughout the backbone
            # to allow for residual connections.
            self.convs.append(
                GATv2Conv(self.embedding_dim, hidden_channels, heads=heads, concat=True, dropout=dropout)
            )
            self.bns.append(nn.BatchNorm1d(self.embedding_dim))

        # 3. Output Heads
        self.out_ad = nn.Linear(self.embedding_dim, 1)
        self.out_pd = nn.Linear(self.embedding_dim, 1)
        self.out_ftd = nn.Linear(self.embedding_dim, 1)
        self.out_als = nn.Linear(self.embedding_dim, 1)

        # Decoder (for reconstruction)
        self.decoder = nn.Linear(self.embedding_dim, in_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # --- 1. Input Projection ---
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # --- 2. Deep GAT Loop with Residuals ---
        for i in range(self.num_layers):
            x_in = x  # Save input for residual connection
            
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual Connection: Add input to output
            x = x + x_in

        shared_embeddings = x
        
        # --- 3. Task Heads ---
        pred_ad = self.out_ad(shared_embeddings)
        pred_pd = self.out_pd(shared_embeddings)
        pred_ftd = self.out_ftd(shared_embeddings)
        pred_als = self.out_als(shared_embeddings)
        
        reconstructed_x = self.decoder(shared_embeddings)
        
        return pred_ad, pred_pd, pred_ftd, pred_als, shared_embeddings, reconstructed_x
