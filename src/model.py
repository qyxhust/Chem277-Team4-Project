import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import torch.nn as nn

class MultiTaskGNN(torch.nn.Module):
    """
    A Graph Attention Network (GAT) for predicting multi-task regression targets.
    
    This model uses two GAT layers to learn node embeddings from the graph structure
    and initial node features. It then uses separate linear output heads to make a
    regression prediction for each of the four diseases.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        """
        Args:
            in_channels (int): Number of input features for each node.
            hidden_channels (int): Number of hidden units in the GAT layers.
            out_channels (int): Must be 4, for the four disease prediction tasks.
            heads (int): Number of attention heads in the first GAT layer.
            dropout (float): Dropout rate for regularization.
        """
        super().__init__()
        
        self.dropout = dropout

        # First GAT layer: learns initial embeddings from input features.
        # Multi-head attention is used here for more robust feature learning.
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        
        # Second GAT layer: aggregates information from the first layer.
        # The input channels must be hidden_channels * heads from the previous layer.
        # We use a single head here to get the final shared embedding.
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat = False, dropout=dropout)

        # Output layers (prediction heads)
        # We create a separate linear layer for each of the 4 regression tasks.
        # This allows the model to learn a specific final transformation for each disease.
        # self.out_ad = nn.Linear(hidden_channels, 1)
        # self.out_pd = nn.Linear(hidden_channels, 1)
        # self.out_ftd = nn.Linear(hidden_channels, 1)
        # self.out_als = nn.Linear(hidden_channels, 1)

    def forward(self, data):
        """
        The forward pass of the model.
        
        Args:
            data (torch_geometric.data.Data): The input graph data object.
        
        Returns:
            tuple: A tuple containing the predictions for each of the 4 tasks
                and the final shared node embeddings for analysis.
        """
        x, edge_index = data.x, data.edge_index
        
        # Apply dropout to the input features for regularization
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pass through the first GAT layer, followed by an ELU activation function
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pass through the second GAT layer to get the shared embeddings
        x = self.conv2(x, edge_index)
        
        # Generate predictions from each disease-specific output head
        # pred_ad = self.out_ad(shared_embeddings)
        # pred_pd = self.out_pd(shared_embeddings)
        # pred_ftd = self.out_ftd(shared_embeddings)
        # pred_als = self.out_als(shared_embeddings)
        
        # return pred_ad, pred_pd, pred_ftd, pred_als, shared_embeddings
        return x
