import torch.nn
import torch.nn.functional as F
import torch_geometric.nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GCN, GAT

class MutliTaskGNN(torch.nn.Module):
    def __init__(self,
                diseases, 
                in_channels, 
                hidden_channels, 
                n_gcn_layers, 
                shared_channels=128, 
                dropout=0.5):
        super().__init__()
        self.diseases = diseases

        # Disease-specific GAT encoders
        self.encoders = torch.nn.ModuleDict()
        for d in diseases:
            self.encoders[d] = \
            GAT(in_channels=in_channels, hidden_channels=hidden_channels, \
                num_layers=n_gcn_layers, out_channels=hidden_channels)

        # Shared representation layer
        self.shared_layer = Linear(hidden_channels * len(diseases), shared_channels)
        self.dropout = torch.nn.Dropout(dropout)

        # Disease-specific task heads
        self.sig_head = torch.nn.ModuleDict()
        self.role_head = torch.nn.ModuleDict()
        self.abundance_head = torch.nn.ModuleDict()
        for d in diseases:
            # Task A: Disease significance (Classification)
            self.sig_head[d] = torch.nn.Sequential(
                torch.nn.Linear(shared_channels, shared_channels//2),
                torch.nn.ReLU(),
                torch.nn.Linear(shared_channels//2, 1)
            )

            # Task B: Mechanistic role (Classification)
            self.role_head[d] = torch.nn.Sequential(
                torch.nn.Linear(shared_channels, shared_channels//2),
                torch.nn.ReLU(),
                torch.nn.Linear(shared_channels//2, 2)
            )

            # Task C: Disease-specific abundance (Regression)
            self.abundance_head[d] = torch.nn.Sequential(
                torch.nn.Linear(shared_channels, shared_channels//2),
                torch.nn.ReLU(),
                torch.nn.Linear(shared_channels//2, 1)
            )


    def forward(self, x, edge_index, disease, edge_weight=None):
        # Get disease encodings
        h = [self.encoders[d](x, edge_index, edge_weight) for d in self.diseases]
        h_shared = torch.cat(h, dim=-1)
        h_shared = F.elu(self.shared_layer(h_shared))
        h_shared = self.dropout(h_shared)

        # Tasks forward
        y_sig = self.sig_head[disease](h_shared)
        y_roles = self.role_head[disease](h_shared)
        y_abundance = self.abundance_head[disease](h_shared)

        return {'sig':y_sig, 'role':y_roles, 'abundance':y_abundance}
        