"""
patient graph dataset class.
creates individual patient graphs that share the same edge structure
to minimize memory usage.
"""

import torch
from torch_geometric.data import Data, Dataset


class patient_graph_dataset(Dataset):
    """
    dataset that creates individual Data objects for each patient.
    edge_index and edge_weight are stored once and referenced by all patients.
    
    attributes:
        x: [num_patients, num_proteins] protein abundance matrix
        y: [num_patients] disease labels
        edge_index: [2, num_edges] ppi network structure
        edge_weight: [num_edges] ppi confidence scores
    """
    
    def __init__(self, x, y, edge_index, edge_weight, num_proteins=6245):
        super().__init__()
        
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_proteins = num_proteins
        self.num_patients = x.shape[0]
    
    def len(self):
        """returns number of patients in dataset."""
        return self.num_patients
    
    def get(self, idx):
        """
        returns Data object for patient at index idx.
        
        args:
            idx: patient index
        
        returns:
            Data object with:
                x: [num_proteins, 1] patient's protein abundances
                edge_index: [2, num_edges] shared ppi network
                edge_weight: [num_edges] shared ppi scores
                y: disease label for this patient
        """
        
        patient_abundance = self.x[idx].unsqueeze(1)  # [num_proteins, 1]
        patient_label = self.y[idx]
        
        graph = Data(
            x=patient_abundance,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            y=patient_label
        )
        
        return graph