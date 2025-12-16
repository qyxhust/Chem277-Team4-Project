"""
Restructure the data into a proper format for GAT:
Create 1180 separate graphs (one per patient), each with:
- 6245 protein nodes
- Same protein-protein interaction edges for all patients
- That patient's protein abundance values as node features
- That patient's disease label
"""

import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from pathlib import Path


def restructure_data_to_graphs(data_path: str):
    """
    Restructure data from:
    - x: [1180 patients, 6245 proteins]
    - y: [1180 patients]
    - edge_index: [2, 1286029] with indices 0-6244
    
    To:
    - 1180 Data objects, each with:
      - x: [6245, 1] (that patient's protein abundances)
      - edge_index: [2, num_valid_edges]
      - y: scalar (that patient's disease label)
    """
    
    print("Loading original data")
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    
    x_patients = data.x  # [1180, 6245]
    y_patients = data.y  # [1180]
    edge_index = data.edge_index  # [2, 1286029]
    edge_weight = data.edge_weight  # [1286029, 1]
    
    num_patients = x_patients.shape[0]
    num_proteins = x_patients.shape[1]
    
    print(f"\nOriginal structure:")
    print(f"  Patients: {num_patients}")
    print(f"  Proteins: {num_proteins}")
    print(f"  Total edges: {edge_index.shape[1]}")
    
    #Fix edges - only keep those with valid protein indices
    print(f"\nFixing edges")
    valid_edges = (edge_index[0] < num_proteins) & (edge_index[1] < num_proteins)
    edge_index_fixed = edge_index[:, valid_edges].clone()
    
    if edge_weight.dim() > 1:
        edge_weight_fixed = edge_weight[valid_edges].squeeze().clone()
    else:
        edge_weight_fixed = edge_weight[valid_edges].clone()
    
    print(f"  Valid edges: {edge_index_fixed.shape[1]}")
    print(f"  Removed: {(~valid_edges).sum().item()} invalid edges")
    
    #Create individual patient graphs
    print(f"\nCreating {num_patients} patient graphs")
    
    patient_graphs = []
    
    for patient_id in range(num_patients):
        # Get this patient's protein abundances
        patient_abundances = x_patients[patient_id]  # [6245]
        patient_label = y_patients[patient_id]  # scalar
        
        # Create a Data object for this patient
        patient_graph = Data(
            x=patient_abundances.unsqueeze(1).float(),  # [6245, 1]
            edge_index=edge_index_fixed.long(),  # [2, num_edges]
            edge_attr=edge_weight_fixed.float().unsqueeze(1),  # [num_edges, 1]
            y=patient_label.float(),  # scalar
            patient_id=patient_id
        )
        
        patient_graphs.append(patient_graph)
        
        if (patient_id + 1) % 200 == 0:
            print(f" Created {patient_id + 1}/{num_patients} graphs")
    
    print(f"Created {len(patient_graphs)} patient graphs")
    print(f"\nExample patient graph 0:")
    print(f"  x (protein abundances): {patient_graphs[0].x.shape}")
    print(f"  edge_index: {patient_graphs[0].edge_index.shape}")
    print(f"  edge_attr: {patient_graphs[0].edge_attr.shape}")
    print(f"  y (disease label): {patient_graphs[0].y.item()}")
    
    #Save individual graphs
    output_dir = Path("patient_graphs")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving patient graphs to {output_dir}/...")
    for patient_id, graph in enumerate(patient_graphs):
        torch.save(graph, output_dir / f"patient_{patient_id:04d}.pt")
        if (patient_id + 1) % 200 == 0:
            print(f"Saved {patient_id + 1}/{len(patient_graphs)} graphs")
    
    #Create a single file with all graphs and splits
    print(f"\nCreating combined dataset file")
    
    # Create train/val/test splits (same as before for consistency)
    np.random.seed(42)
    torch.manual_seed(42)
    
    N = len(patient_graphs)
    split_labels = np.random.choice([0, 1, 2], N, p=[0.7, 0.15, 0.15])
    
    train_indices = np.where(split_labels == 0)[0]
    val_indices = np.where(split_labels == 1)[0]
    test_indices = np.where(split_labels == 2)[0]
    
    dataset_info = {
        'graphs': patient_graphs,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'num_patients': num_patients,
        'num_proteins': num_proteins,
        'num_edges': edge_index_fixed.shape[1]
    }
    
    output_file = "patient_graphs_dataset.pt"
    torch.save(dataset_info, output_file)
    
    print(f"Saved dataset to {output_file}")
    print(f"\nDataset split:")
    print(f"  Train: {len(train_indices)} ({len(train_indices)/N*100:.1f}%)")
    print(f"  Val:   {len(val_indices)} ({len(val_indices)/N*100:.1f}%)")
    print(f"  Test:  {len(test_indices)} ({len(test_indices)/N*100:.1f}%)")
    
    # Print summary
    print(f"\nDataset summary:")
    labels = np.array([g.y.item() for g in patient_graphs])
    print(f"  Total patients: {len(patient_graphs)}")
    print(f"  Disease: {(labels == 1).sum()} ({(labels == 1).sum()/len(labels)*100:.1f}%)")
    print(f"  Healthy: {(labels == 0).sum()} ({(labels == 0).sum()/len(labels)*100:.1f}%)")
    
    return output_file


if __name__ == "__main__":
    data_path = "./binary_label_data.pt"
    output_file = restructure_data_to_graphs(data_path)
    print(f"\nRestructuring complete")
   