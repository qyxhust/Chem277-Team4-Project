import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from umap import UMAP

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.model import MultiTaskGNN

# Configuration
MODEL_PATH = 'models/best_model.pt'
DATA_PATH = 'data/02-preprocessed/processed_graph.pt'
HIDDEN_CHANNELS = 64
HEADS = 8
DROPOUT = 0.6
N_CLUSTERS = 5 # 假设聚成5类，您可以根据需要调整

def plot_umap(embeddings, labels, title, save_path):
    """Helper function to plot UMAP embeddings."""
    print(f"Running UMAP for {title}...")
    reducer = UMAP(n_neighbors=30, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding_2d[:, 0], 
        embedding_2d[:, 1], 
        c=labels, 
        cmap='viridis', 
        s=10, 
        alpha=0.6
    )
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(title)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

def main():
    print("Starting Cluster Analysis...")
    
    # 1. Load Data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(DATA_PATH).to(device)
    
    # 2. Load Model
    model = MultiTaskGNN(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=data.num_node_features,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("Error: Model file not found! Please train the model first.")
        return

    # 3. Extract Features
    model.eval()
    with torch.no_grad():
        # Raw Features (Input)
        raw_features = data.x.cpu().numpy()
        
        # GNN Embeddings (Output)
        # 注意：这里我们用的是模型输出的重构特征。
        # 如果您想要隐层特征，需要修改 model.py 返回中间层，或者直接用最后一层
        gnn_embeddings = model(data).cpu().numpy()

    print(f"\nRaw Features Shape: {raw_features.shape}")
    print(f"GNN Embeddings Shape: {gnn_embeddings.shape}")

    # 4. Clustering & Evaluation
    print("\n--- Performing Clustering (K-Means) ---")
    
    # A. Raw Features Clustering
    kmeans_raw = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    labels_raw = kmeans_raw.fit_predict(raw_features)
    score_raw = silhouette_score(raw_features, labels_raw)
    print(f"Raw Features Silhouette Score: {score_raw:.4f}")
    
    # B. GNN Embeddings Clustering
    kmeans_gnn = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    labels_gnn = kmeans_gnn.fit_predict(gnn_embeddings)
    score_gnn = silhouette_score(gnn_embeddings, labels_gnn)
    print(f"GNN Embeddings Silhouette Score: {score_gnn:.4f}")
    
    print(f"\nImprovement: {(score_gnn - score_raw) / score_raw * 100:.2f}%")

    # 5. Visualization
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    plot_umap(raw_features, labels_raw, 
              f'Clustering on Raw Features (Silhouette: {score_raw:.2f})', 
              'plots/cluster_raw.png')
              
    plot_umap(gnn_embeddings, labels_gnn, 
              f'Clustering on GNN Embeddings (Silhouette: {score_gnn:.2f})', 
              'plots/cluster_gnn.png')

    print("\nAnalysis Complete. Check the 'plots' directory.")

if __name__ == "__main__":
    main()