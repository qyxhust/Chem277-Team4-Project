import sys
import os

# Get the absolute path of the directory containing this script (entrypoint)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project's root directory (one level up)
project_root = os.path.dirname(script_dir)
# Add the project root to Python's path
sys.path.insert(0, project_root)


import torch
import numpy as np
import pandas as pd
from src.model import MultiTaskGNN 
import umap.umap_ as umap
import hdbscan
from sklearn.metrics import silhouette_score 
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    print("--- Clustering Optimization and Profiling ---")
    
    # --- 1. Load Data and Model  ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data = torch.load('data/02-preprocessed/processed_graph.pt')
    data = data.to(device)
    model_path = 'models/best_model.pt'
    
    model = MultiTaskGNN(in_channels=data.num_node_features, hidden_channels=64, out_channels=4, heads=8, dropout=0.6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Loaded data and trained model.")

    # --- 2. Extract Embeddings and run UMAP ---
    with torch.no_grad():
        _, _, _, _, shared_embeddings = model(data)
    embeddings_np = shared_embeddings.cpu().numpy()
    
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings_np)

    # --- 3. Technical Benchmarking for Clustering Quality ---
    print("\n--- Benchmarking Clustering Quality (Silhouette Score) ---")
    
    # We will test a range of min_cluster_size values
    min_cluster_sizes_to_test = [15, 25, 35, 50, 75, 100]
    best_score = -1
    best_labels = None
    best_min_size = None

    for min_size in min_cluster_sizes_to_test:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        labels = clusterer.fit_predict(embedding_2d)
        
        # We can only calculate silhouette score if we have at least 2 clusters
        num_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        if num_clusters >= 2:
            # We must exclude noise points for a meaningful score
            clustered_points_mask = labels != -1
            score = silhouette_score(embedding_2d[clustered_points_mask], labels[clustered_points_mask])
            print(f"min_cluster_size = {min_size:3d} -> Found {num_clusters:2d} clusters with Silhouette Score = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_labels = labels
                best_min_size = min_size
        else:
            print(f"min_cluster_size = {min_size:3d} -> Found {num_clusters:2d} clusters. Not enough to score.")

    print(f"\n---> Best result found with min_cluster_size = {best_min_size}. Using these clusters for analysis.")
    cluster_labels = best_labels # Use the best labels for the rest of the script
    
    # --- 4. Generate Plots and Profiles (using the BEST clusters) ---
    num_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    results_df = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
    results_df['GeneSymbol'] = data.gene_symbols
    results_df['Cluster'] = ['Cluster ' + str(label) for label in cluster_labels]
    results_df.loc[results_df['Cluster'] == 'Cluster -1', 'Cluster'] = 'Noise'

    # ... generate colored UMAP ...
    # ... generate cluster profile heatmap ...
    # ... export protein_clusters.csv ...

    # --- Plotting and export code ---
    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=results_df, x='UMAP1', y='UMAP2', hue='Cluster', palette=sns.color_palette("hsv", n_colors=num_clusters + 1), alpha=0.7, s=15)
    plt.title(f'UMAP Projection with Optimal HDBSCAN Clusters (min_size={best_min_size})', fontsize=16)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('plots/protein_embeddings_clustered_umap.png', dpi=300)
    print("\nOptimal clustered UMAP plot saved.")
    plt.show()

    features_df = pd.read_csv('data/02-preprocessed/protein_features.csv', index_col='GeneSymbol')
    full_profile_df = features_df.join(results_df.set_index('GeneSymbol')['Cluster'])
    beta_cols = ['AD_beta', 'PD_beta', 'FTD_beta', 'ALS_beta']
    cluster_profile = full_profile_df.groupby('Cluster')[beta_cols].mean().drop('Noise', errors='ignore')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cluster_profile, cmap='coolwarm', annot=True, fmt=".3f", center=0)
    plt.title('Average Protein Abundance Profile by Cluster', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/cluster_profile_heatmap.png', dpi=300)
    print("Cluster profile heatmap saved.")
    plt.show()
    
    export_df = results_df[results_df['Cluster'] != 'Noise'][['GeneSymbol', 'Cluster']]
    export_df.to_csv('data/04-predictions/protein_clusters.csv', index=False)
    print("Protein lists for pathway analysis saved to 'data/04-predictions/protein_clusters.csv'")

if __name__ == '__main__':
    main()
