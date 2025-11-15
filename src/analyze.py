import torch
import numpy as np
import pandas as pd
from model import MultiTaskGNN
import umap.umap_ as umap
import hdbscan  # <-- New import
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("--- Starting Model Analysis with Clustering ---")
    
    # --- 1. Load Data and Trained Model (Same as before) ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_path = 'data/processed_graph.pt'
    model_path = 'models/best_model.pt'
    data = torch.load(data_path)
    data = data.to(device)
    print("Loaded data object.")

    HIDDEN_CHANNELS = 64
    ATTENTION_HEADS = 8
    DROPOUT_RATE = 0.6
    
    model = MultiTaskGNN(in_channels=data.num_node_features, hidden_channels=HIDDEN_CHANNELS, out_channels=4, heads=ATTENTION_HEADS, dropout=DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Successfully loaded trained model weights.")

    # --- 2. Extract Shared Embeddings (Same as before) ---
    with torch.no_grad():
        _, _, _, _, shared_embeddings = model(data)
    embeddings_np = shared_embeddings.cpu().numpy()
    print(f"Extracted shared embeddings. Shape: {embeddings_np.shape}")

    # --- 3. Run UMAP (Same as before) ---
    print("\nRunning UMAP for dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings_np)
    print("UMAP complete.")

    # --- 4. NEW: Run HDBSCAN Clustering ---
    print("\nRunning HDBSCAN to find clusters...")
    # min_cluster_size is the most important parameter. Let's start with 15.
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(embedding_2d)
    num_clusters = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Found {num_clusters} clusters. (Label -1 represents noise points)")
    
    # --- 5. Create DataFrame and Save Plot ---
    plot_df = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
    plot_df['GeneSymbol'] = data.gene_symbols
    plot_df['Cluster'] = cluster_labels
    # Make cluster labels more plot-friendly (strings)
    plot_df['Cluster'] = 'Cluster ' + plot_df['Cluster'].astype(str)
    plot_df.loc[plot_df['Cluster'] == 'Cluster -1', 'Cluster'] = 'Noise'

    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=plot_df,
        x='UMAP1',
        y='UMAP2',
        hue='Cluster', # <-- Color points by their cluster label
        palette=sns.color_palette("hsv", n_colors=num_clusters + 1),
        alpha=0.7,
        s=15
    )
    plt.title('UMAP Projection with HDBSCAN Clusters', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the new plot
    output_plot_path = 'plots/protein_embeddings_clustered_umap.png'
    plt.savefig(output_plot_path, dpi=300)
    print(f"\nClustered plot saved to '{output_plot_path}'")
    plt.show()

    # --- 6. NEW: Export Cluster Protein Lists ---
    output_clusters_path = 'data/protein_clusters.csv'
    # Filter out noise points for the export
    export_df = plot_df[plot_df['Cluster'] != 'Noise'][['GeneSymbol', 'Cluster']]
    export_df.to_csv(output_clusters_path, index=False)
    print(f"Protein lists for each cluster saved to '{output_clusters_path}'")


if __name__ == '__main__':
    main()
