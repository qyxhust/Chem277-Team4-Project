import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Configuration
ORIGINAL_DATA_PATH = 'data/02-preprocessed/processed_graph.pt'
IMPUTED_DATA_PATH = 'results/inference/imputed_data.csv'
PLOT_SAVE_PATH = 'plots/inference/imputation_comparison_umap.png'

def main():
    print("🚀 Starting Cluster Comparison...")

    # 1. Load Original Data
    print(f"Loading original data from {ORIGINAL_DATA_PATH}...")
    data = torch.load(ORIGINAL_DATA_PATH)
    orig_features = data.x.cpu().numpy() # [6386, 8]
    print(f"Original shape: {orig_features.shape}")

    # 2. Load Imputed Data
    print(f"Loading imputed data from {IMPUTED_DATA_PATH}...")
    df_imputed = pd.read_csv(IMPUTED_DATA_PATH, index_col=0)
    # 确保特征维度一致 (只取前8列，如果有额外列的话)
    imputed_features = df_imputed.values[:, :8]
    print(f"Imputed shape: {imputed_features.shape}")

    # 3. Combine Data
    # Stack them vertically
    combined_features = np.vstack([orig_features, imputed_features])
    
    # Create labels for coloring
    # 0 = Original, 1 = Imputed
    labels = np.array(['Original'] * len(orig_features) + ['Imputed (New)'] * len(imputed_features))
    
    print(f"Combined shape: {combined_features.shape}")

    # 4. UMAP Reduction
    print("Running UMAP...")
    reducer = UMAP(n_neighbors=30, min_dist=0.3, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(combined_features)

    # 5. Visualization
    print("Plotting...")
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for better plotting
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'UMAP 1': embedding_2d[:, 0],
        'UMAP 2': embedding_2d[:, 1],
        'Source': labels
    })
    
    # Plot with some transparency to see overlap
    sns.scatterplot(
        data=plot_df, 
        x='UMAP 1', 
        y='UMAP 2', 
        hue='Source', 
        style='Source',
        alpha=0.6,
        s=15,
        palette={'Original': 'steelblue', 'Imputed (New)': 'darkorange'}
    )
    
    plt.title("Distribution Comparison: Original vs. Imputed Data")
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    print(f"✅ Plot saved to {PLOT_SAVE_PATH}")
    
    # 6. Optional: Quantitative Check (MMD or distance between centroids)
    # Calculate centroid distance
    centroid_orig = np.mean(embedding_2d[labels == 'Original'], axis=0)
    centroid_new = np.mean(embedding_2d[labels == 'Imputed (New)'], axis=0)
    dist = np.linalg.norm(centroid_orig - centroid_new)
    print(f"\nDistance between centroids in UMAP space: {dist:.4f}")
    print("(Smaller is better, implies distributions are aligned)")

if __name__ == "__main__":
    main()