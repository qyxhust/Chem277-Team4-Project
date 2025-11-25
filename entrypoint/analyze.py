import random
import numpy as np
import torch
from datetime import datetime


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import sys
import os

# path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# create id based on timestamp
run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

# base output folders for this run
plots_dir = os.path.join("plots", run_id)
pred_dir  = os.path.join("data", "04-predictions", run_id)

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

print(f"Saving plots to: {plots_dir}")
print(f"Saving predictions to: {pred_dir}")


from src.model import MultiTaskGNN
import pandas as pd
import umap.umap_ as umap
import hdbscan
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns 



#adding a helper function to build dataframe
def build_embedding_df(embedding_2d, labels, data):
    df = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
    df['GeneSymbol'] = data.gene_symbols
    df['ClusterLabel'] = labels
    df['Cluster'] = df['ClusterLabel'].apply(lambda x: f'Cluster {x}' if x != -1 else 'Noise')
    return df


# load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

data = torch.load('data/02-preprocessed/processed_graph.pt')
data = data.to(device)

# recreate model architecture (needs to match what we trained)
model = MultiTaskGNN(
    in_channels=data.num_node_features,
    hidden_channels=64,
    out_channels=4,
    heads=8,
    dropout=0.6
).to(device)

model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
model.eval()
print("Model loaded")

# get the embeddings from the trained model
with torch.no_grad():
    _, _, _, _, embeddings = model(data)

embeddings_np = embeddings.cpu().numpy()
print(f"Got embeddings shape: {embeddings_np.shape}")

# run UMAP to reduce to 2D for visualization
# tried a few different parameters, these seemed to work well, we can expand these a bit later 

umap_model = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
embedding_2d = umap_model.fit_transform(embeddings_np)


# clustering with HDBSCAN


print("\nTrying different clustering parameters")
sizes_to_try = [15, 25, 35, 50, 75, 100]

results = []       
labels_dict = {}   # config_id -> labels

for idx, size in enumerate(sizes_to_try):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
    labels = clusterer.fit_predict(embedding_2d)

    # count clusters (exclude noise which is -1)
    n_clusters = len(np.unique(labels[labels != -1]))

    if n_clusters < 2:
        print(f"  size={size:3d}: only {n_clusters} clusters, skipping")
        continue

    # only score the non-noise points
    mask = labels != -1
    score = silhouette_score(embedding_2d[mask], labels[mask])

    config_id = f"min{size}_cfg{idx}"
    print(f"  size={size:3d}: {n_clusters:2d} clusters, silhouette={score:.4f} (id={config_id})")

    results.append({
        "config_id": config_id,
        "min_cluster_size": size,
        "n_clusters": n_clusters,
        "silhouette": score,
    })
    labels_dict[config_id] = labels

# turn into DataFrame and sort by silhouette
results_df = pd.DataFrame(results).sort_values("silhouette", ascending=False)
print("\nAll HDBSCAN configs sorted by silhouette:")
print(results_df)

# summary plot
plt.figure()
sns.barplot(data=results_df, x="min_cluster_size", y="silhouette")
plt.xlabel("HDBSCAN min_cluster_size")
plt.ylabel("Silhouette score")
plt.title("Clustering quality across HDBSCAN configs")
plt.tight_layout()
summary_path = os.path.join(plots_dir, "hdbscan_silhouette_by_min_cluster_size.png")
plt.savefig(summary_path, dpi=300)
plt.close()
print("Saved silhouette summary plot to", summary_path)


## added a section to generate detailed outputs for top configs for clustering, then we can validate with the biological benchmarking
# load original features 
features = pd.read_csv('data/02-preprocessed/protein_features.csv', index_col='GeneSymbol')
beta_cols = ['AD_beta', 'PD_beta', 'FTD_beta', 'ALS_beta']


# how many configs to keep?
top_k = min(10, len(results_df))
print(f"\nGenerating detailed outputs for top {top_k} HDBSCAN configs")

for _, row in results_df.head(top_k).iterrows():
    cfg_id = row["config_id"]
    size = int(row["min_cluster_size"])
    sil = row["silhouette"]
    labels = labels_dict[cfg_id]

    print(f"\nConfig {cfg_id} (min_cluster_size={size}, silhouette={sil:.3f})")

    # build dataframe for this config
    df = build_embedding_df(embedding_2d, labels, data)

    # basic counts
    n_real_clusters = len(df[df['Cluster'] != 'Noise']['Cluster'].unique())
    print(
        f"Total proteins: {len(df)}, "
        f"Clustered: {(df['Cluster'] != 'Noise').sum()}, "
        f"Noise: {(df['Cluster'] == 'Noise').sum()}, "
        f"n_real_clusters={n_real_clusters}"
    )

    # UMAP plot for this config
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=df,
        x='UMAP1',
        y='UMAP2',
        hue='Cluster',
        palette=sns.color_palette("hsv", n_colors=n_real_clusters + 1),
        alpha=0.7,
        s=15
    )
    plt.title(f'Protein Embeddings UMAP – {cfg_id} (min_size={size}, sil={sil:.2f})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    umap_path = os.path.join(
        plots_dir,
        f'protein_embeddings_umap_{cfg_id}_min{size}_sil{sil:.2f}.png'
    )

    plt.savefig(umap_path, dpi=300)

    plt.close()
    print("Saved UMAP plot:", umap_path)

    # join with features to get disease betas
    merged = features.join(df.set_index('GeneSymbol')['Cluster'])
    cluster_avg = merged.groupby('Cluster')[beta_cols].mean()

    # drop noise if present
    if 'Noise' in cluster_avg.index:
        cluster_avg = cluster_avg.drop('Noise')

    # heatmap of disease betas per cluster
    plt.figure(figsize=(8, 6))
    sns.heatmap(cluster_avg, cmap='coolwarm', annot=True, fmt=".3f", center=0)
    plt.title(f'Average Disease Beta Values by Cluster – {cfg_id} (min_size={size})')
    plt.tight_layout()
    
    heatmap_path = os.path.join(
        plots_dir,
        f'cluster_profile_heatmap_{cfg_id}_min{size}_sil{sil:.2f}.png'
    )

    plt.savefig(heatmap_path, dpi=300)

    plt.close()
    print("Saved cluster profile heatmap:", heatmap_path)

    # save cluster assignments (excluding noise)
    output = df[df['Cluster'] != 'Noise'][['GeneSymbol', 'Cluster']]
    out_csv = os.path.join(
        pred_dir,
        f'protein_clusters_{cfg_id}_min{size}_sil{sil:.2f}.csv'
    )
    output.to_csv(out_csv, index=False)
    print(f"Saved {len(output)} protein cluster assignments to {out_csv}")

