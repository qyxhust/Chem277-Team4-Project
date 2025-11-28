import random
import numpy as np
import torch
import json
from datetime import datetime

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

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

def build_embedding_df(embedding_2d, labels, data):
    df = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
    df['GeneSymbol'] = data.gene_symbols
    df['ClusterLabel'] = labels
    df['Cluster'] = df['ClusterLabel'].apply(lambda x: f'Cluster {x}' if x != -1 else 'Noise')
    return df

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

data = torch.load('data/02-preprocessed/processed_graph.pt')
data = data.to(device)

with open('models/best_model_config.json', 'r') as f:
    best_config = json.load(f)

# Apply the same scaling as in training (for X only)

scalers_path = 'data/02-preprocessed/gat_scalers_train_based.npz'
scalers = np.load(scalers_path)

x_mean = scalers['x_mean']
x_std  = scalers['x_std']

x_np = data.x.cpu().numpy()
x_scaled = (x_np - x_mean) / x_std
data.x = torch.from_numpy(x_scaled).to(device)

## We don't need to scale the data.y in here because we are only using the get embeddings and it uses the protein_features.csv betas for cluster summaries
print("Applied train-based scaling to node features (X)")


model = MultiTaskGNN(
    in_channels=data.num_node_features,
    hidden_channels=best_config["hidden_dim"],
    out_channels=4,
    heads=best_config["heads"],
    dropout=best_config["dropout"],
).to(device)

model.load_state_dict(torch.load('models/best_model.pt', map_location=device))
model.eval()
print("Model loaded with tuned hyperparameters:", best_config)

with torch.no_grad():
    _, _, _, _, embeddings = model(data)

embeddings_np = embeddings.cpu().numpy()
print(f"Got embeddings shape: {embeddings_np.shape}")

umap_model = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
embedding_2d = umap_model.fit_transform(embeddings_np)

umap_coords_path = os.path.join(pred_dir, "protein_embeddings_umap_coords.csv")
pd.DataFrame(embedding_2d, columns=["UMAP1", "UMAP2"]).assign(
    GeneSymbol=data.gene_symbols
).to_csv(umap_coords_path, index=False)
print("Saved 2D UMAP coordinates to", umap_coords_path)


print("\nTrying different clustering parameters")

sizes_to_try = [5, 10, 15, 20, 25, 30, 35, 40, 50]

min_clusters = 2
max_clusters = 60

results = []
labels_dict = {}

## Note: Use silhouette_score only to compare HDBSCAN configs, final biological interpretation will rely on disease beta profiles, and pathway enrichmen analysis not only silhoutte score alone

for idx, size in enumerate(sizes_to_try):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
    labels = clusterer.fit_predict(embedding_2d)

    n_clusters = len(np.unique(labels[labels != -1]))

    if n_clusters == 0:
        print(f"  size={size:3d}: no clusters (all noise), skipping")
        continue

    if not (min_clusters <= n_clusters <= max_clusters):
        print(
            f"  size={size:3d}: {n_clusters} clusters, "
            f"outside [{min_clusters}, {max_clusters}], skipping"
        )
        continue

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

results_df = pd.DataFrame(results).sort_values("silhouette", ascending=False)
print("\nAll HDBSCAN configs sorted by silhouette:")
print(results_df)

if results_df.empty:
    print("No valid HDBSCAN configs in the desired cluster range, stopping.")
    raise SystemExit

best_row = results_df.iloc[0]
print(
    f"\nUsing primary config for interpretation: "
    f"{best_row['config_id']} "
    f"(min_cluster_size={int(best_row['min_cluster_size'])}, "
    f"n_clusters={int(best_row['n_clusters'])}, "
    f"silhouette={best_row['silhouette']:.4f})"
)

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

features = pd.read_csv('data/02-preprocessed/protein_features.csv', index_col='GeneSymbol')
beta_cols = ['AD_beta', 'PD_beta', 'FTD_beta', 'ALS_beta']
disease_names = ['AD', 'PD', 'FTD', 'ALS']

def dominant_disease(row, min_beta=0.1, min_margin=0.1):
    vals = row[beta_cols].values
    max_idx = np.argmax(vals)
    max_val = vals[max_idx]
    sorted_vals = np.sort(vals)
    if len(sorted_vals) > 1:
        margin = max_val - sorted_vals[-2]
    else:
        margin = 0.0
    if max_val < min_beta:
        return "neutral"
    if margin < min_margin:
        return "mixed"
    return disease_names[max_idx]

top_k = min(30, len(results_df))
print(f"\nGenerating detailed outputs for top {top_k} HDBSCAN configs")

for _, row in results_df.head(top_k).iterrows():
    cfg_id = row["config_id"]
    size = int(row["min_cluster_size"])
    sil = row["silhouette"]
    labels = labels_dict[cfg_id]

    print(f"\nConfig {cfg_id} (min_cluster_size={size}, silhouette={sil:.3f})")

    df = build_embedding_df(embedding_2d, labels, data)

    n_real_clusters = len(df[df['Cluster'] != 'Noise']['Cluster'].unique())
    n_noise = (df['Cluster'] == 'Noise').sum()
    n_total = len(df)
    n_clustered = n_total - n_noise

    print(
        f"Total proteins: {n_total}, "
        f"Clustered (non-noise): {n_clustered}, "
        "Noise: {0}, n_real_clusters={1}".format(n_noise, n_real_clusters)
    )

    df_clustered = df[df['Cluster'] != 'Noise'].copy()

    if n_real_clusters > 0:
        plt.figure(figsize=(14, 10))
        sns.scatterplot(
            data=df_clustered,
            x='UMAP1',
            y='UMAP2',
            hue='Cluster',
            palette=sns.color_palette("hsv", n_colors=n_real_clusters),
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
    else:
        print("No real clusters (all noise), skipping UMAP plot.")

    merged = features.join(df.set_index('GeneSymbol')['Cluster'])
    cluster_avg = merged.groupby('Cluster')[beta_cols].mean()
    cluster_counts = merged.groupby('Cluster').size().rename('n_genes')
    cluster_summary = cluster_avg.join(cluster_counts)
    cluster_summary['dominant_disease'] = cluster_summary.apply(dominant_disease, axis=1)
    cluster_summary['pan_score'] = cluster_summary[beta_cols].mean(axis=1)
    cluster_summary['config_id'] = cfg_id
    cluster_summary['min_cluster_size'] = size
    cluster_summary['silhouette'] = sil

    if 'Noise' in cluster_summary.index:
        cluster_summary_no_noise = cluster_summary.drop('Noise')
    else:
        cluster_summary_no_noise = cluster_summary

    plt.figure(figsize=(8, 6))
    sns.heatmap(cluster_summary_no_noise[beta_cols], cmap='coolwarm', annot=True, fmt=".3f", center=0)
    plt.title(f'Average Disease Beta Values by Cluster – {cfg_id} (min_size={size})')
    plt.tight_layout()
    
    heatmap_path = os.path.join(
        plots_dir,
        f'cluster_profile_heatmap_{cfg_id}_min{size}_sil{sil:.2f}.png'
    )

    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print("Saved cluster profile heatmap:", heatmap_path)

    summary_path = os.path.join(
        pred_dir,
        f'cluster_summary_{cfg_id}_min{size}_sil{sil:.2f}.csv'
    )
    cluster_summary_no_noise.to_csv(summary_path)
    print("Saved cluster summary to", summary_path)

    output = df[df['Cluster'] != 'Noise'][['GeneSymbol', 'Cluster']]
    out_csv = os.path.join(
        pred_dir,
        f'protein_clusters_{cfg_id}_min{size}_sil{sil:.2f}.csv'
    )
    output.to_csv(out_csv, index=False)
    print(f"Saved {len(output)} protein cluster assignments to {out_csv}")

    metascape_dir = os.path.join(
        pred_dir,
        f'metascape_lists_{cfg_id}_min{size}_sil{sil:.2f}'
    )
    os.makedirs(metascape_dir, exist_ok=True)

    for cluster_name, group in df_clustered.groupby('Cluster'):
        genes = group['GeneSymbol'].dropna().unique()
        genes = [g for g in genes if isinstance(g, str) and g.strip() != ""]

        if len(genes) == 0:
            continue

        txt_path = os.path.join(
            metascape_dir,
            f'{cfg_id}_{cluster_name}_genes.txt'
        )
        with open(txt_path, 'w') as f:
            f.write("\n".join(genes))

        print(f"Saved {len(genes)} genes for {cluster_name} to {txt_path}")
