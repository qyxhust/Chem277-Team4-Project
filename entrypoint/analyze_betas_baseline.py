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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

run_id = datetime.now().strftime("baseline_run_%Y%m%d_%H%M%S")

plots_dir = os.path.join("plots", run_id)
pred_dir  = os.path.join("data", "04-predictions", run_id)

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

print(f"[BASELINE] Saving plots to: {plots_dir}")
print(f"[BASELINE] Saving predictions to: {pred_dir}")

import pandas as pd
import umap.umap_ as umap
import hdbscan
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def build_embedding_df(embedding_2d, genes):
    df = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
    df['GeneSymbol'] = genes
    return df

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[BASELINE] Device: {device}")

data = torch.load('data/02-preprocessed/processed_graph.pt')
genes = np.array(data.gene_symbols)

features = pd.read_csv('data/02-preprocessed/protein_features.csv', index_col='GeneSymbol')
beta_cols = ['AD_beta', 'PD_beta', 'FTD_beta', 'ALS_beta']
disease_names = ['AD', 'PD', 'FTD', 'ALS']

X = features.loc[genes, beta_cols].values

umap_model = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
embedding_2d = umap_model.fit_transform(X)
print(f"[BASELINE] Got embedding shape (no graph): {embedding_2d.shape}")

print("\n[BASELINE] Trying different clustering parameters")

sizes_to_try = [5, 10, 15, 20, 25, 35, 50]

min_clusters = 4
max_clusters = 25

results = []
labels_dict = {}

for idx, size in enumerate(sizes_to_try):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
    labels = clusterer.fit_predict(embedding_2d)

    n_clusters = len(np.unique(labels[labels != -1]))

    if n_clusters == 0:
        print(f"  [BASELINE] size={size:3d}: no clusters (all noise), skipping")
        continue

    if not (min_clusters <= n_clusters <= max_clusters):
        print(
            f"  [BASELINE] size={size:3d}: {n_clusters} clusters, "
            f"outside [{min_clusters}, {max_clusters}], skipping"
        )
        continue

    mask = labels != -1
    score = silhouette_score(embedding_2d[mask], labels[mask])

    config_id = f"min{size}_cfg{idx}"
    print(f"  [BASELINE] size={size:3d}: {n_clusters:2d} clusters, silhouette={score:.4f} (id={config_id})")

    results.append({
        "config_id": config_id,
        "min_cluster_size": size,
        "n_clusters": n_clusters,
        "silhouette": score,
    })
    labels_dict[config_id] = labels

results_df = pd.DataFrame(results).sort_values("silhouette", ascending=False)
print("\n[BASELINE] All HDBSCAN configs sorted by silhouette:")
print(results_df)

if results_df.empty:
    print("[BASELINE] No valid HDBSCAN configs in the desired cluster range, stopping.")
    raise SystemExit

plt.figure()
sns.barplot(data=results_df, x="min_cluster_size", y="silhouette")
plt.xlabel("HDBSCAN min_cluster_size")
plt.ylabel("Silhouette score")
plt.title("[BASELINE] Clustering quality across HDBSCAN configs")
plt.tight_layout()
summary_path = os.path.join(plots_dir, "baseline_hdbscan_silhouette_by_min_cluster_size.png")
plt.savefig(summary_path, dpi=300)
plt.close()
print("[BASELINE] Saved silhouette summary plot to", summary_path)

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
print(f"\n[BASELINE] Generating detailed outputs for top {top_k} HDBSCAN configs")

for _, row in results_df.head(top_k).iterrows():
    cfg_id = row["config_id"]
    size = int(row["min_cluster_size"])
    sil = row["silhouette"]
    labels = labels_dict[cfg_id]

    print(f"\n[BASELINE] Config {cfg_id} (min_cluster_size={size}, silhouette={sil:.3f})")

    df = build_embedding_df(embedding_2d, genes)
    df['ClusterLabel'] = labels
    df['Cluster'] = df['ClusterLabel'].apply(lambda x: f'Cluster {x}' if x != -1 else 'Noise')

    n_real_clusters = len(df[df['Cluster'] != 'Noise']['Cluster'].unique())
    n_noise = (df['Cluster'] == 'Noise').sum()
    n_total = len(df)
    n_clustered = n_total - n_noise

    print(
        f"[BASELINE] Total proteins: {n_total}, "
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
        plt.title(f'[BASELINE] Protein Betas UMAP – {cfg_id} (min_size={size}, sil={sil:.2f})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        umap_path = os.path.join(
            plots_dir,
            f'baseline_protein_betas_umap_{cfg_id}_min{size}_sil{sil:.2f}.png'
        )

        plt.savefig(umap_path, dpi=300)
        plt.close()
        print("[BASELINE] Saved UMAP plot:", umap_path)
    else:
        print("[BASELINE] No real clusters (all noise), skipping UMAP plot.")

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
    plt.title(f'[BASELINE] Avg Disease Betas by Cluster – {cfg_id} (min_size={size})')
    plt.tight_layout()
    
    heatmap_path = os.path.join(
        plots_dir,
        f'baseline_cluster_profile_heatmap_{cfg_id}_min{size}_sil{sil:.2f}.png'
    )

    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print("[BASELINE] Saved cluster profile heatmap:", heatmap_path)

    summary_path = os.path.join(
        pred_dir,
        f'baseline_cluster_summary_{cfg_id}_min{size}_sil{sil:.2f}.csv'
    )
    cluster_summary_no_noise.to_csv(summary_path)
    print("[BASELINE] Saved cluster summary to", summary_path)

    output = df[df['Cluster'] != 'Noise'][['GeneSymbol', 'Cluster']]
    out_csv = os.path.join(
        pred_dir,
        f'baseline_protein_clusters_{cfg_id}_min{size}_sil{sil:.2f}.csv'
    )
    output.to_csv(out_csv, index=False)
    print(f"[BASELINE] Saved {len(output)} protein cluster assignments to {out_csv}")

    metascape_dir = os.path.join(
        pred_dir,
        f'baseline_metascape_lists_{cfg_id}_min{size}_sil{sil:.2f}'
    )
    os.makedirs(metascape_dir, exist_ok=True)

    for cluster_name, group in df_clustered.groupby('Cluster'):
        genes_cluster = group['GeneSymbol'].dropna().unique()
        genes_cluster = [g for g in genes_cluster if isinstance(g, str) and g.strip() != ""]

        if len(genes_cluster) == 0:
            continue

        txt_path = os.path.join(
            metascape_dir,
            f'baseline_{cfg_id}_{cluster_name}_genes.txt'
        )
        with open(txt_path, 'w') as f:
            f.write("\n".join(genes_cluster))

        print(f"[BASELINE] Saved {len(genes_cluster)} genes for {cluster_name} to {txt_path}")
