import sys
import os
import torch
import numpy as np
import pandas as pd
from src.model import MultiTaskGNN
import umap.umap_ as umap
import hdbscan
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# path setup 
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


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


# now try clustering with HDBSCAN
# need to find good min_cluster_size parameter
# going to try some and use silhouette score to pick the best clustering

print("\nTrying different clustering parameters")
sizes_to_try = [15, 25, 35, 50, 75, 100]
best_score = -1
best_labels = None
best_size = None

for size in sizes_to_try:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=size)
    labels = clusterer.fit_predict(embedding_2d)

    # count clusters (exclude noise which is -1)
    n_clusters = len(np.unique(labels[labels != -1]))

    if n_clusters >= 2:
        # only score the non-noise points
        mask = labels != -1
        score = silhouette_score(embedding_2d[mask], labels[mask])
        print(f"  size={size:3d}: {n_clusters:2d} clusters, silhouette={score:.4f}")

        if score > best_score:
            best_score = score
            best_labels = labels
            best_size = size
    else:
        print(f"  size={size:3d}: only {n_clusters} clusters, skipping")

print(f"\nBest clustering: min_size={best_size}, silhouette={best_score:.4f}")

# make dataframe with results
df = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
df['GeneSymbol'] = data.gene_symbols
df['Cluster'] = best_labels
df['Cluster'] = df['Cluster'].apply(lambda x: f'Cluster {x}' if x != -1 else 'Noise')

n_real_clusters = len(df[df['Cluster'] != 'Noise']['Cluster'].unique())
print(f"Total proteins: {len(df)}, Clustered: {(df['Cluster'] != 'Noise').sum()}, Noise: {(df['Cluster'] == 'Noise').sum()}")

# plot the UMAP with clusters
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
plt.title(f'Protein Embeddings UMAP (HDBSCAN min_size={best_size})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plots/protein_embeddings_clustered_umap.png', dpi=300)
print("Saved UMAP plot")
plt.show()

# load original features and join with clusters
features = pd.read_csv('data/02-preprocessed/protein_features.csv', index_col='GeneSymbol')
merged = features.join(df.set_index('GeneSymbol')['Cluster'])

# get average beta values for each cluster
beta_cols = ['AD_beta', 'PD_beta', 'FTD_beta', 'ALS_beta']
cluster_avg = merged.groupby('Cluster')[beta_cols].mean()

# drop noise if it exists
if 'Noise' in cluster_avg.index:
    cluster_avg = cluster_avg.drop('Noise')

# make heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cluster_avg, cmap='coolwarm', annot=True, fmt=".3f", center=0)
plt.title('Average Disease Beta Values by Cluster')
plt.tight_layout()
plt.savefig('plots/cluster_profile_heatmap.png', dpi=300)
print("Saved cluster profile heatmap")
plt.show()

# save cluster assignments (excluding noise)
output = df[df['Cluster'] != 'Noise'][['GeneSymbol', 'Cluster']]
output.to_csv('data/04-predictions/protein_clusters.csv', index=False)
print(f"\nSaved {len(output)} protein cluster assignments to data/04-predictions/protein_clusters.csv")

