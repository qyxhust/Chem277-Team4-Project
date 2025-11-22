import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.model import MultiTaskGNN

# Configuration
MODEL_PATH = 'models/best_model.pt'
DATA_PATH = 'data/02-preprocessed/processed_graph.pt'
FEATURE_FILE = 'data/02-preprocessed/protein_features.csv' 

HIDDEN_CHANNELS = 64
HEADS = 8
DROPOUT = 0.6

def plot_top_k_anomalies(df_anomaly, original_x, reconstructed_x, k=10, save_path='plots/top_10_anomalies.png'):
    """Plot feature comparison for top K anomalies in a grid."""
    
    # Setup subplot grid: 2 rows, 5 columns (for k=10)
    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(25, 10))
    axes = axes.flatten() # Flatten to 1D array for easy indexing
    
    features = ['AD_beta', 'AD_logp', 'PD_beta', 'PD_logp', 'FTD_beta', 'FTD_logp', 'ALS_beta', 'ALS_logp']
    x_pos = np.arange(len(features))
    width = 0.35
    
    print(f"\nGenerating plots for Top {k} anomalies...")
    
    for i in range(k):
        if i >= len(df_anomaly): break # Prevent index error if less than k anomalies
        
        gene_name = df_anomaly.iloc[i]['Gene']
        # Note: df_anomaly.index holds the original integer indices from the tensor
        gene_idx = df_anomaly.index[i] 
        score = df_anomaly.iloc[i]['AnomalyScore']
        
        real_vals = original_x[gene_idx].cpu().numpy()
        pred_vals = reconstructed_x[gene_idx].cpu().numpy()
        
        ax = axes[i]
        ax.bar(x_pos - width/2, real_vals, width, label='Real', color='steelblue')
        ax.bar(x_pos + width/2, pred_vals, width, label='Pred', color='darkorange')
        
        # Title & Labels
        ax.set_title(f"{gene_name}\nScore: {score:.1f}", fontsize=10, fontweight='bold')
        ax.set_xticks(x_pos)
        
        # Only show x-labels for bottom row to save space
        if i >= 5:
            ax.set_xticklabels(features, rotation=45, ha='right', fontsize=8)
        else:
            ax.set_xticklabels([])
            
        # Legend only on the first plot
        if i == 0:
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved combined plot: {save_path}")
    plt.close()

def main():
    print("🚀 Starting Anomaly Detection Analysis...")
    
    # 1. Load Data & Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(DATA_PATH).to(device)
    
    try:
        df_features = pd.read_csv(FEATURE_FILE, index_col=0)
        gene_symbols = df_features.index.tolist()
    except Exception as e:
        print(f"Warning: Could not load gene symbols ({e}). Using indices instead.")
        gene_symbols = [f"Protein_{i}" for i in range(data.num_nodes)]

    model = MultiTaskGNN(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=data.num_node_features,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✅ Loaded model from {MODEL_PATH}")
    else:
        print("❌ Error: Model file not found!")
        return

    # 2. Inference
    model.eval()
    with torch.no_grad():
        reconstructed_x = model(data)
        original_x = data.x
        squared_diff = (reconstructed_x - original_x) ** 2
        anomaly_scores = torch.mean(squared_diff, dim=1).cpu().numpy()

    # 3. Prepare Dataframe
    df_anomaly = pd.DataFrame({
        'Gene': gene_symbols[:len(anomaly_scores)], 
        'AnomalyScore': anomaly_scores
    })
    # Important: Keep the original index as a column before sorting
    # Or just use the index after sorting if it preserves original indices
    # Default pandas sort preserves index, so df_anomaly.index[i] will give the original row ID
    df_anomaly = df_anomaly.sort_values(by='AnomalyScore', ascending=False)
    
    # 4. Save CSV
    os.makedirs('results', exist_ok=True)
    df_anomaly.to_csv('results/anomaly_scores.csv')
    print(f"💾 Full results saved to 'results/anomaly_scores.csv'")

    # 5. Plot Top 10
    os.makedirs('plots', exist_ok=True)
    plot_top_k_anomalies(df_anomaly, original_x, reconstructed_x, k=10)

if __name__ == "__main__":
    main()