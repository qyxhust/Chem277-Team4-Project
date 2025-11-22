import sys
import os
import torch
import pandas as pd
import numpy as np

# Get the absolute path of the directory containing this script (entrypoint/inference)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Up one level -> entrypoint
entrypoint_dir = os.path.dirname(script_dir)
# Up another level -> root
project_root = os.path.dirname(entrypoint_dir)
# Add the project root to Python's path
sys.path.insert(0, project_root)

from src.model import MultiTaskGNN
from torch_geometric.data import Data

# ================= 配置 =================
MODEL_PATH = 'models/best_model.pt'
ORIGINAL_DATA_PATH = 'data/02-preprocessed/processed_graph.pt'

# 您的输入文件路径
# 格式要求：Index=GeneSymbol, Columns=['AD_beta', ..., 'ALS_logp'] (顺序最好一致，也可以乱序)
# 缺失值请留空或设为 NaN
INPUT_DATA_PATH = 'data/raw/filtered_data.csv' 

OUTPUT_PATH = 'results/inference/imputed_data.csv'

# 模型参数
HIDDEN_CHANNELS = 64
HEADS = 8
DROPOUT = 0.6

# 功能开关
MODE = 'PROD'  # 'TEST': 使用原数据挖洞测试 | 'PROD': 读取 INPUT_DATA_PATH 进行生产补全
# =======================================

def main():
    print(f"🚀 Starting Inference (Mode: {MODE})...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Original Graph (Structure & Mapping)
    print(f"Loading graph structure from {ORIGINAL_DATA_PATH}...")
    try:
        orig_data = torch.load(ORIGINAL_DATA_PATH).to(device)
    except FileNotFoundError:
        print("❌ Error: Original graph file not found.")
        return

    # Retrieve Gene List
    if hasattr(orig_data, 'gene_symbols'):
        all_genes = orig_data.gene_symbols
    else:
        print("Warning: 'gene_symbols' missing in .pt. Loading from features csv...")
        try:
            feat = pd.read_csv('data/02-preprocessed/protein_features.csv', index_col=0)
            all_genes = feat.index.tolist()
        except:
            print("❌ Error: Cannot load gene list.")
            return

    gene_to_idx = {gene: i for i, gene in enumerate(all_genes)}
    num_nodes = len(all_genes)
    num_features = orig_data.num_node_features # Should be 8
    print(f"Graph loaded: {num_nodes} nodes, {num_features} features.")

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = MultiTaskGNN(
        in_channels=num_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=num_features,
        heads=HEADS,
        dropout=DROPOUT
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print("❌ Error: Model weights not found!")
        return

    # 3. Prepare Input Data
    full_input = torch.zeros(num_nodes, num_features, device=device)
    
    if MODE == 'TEST':
        print("\n[TEST MODE] Simulating mixed missing values from original data...")
        # 从原数据中获取真实值
        real_values = orig_data.x.clone()
        
        # 随机挖洞：生成一个 Mask，30% 的位置保留，70% 的位置设为 0 (模拟严重缺失)
        # mask = 1 (Known), 0 (Missing)
        mask = torch.rand_like(real_values) > 0.7
        
        # 构造残缺输入
        full_input = real_values * mask.float()
        
        print(f"Simulated input: {mask.sum().item()} values known (out of {mask.numel()})")
        
    elif MODE == 'PROD':
        print(f"\n[PROD MODE] Loading external data from {INPUT_DATA_PATH}...")
        try:
            input_df = pd.read_csv(INPUT_DATA_PATH, index_col=0)
        except FileNotFoundError:
            print("❌ Error: Input file not found.")
            return
            
        # 假设模型训练时的特征顺序
        # 如果您的 CSV 列名与此不同，请在此处建立映射
        feature_names = ['AD_beta', 'AD_logp', 'PD_beta', 'PD_logp', 'FTD_beta', 'FTD_logp', 'ALS_beta', 'ALS_logp']
        
        mapped_count = 0
        # 遍历输入的每一行
        for gene, row in input_df.iterrows():
            if gene in gene_to_idx:
                idx = gene_to_idx[gene]
                
                # 智能填空：只填入非 NaN 的值
                for col_name in input_df.columns:
                    if col_name in feature_names:
                        feat_idx = feature_names.index(col_name)
                        val = row[col_name]
                        if not pd.isna(val):
                            full_input[idx, feat_idx] = float(val)
                
                mapped_count += 1
        print(f"Mapped {mapped_count} genes from input file.")

    # 4. Run Inference
    print("\nRunning GNN Inference...")
    with torch.no_grad():
        # 构造临时数据对象：新特征 + 旧结构
        temp_data = Data(x=full_input, edge_index=orig_data.edge_index).to(device)
        
        # 预测
        predicted_output = model(temp_data)

    # 5. Evaluation (Only for TEST Mode)
    if MODE == 'TEST':
        # 计算只针对缺失部分的恢复效果
        missing_mask = ~mask
        mse = torch.nn.functional.mse_loss(predicted_output[missing_mask], real_values[missing_mask])
        print(f"\n📊 Evaluation on Missing Values:")
        print(f"MSE Loss: {mse.item():.4f}")
        
        # 简单对比前5个值
        print("\nSample Comparison (Real vs Pred):")
        # flatten mask to find indices
        flat_mask = missing_mask.flatten()
        flat_indices = torch.where(flat_mask)[0][:5]
        
        # Get values from flattened tensors
        flat_real = real_values.flatten()
        flat_pred = predicted_output.flatten()
        
        for idx in flat_indices:
            print(f"Real: {flat_real[idx]:.4f} | Pred: {flat_pred[idx]:.4f}")

    # 6. Save Results
    print(f"\nSaving full imputed matrix to {OUTPUT_PATH}...")
    output_df = pd.DataFrame(
        predicted_output.cpu().numpy(),
        index=all_genes,
        columns=['AD_beta', 'AD_logp', 'PD_beta', 'PD_logp', 'FTD_beta', 'FTD_logp', 'ALS_beta', 'ALS_logp']
    )
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_df.to_csv(OUTPUT_PATH)
    print("✅ Done!")

if __name__ == "__main__":
    main()
