import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import sys
import os

# need to add project root to path so we can import from src

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# per-run model directory

run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
models_dir = os.path.join("models", run_id)
os.makedirs(models_dir, exist_ok=True)
print(f"Saving models to: {models_dir}")

from src.model import MultiTaskGNN


# hyperparameters, we can try different configurations later too
lr = 0.005
weight_decay = 5e-4
epochs = 300
hidden_dim = 64
num_heads = 8
dropout = 0.6

def train_step(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    pred_ad, pred_pd, pred_ftd, pred_als, _ = model(data)

    # only use training nodes for loss
    mask = data.train_mask

    # calculate MSE for each disease
    loss_ad = F.mse_loss(pred_ad[mask], data.y[mask, 0].unsqueeze(1))
    loss_pd = F.mse_loss(pred_pd[mask], data.y[mask, 1].unsqueeze(1))
    loss_ftd = F.mse_loss(pred_ftd[mask], data.y[mask, 2].unsqueeze(1))
    loss_als = F.mse_loss(pred_als[mask], data.y[mask, 3].unsqueeze(1))

    # sum them up
    loss = loss_ad + loss_pd + loss_ftd + loss_als

    loss.backward()
    optimizer.step()

    return loss.item()

def eval_model(model, data, mask):
    model.eval()
    with torch.no_grad():
        pred_ad, pred_pd, pred_ftd, pred_als, _ = model(data)

        # same as training but on val/test set
        loss_ad = F.mse_loss(pred_ad[mask], data.y[mask, 0].unsqueeze(1))
        loss_pd = F.mse_loss(pred_pd[mask], data.y[mask, 1].unsqueeze(1))
        loss_ftd = F.mse_loss(pred_ftd[mask], data.y[mask, 2].unsqueeze(1))
        loss_als = F.mse_loss(pred_als[mask], data.y[mask, 3].unsqueeze(1))

        total_loss = loss_ad + loss_pd + loss_ftd + loss_als

    return total_loss.item()

## added a detailed evaluation function to get per-disease metrics
def eval_model_detailed(model, data, mask, split_name="Test"):
    model.eval()
    with torch.no_grad():
        pred_ad, pred_pd, pred_ftd, pred_als, _ = model(data)
        preds = torch.cat([pred_ad, pred_pd, pred_ftd, pred_als], dim=1)

    y = data.y[mask].cpu().numpy()
    yhat = preds[mask].cpu().numpy()

    diseases = ["AD", "PD", "FTD", "ALS"]
    print(f"\n{split_name} metrics (scaled space):")
    for i, name in enumerate(diseases):
        mse = mean_squared_error(y[:, i], yhat[:, i])
        r2 = r2_score(y[:, i], yhat[:, i])
        print(f"  {name}: MSE={mse:.4f}, R²={r2:.4f}")

if __name__ == '__main__':

    # check if we have GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load the preprocessed graph
    data = torch.load('data/02-preprocessed/processed_graph.pt')
    data = data.to(device)
    print(f"Loaded graph with {data.num_nodes} nodes and {data.num_edges} edges")
    
    train_mask = data.train_mask.cpu().numpy().astype(bool)

    x_np = data.x.cpu().numpy()
    y_np = data.y.cpu().numpy()

    # X: mean/std over train nodes
    x_mean = x_np[train_mask].mean(axis=0)
    x_std  = x_np[train_mask].std(axis=0)
    x_std[x_std == 0] = 1.0
    x_scaled = (x_np - x_mean) / x_std

    # Y: mean/std over train nodes
    y_mean = y_np[train_mask].mean(axis=0)
    y_std  = y_np[train_mask].std(axis=0)
    y_std[y_std == 0] = 1.0
    y_scaled = (y_np - y_mean) / y_std

    data.x = torch.from_numpy(x_scaled).to(device)
    data.y = torch.from_numpy(y_scaled).to(device)

    # save scalers (shared across runs)
    os.makedirs('data/02-preprocessed', exist_ok=True)
    np.savez(
        'data/02-preprocessed/gat_scalers_train_based.npz',
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )

    
    # initialize model
    model = MultiTaskGNN(
        in_channels=data.num_node_features,
        hidden_channels=hidden_dim,
        out_channels=4,  # 4 diseases
        heads=num_heads,
        dropout=dropout
    ).to(device)

    print(f"\nModel has {sum(p.numel() for p in model.parameters())} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training loop
    best_val = float('inf')
    best_weights = None


    for epoch in range(1, epochs + 1):
        train_loss = train_step(model, data, optimizer)
        val_loss = eval_model(model, data, data.val_mask)

        # save best model based on validation
        if val_loss < best_val:
            best_val = val_loss
            best_weights = model.state_dict()
            print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} * NEW BEST")
        else:
            print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    # load best model and test
    model.load_state_dict(best_weights)
    test_loss = eval_model(model, data, data.test_mask)

    print(f"\nTraining done!")
    print(f"Best val loss: {best_val:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # detailed test metrics
    eval_model_detailed(model, data, data.train_mask, split_name="Train")
    eval_model_detailed(model, data, data.val_mask, split_name="Val")
    eval_model_detailed(model, data, data.test_mask, split_name="Test")

    # save the model: per-run file + best_model.pt
    best_model_path = os.path.join(models_dir, 'best_model.pt')
    torch.save(best_weights, best_model_path)
    torch.save(best_weights, 'models/best_model.pt')  # for analyze.py
    print("Saved run-specific model to", best_model_path)
    print("Also updated models/best_model.pt for downstream analysis")

