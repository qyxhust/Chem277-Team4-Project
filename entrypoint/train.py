import random
import numpy as np
import itertools
import json
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

def train_one_config(
    data,
    in_channels,
    hidden_dim,
    out_channels,
    heads,
    dropout,
    lr,
    weight_decay,
    epochs,
    device,
):
    
    ## Train a MultiTaskGNN with a given hyperparameter config
   
    model = MultiTaskGNN(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=out_channels,
        heads=heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_step(model, data, optimizer)
        val_loss = eval_model(model, data, data.val_mask)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    return best_val, best_state


def hp_tuning(data, device):
    
    ## simple grid search over a small hyperparameter space.
  
    param_grid = {
        "learning_rate": [1e-3, 5e-4],
        "hidden_dim": [64, 128],
        "dropout": [0.3, 0.5],
        "heads": [4, 8],
        "weight_decay": [1e-4, 5e-4],
    }

    best_loss = float("inf")
    best_config = None
    best_state = None

    in_channels = data.num_node_features
    out_channels = 4  # AD, PD, FTD, ALS

    # try all combinations
    for lr, hd, dr, ah, wd in itertools.product(
        param_grid["learning_rate"],
        param_grid["hidden_dim"],
        param_grid["dropout"],
        param_grid["heads"],
        param_grid["weight_decay"],
    ):
        config = {
            "lr": lr,
            "hidden_dim": hd,
            "dropout": dr,
            "heads": ah,
            "weight_decay": wd,
        }
        print(f"\nTesting config: {config}")

        val_loss, state_dict = train_one_config(
            data=data,
            in_channels=in_channels,
            hidden_dim=hd,
            out_channels=out_channels,
            heads=ah,
            dropout=dr,
            lr=lr,
            weight_decay=wd,
            epochs=120,         
            device=device,
        )

        print(f"  → val_loss = {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = config
            best_state = state_dict
            print("  NEW BEST CONFIG!")

    print("\nHyperparameter tuning done.")
    print(f"Best config: {best_config}")
    print(f"Best val loss: {best_loss:.4f}")

    return best_config, best_state

if __name__ == '__main__':

    # check if we have GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load the preprocessed graph
    data = torch.load('data/02-preprocessed/processed_graph.pt')
    data = data.to(device)
    print(f"Loaded graph with {data.num_nodes} nodes and {data.num_edges} edges")
    
    # scaling 
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

    ## Final Training 

    best_config = {
        "lr": 0.001,
        "hidden_dim": 128,
        "dropout": 0.3,
        "heads": 8,
        "weight_decay": 1e-4,
    }

    max_epochs = 300

    model = MultiTaskGNN(
        in_channels=data.num_node_features,
        hidden_channels=best_config["hidden_dim"],
        out_channels=4,
        heads=best_config["heads"],
        dropout=best_config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_config["lr"],
        weight_decay=best_config["weight_decay"],
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, max_epochs + 1):
        train_loss = train_step(model, data, optimizer)
        val_loss   = eval_model(model, data, data.val_mask)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = model.state_dict()
            tag = "* New best"
        else:
            tag = ""

        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} {tag}")

    # after training loop: load best weights, evaluate, save 
    model.load_state_dict(best_state)
    test_loss = eval_model(model, data, data.test_mask)

    print(f"\nFinal 300-epoch run done!")
    print(f"Best config: {best_config}")
    print(f"Best val loss: {best_val:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # detailed metrics
    eval_model_detailed(model, data, data.train_mask, split_name="Train")
    eval_model_detailed(model, data, data.val_mask,   split_name="Val")
    eval_model_detailed(model, data, data.test_mask,  split_name="Test")

    # baseline zero-prediction comparison
    y = data.y[data.test_mask].cpu().numpy()
    yhat_zero = np.zeros_like(y)

    for i, name in enumerate(["AD", "PD", "FTD", "ALS"]):
        mse0 = mean_squared_error(y[:, i], yhat_zero[:, i])
        r20 = r2_score(y[:, i], yhat_zero[:, i])
        print(f"Baseline ({name}) – MSE={mse0:.4f}, R²={r20:.4f}")

    # save model + config
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, 'best_model.pt')
    torch.save(best_state, best_model_path)
    print("Saved run-specific model to", best_model_path)

    os.makedirs('models', exist_ok=True)
    torch.save(best_state, 'models/best_model.pt')
    print("Also updated models/best_model.pt for downstream analysis")

    # save best hyperparameters so analyze.py can reconstruct the same architecture
    config_run_path = os.path.join(models_dir, 'best_model_config.json')
    config_root_path = os.path.join('models', 'best_model_config.json')
    with open(config_run_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    with open(config_root_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print("Saved best hyperparameters to", config_root_path)
