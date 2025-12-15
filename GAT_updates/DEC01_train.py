import random
import numpy as np
import itertools
import json
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
models_dir = os.path.join("GAT_updates", run_id)
os.makedirs(models_dir, exist_ok=True)
print(f"Saving models to: {models_dir}")

from DEC01_model import MultiTaskGNN


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
    loss = data.loss_weights[0]*loss_ad + data.loss_weights[1]*loss_pd + data.loss_weights[2]*loss_ftd + data.loss_weights[3]*loss_als

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

        total_loss = data.loss_weights[0]*loss_ad + data.loss_weights[1]*loss_pd + data.loss_weights[2]*loss_ftd + data.loss_weights[3]*loss_als

    return total_loss.item()

## added a detailed evaluation function to get per-disease metrics
def eval_model_detailed(model, data, mask, y_mean, y_std, split_name="Test"):
    model.eval()
    with torch.no_grad():
        pred_ad, pred_pd, pred_ftd, pred_als, _ = model(data)
        preds = torch.cat([pred_ad, pred_pd, pred_ftd, pred_als], dim=1)

    y = data.y[mask].cpu().numpy() # move true values and predicts back to CPU
    yhat = preds[mask].cpu().numpy()

    y_orig = (y * y_std[None, :]) + y_mean[None, :] # broadcasting needed so that inverse transformation is correct
    yhat_orig = (yhat * y_std[None, :]) + y_mean[None, :]

    diseases = ["AD", "PD", "FTD", "ALS"]
    print(f"\n{split_name} metrics (original space):")
    for i, name in enumerate(diseases):
        mse = mean_squared_error(y_orig[:, i], yhat_orig[:, i])
        r2 = r2_score(y_orig[:, i], yhat_orig[:, i])
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
    '''
    Citation:
    Van den Berg T. Parameter Grid-searching with Python's itertools. SITMO Machine Learning | Quantitative Finance. 
    Published December 29, 2020. Accessed December 13, 2025. 
    https://www.sitmo.com/grid-searching-for-optimal-hyperparameters-with-itertools/
    '''
    
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

def plot(t_losses, v_losses, epochs):
    '''
    Plots MSE loss over epochs for training and validation.

    Parameters
    ----------
    t_losses : list
        List containing training losses for all epochs.
    v_losses : list
        List containing validation losses for all epochs.
    epochs : list
        List containing epoch numbers. 
    '''

    plt.figure(figsize=(8,6))
    plt.plot(epochs, t_losses, label = 'Training MSE')
    plt.plot(epochs, v_losses, label = 'Validation MSE')

    plt.title("Loss Over Epochs for Multi-task GAT", fontsize=18)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Log Total MSE", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.yscale("log")
    plt.legend()
    plt.savefig("loss_over_epochs")

if __name__ == '__main__':

    # check if we have GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load the preprocessed graph
    data = torch.load('processed_graph.pt', weights_only=False)
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

    print(x_std, y_std)

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

    # intialize lists to keep track of losses, epochs for plotting
    train_losses = []
    val_losses = []
    epoch_nums = []

    for epoch in range(1, max_epochs + 1):
        train_loss = train_step(model, data, optimizer)
        train_losses.append(train_loss)
        val_loss   = eval_model(model, data, data.val_mask)
        val_losses.append(val_loss)
        epoch_nums.append(epoch)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = model.state_dict()
            tag = "* New best"
        else:
            tag = ""

        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} {tag}")

    # plot loss over epochs
    plot(train_losses, val_losses, epoch_nums)

    # after training loop: load best weights, evaluate, save 
    model.load_state_dict(best_state)
    test_loss = eval_model(model, data, data.test_mask)

    print(f"\nFinal 300-epoch run done!")
    print(f"Best config: {best_config}")
    print(f"Best val loss: {best_val:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # detailed metrics
    eval_model_detailed(model, data, data.train_mask, y_mean=y_mean, y_std=y_std, split_name="Train")
    eval_model_detailed(model, data, data.val_mask, y_mean=y_mean, y_std=y_std, split_name="Val")
    eval_model_detailed(model, data, data.test_mask, y_mean=y_mean, y_std=y_std, split_name="Test")

    # baseline zero-prediction comparison
    y_scaled_test = data.y[data.test_mask].cpu().numpy()
    y_orig = (y_scaled_test * y_std[None, :]) + y_mean[None, :] # broadcasting y_mean and y_std
    yhat_zero_scaled = np.zeros_like(y_orig)
    yhat_zero_orig = (yhat_zero_scaled * y_std[None, :]) + y_mean[None, :]

    for i, name in enumerate(["AD", "PD", "FTD", "ALS"]):
        mse0 = mean_squared_error(y_orig[:, i], yhat_zero_orig[:, i])
        r20 = r2_score(y_orig[:, i], yhat_zero_orig[:, i])
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

    # plot metrics for each split
    diseases = ['AD', 'PD', 'FTD', 'ALS']

    # Train metrics (original space):
    # AD: MSE=0.0022, R²=0.2787
    # PD: MSE=0.0009, R²=0.3068
    # FTD: MSE=0.0014, R²=0.1625
    # ALS: MSE=0.0046, R²=0.0682

    best_train = {'MSE_ad': 0.0022, 'R2_ad': 0.2787, 
                  'MSE_pd': 0.0009, 'R2_pd': 0.3068, 
                  'MSE_ftd': 0.0014, 'R2_ftd': 0.1625, 
                  'MSE_als': 0.0046, 'R2_als': 0.0682}

    # Val metrics (original space):
    # AD: MSE=0.0022, R²=0.2471
    # PD: MSE=0.0010, R²=0.2840
    # FTD: MSE=0.0012, R²=0.1413
    # ALS: MSE=0.0043, R²=0.0261
    best_val = {'MSE_ad': 0.0022, 'R2_ad': 0.2840, 
                'MSE_pd': 0.0010, 'R2_pd': 0.2089, 
                'MSE_ftd': 0.0012, 'R2_ftd': 0.1413, 
                'MSE_als': 0.0043, 'R2_als': 0.0261}


    # Test metrics (original space):
    # AD: MSE=0.0023, R²=0.2704
    # PD: MSE=0.0009, R²=0.2911
    # FTD: MSE=0.0014, R²=0.1434
    # ALS: MSE=0.0039, R²=0.0383
    best_test = {'MSE_ad': 0.0023, 'R2_ad': 0.2704, 
                'MSE_pd': 0.0009, 'R2_pd': 0.2911, 
                'MSE_ftd': 0.0014, 'R2_ftd': 0.1434, 
                'MSE_als': 0.0039, 'R2_als': 0.0383}
    
    # put MSE values for all splits in a dict
    all_mse = {
        'Training': [best_train[f'MSE_{d.lower()}'] for d in diseases],
        'Validation': [best_val[f'MSE_{d.lower()}'] for d in diseases],
        'Testing': [best_test[f'MSE_{d.lower()}'] for d in diseases]
    }

    mse_df = pd.DataFrame(all_mse, index=diseases)
    mse_df.index.name = 'disease'

    # melt df to get disease, splits, and mse columns
    mse_melted = mse_df.reset_index().melt(
        id_vars='disease', 
        var_name='split', 
        value_name='MSE'
    )

    # plotting MSE
    plt.figure(figsize=(10,6))
    sns.barplot(x='disease', y='MSE', hue='split', data=mse_melted, palette='viridis')
    plt.ylabel('MSE Value', fontsize=16)
    plt.xlabel('Disease', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('MSE for Each Split', fontsize=18)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=14)
    plt.ylim(0, 0.005)
    plt.tight_layout()
    plt.savefig("GAT_MSE.png")

    # plotting R^2
    # put R^2 values for all splits in a dict
    all_r2 = {
        'Training': [best_train[f'R2_{d.lower()}'] for d in diseases],
        'Validation': [best_val[f'R2_{d.lower()}'] for d in diseases],
        'Testing': [best_test[f'R2_{d.lower()}'] for d in diseases]
    }

    r2_df = pd.DataFrame(all_r2, index=diseases)
    r2_df.index.name = 'disease'

    # melt df to get disease, splits, and r2 columns
    r2_melted = r2_df.reset_index().melt(
        id_vars='disease', 
        var_name='split', 
        value_name='R2'
    )

    # plotting R^2
    plt.figure(figsize=(10,6))
    sns.barplot(x='disease', y='R2', hue='split', data=r2_melted, palette='viridis')
    plt.ylabel('$R^2$', fontsize=16)
    plt.xlabel('Disease', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('$R^2$ for Each Split', fontsize=18)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=14)
    plt.ylim(0, 0.35)
    plt.tight_layout()
    plt.savefig("GAT_R2.png")


    # plotting for baseline
    # Baseline (AD) – MSE=0.0032, R²=-0.0009
    # Baseline (PD) – MSE=0.0012, R²=-0.0001
    # Baseline (FTD) – MSE=0.0016, R²=-0.0003
    # Baseline (ALS) – MSE=0.0040, R²=-0.0000
    baseline = {'MSE_ad': 0.0032, 'R2_ad': -0.0009, 
                'MSE_pd': 0.0012, 'R2_pd': -0.0001, 
                'MSE_ftd': 0.0016, 'R2_ftd': -0.0003, 
                'MSE_als': 0.0040, 'R2_als': -0.0000}
    
    # all metrics to turn into df
    all_bl_metrics = {'Disease' : diseases,
        'MSE': [baseline[f'MSE_{d.lower()}'] for d in diseases],
        'R2' : [baseline[f'R2_{d.lower()}'] for d in diseases]
        }

    all_bl_df = pd.DataFrame(all_bl_metrics)
  

    # melt df to get just MSE
    mse_melted_bl = all_bl_df.reset_index().melt(
        id_vars='Disease', 
        value_vars='MSE',
        var_name='Metric', 
        value_name='Value'
    )

    # plotting MSE
    plt.figure(figsize=(10,6))
    sns.barplot(x='Disease', y='Value', data=mse_melted_bl, palette='viridis')
    plt.ylabel('MSE Value', fontsize=16)
    plt.xlabel('Disease', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Baseline Model MSE by Disease', fontsize=18)
    #plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=14)
    plt.ylim(0, 0.005)
    plt.tight_layout()
    plt.savefig("baseline_MSE.png")

    # plotting R^2

    # melt df to get just R2
    r2_melted_bl = all_bl_df.reset_index().melt(
    id_vars='Disease', 
    value_vars='R2',
    var_name='Metric', 
    value_name='Value'
    )

    # plotting R^2
    plt.figure(figsize=(10,6))
    sns.barplot(x='Disease', y='Value', data=r2_melted_bl, palette='viridis')
    plt.ylabel('$R^2$', fontsize=16)
    plt.xlabel('Disease', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Baseline Model $R^2$ by Disease', fontsize=18)
    #plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=14)
    plt.ylim(0, 0.35)
    plt.tight_layout()
    plt.savefig("baseline_R2.png")
