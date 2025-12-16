"""
Training script for Graph Attention Network with threshold optimization.
Uses PyTorch Geometric DataLoader for parallel data loading.
Incorporates threshold optimization on validation set to maximize F1-score.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader

from data.dataset import patient_graph_dataset
from gat_proper import gat_model
from protein_gat_trainer import plot_training_history, plot_confusion_matrix


def reduce_graph_topk(edge_index, edge_weight, k=50, num_proteins=6245):
    """Reduce graph to top-k neighbors per protein by confidence score."""
    print(f"reducing graph to top-{k} neighbors per protein")
    
    new_edges = []
    new_weights = []
    
    for dst in range(num_proteins):
        mask = (edge_index[1] == dst)
        src_nodes = edge_index[0][mask]
        weights = edge_weight[mask]
        
        if len(weights) > k:
            topk_indices = torch.topk(weights, k)[1]
            src_nodes = src_nodes[topk_indices]
            weights = weights[topk_indices]
        
        for src, w in zip(src_nodes, weights):
            new_edges.append([src.item(), dst])
            new_weights.append(w.item())
    
    if new_edges:
        edge_index_topk = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        edge_weight_topk = torch.tensor(new_weights, dtype=torch.float32)
    else:
        edge_index_topk = edge_index
        edge_weight_topk = edge_weight
    
    reduction = edge_index.shape[1] / edge_index_topk.shape[1]
    print(f"reduced edges: {edge_index.shape[1]:,} -> {edge_index_topk.shape[1]:,} ({reduction:.1f}x)")
    
    return edge_index_topk, edge_weight_topk


def ensure_2d(y):
    """Ensure tensor is 2D: [batch_size, 1]."""
    return y.unsqueeze(1) if y.dim() == 1 else y


def optimize_threshold(val_logits, val_labels):
    """
    Find optimal threshold by maximizing F1-score on validation set.
    
    Args:
        val_logits: raw logits from model
        val_labels: binary labels
    
    Returns:
        optimal_threshold: float
        f1_scores: dict of threshold -> f1_score
    """
    val_logits = np.array(val_logits)
    val_labels = np.array(val_labels).astype(int)
    
    thresholds = np.arange(-1.0, 1.01, 0.01)
    f1_scores = {}
    
    for threshold in thresholds:
        predictions = (val_logits > threshold).astype(int)
        f1 = f1_score(val_labels, predictions, zero_division=0)
        f1_scores[threshold] = f1
    
    optimal_threshold = max(f1_scores, key=f1_scores.get)
    max_f1 = f1_scores[optimal_threshold]
    
    print(f"\nthreshold optimization:")
    print(f"  optimal threshold: {optimal_threshold:.4f}")
    print(f"  max F1-score: {max_f1:.4f}")
    
    return optimal_threshold, f1_scores


def main():
    print("training Graph Attention Network with threshold optimization\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}\n")
    
    output_dir = Path("../results/gat_proper")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Load data
    print("loading data...")
    original_data = torch.load("./binary_label_data.pt", map_location=device, weights_only=False)
    x_np = original_data.x.cpu().numpy()
    y_np = original_data.y.cpu().numpy()
    print(f"data loaded: {x_np.shape[0]} patients, {x_np.shape[1]} proteins")
    
    # Load graph
    print("loading protein interaction network...")
    dataset_info = torch.load("./patient_graphs_dataset.pt", map_location=device, weights_only=False)
    example_graph = dataset_info['graphs'][0]
    
    edge_index = example_graph.edge_index.to(device)
    edge_weight = example_graph.edge_attr.squeeze().to(device)
    print(f"network loaded: {edge_index.shape[1]:,} edges")
    
    # Reduce graph
    print("\noptimizing network for computational efficiency...")
    edge_index, edge_weight = reduce_graph_topk(edge_index, edge_weight, k=50)
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    
    # Create splits
    print("\ncreating train/validation/test splits...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    n = x_np.shape[0]
    split_labels = np.random.choice([0, 1, 2], n, p=[0.7, 0.15, 0.15])
    
    train_mask = (split_labels == 0)
    val_mask = (split_labels == 1)
    test_mask = (split_labels == 2)
    
    print(f"splits: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    
    # Preprocess features
    print("preprocessing features...")
    scaler = StandardScaler()
    scaler.fit(x_np[train_mask])
    x_scaled = scaler.transform(x_np)
    
    disease_count = y_np.sum()
    pos_weight = len(y_np) / (disease_count * 2)
    print(f"pos_weight for class imbalance: {pos_weight:.4f}")
    
    # Create tensors and datasets
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_np.reshape(-1, 1), dtype=torch.float32)
    
    train_dataset = patient_graph_dataset(
        x_tensor[train_mask], y_tensor[train_mask], edge_index, edge_weight
    )
    val_dataset = patient_graph_dataset(
        x_tensor[val_mask], y_tensor[val_mask], edge_index, edge_weight
    )
    test_dataset = patient_graph_dataset(
        x_tensor[test_mask], y_tensor[test_mask], edge_index, edge_weight
    )
    
    # Create dataloaders
    print("creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                            num_workers=0)
    
    # Initialize model
    print("\ninitializing model:")
    print("  architecture: 2 GAT layers with attention")
    print("  layer 1: 1 input -> 8 heads -> 256 dimensions (32 per head)")
    print("  layer 2: 256 -> 1 head -> 32 dimensions")
    print("  output: 32 -> 1 (binary classification)")
    
    model = gat_model(input_dim=1, hidden_dim=32, output_dim=1, 
                     num_heads=8, dropout=0.5)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  total parameters: {num_params:,}")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("\ntraining started...")
    
    # Training loop
    for epoch in range(100):
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            logits = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            loss = criterion(logits, ensure_2d(batch.y))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            with torch.no_grad():
                pred = (logits > 0).float()
                acc = (pred == ensure_2d(batch.y)).float().mean()
            
            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()
            num_batches += 1
        
        epoch_train_loss /= num_batches
        epoch_train_acc /= num_batches
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
                loss = criterion(logits, ensure_2d(batch.y))
                
                pred = (logits > 0).float()
                acc = (pred == ensure_2d(batch.y)).float().mean()
                
                epoch_val_loss += loss.item()
                epoch_val_acc += acc.item()
                num_batches += 1
            
            epoch_val_loss /= num_batches
            epoch_val_acc /= num_batches
            val_losses.append(epoch_val_loss)
            val_accs.append(epoch_val_acc)
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or patience_counter >= patience:
                print(f"epoch {epoch+1:3d} | train: {epoch_train_loss:.4f} ({epoch_train_acc:.4f}) | "
                      f"val: {epoch_val_loss:.4f} ({epoch_val_acc:.4f}) | patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"early stopping at epoch {best_epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), output_dir / 'best_model.pt')
    
    # Evaluation
    print("\nevaluating on all splits...")
    model.eval()
    
    # Training set
    train_logits, train_labels = [], []
    train_loss_total = 0
    train_batches = 0
    
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            loss = criterion(logits, ensure_2d(batch.y))
            
            train_logits.extend(logits.cpu().numpy().flatten())
            train_labels.extend(ensure_2d(batch.y).cpu().numpy().flatten())
            train_loss_total += loss.item()
            train_batches += 1
    
    train_loss = train_loss_total / train_batches
    train_preds = (np.array(train_logits) > 0).astype(int)
    train_acc = (train_preds == np.array(train_labels)).mean()
    cm_train = confusion_matrix(train_labels, train_preds)
    
    # Validation set
    val_logits, val_labels = [], []
    val_loss_total = 0
    val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            loss = criterion(logits, ensure_2d(batch.y))
            
            val_logits.extend(logits.cpu().numpy().flatten())
            val_labels.extend(ensure_2d(batch.y).cpu().numpy().flatten())
            val_loss_total += loss.item()
            val_batches += 1
    
    val_loss = val_loss_total / val_batches
    val_preds = (np.array(val_logits) > 0).astype(int)
    val_acc = (val_preds == np.array(val_labels)).mean()
    
    # Threshold optimization
    optimal_threshold, f1_scores = optimize_threshold(val_logits, val_labels)
    
    val_preds_optimal = (np.array(val_logits) > optimal_threshold).astype(int)
    val_acc_optimal = (val_preds_optimal == np.array(val_labels)).mean()
    cm_val_optimal = confusion_matrix(val_labels, val_preds_optimal)
    
    # Test set
    test_logits, test_labels = [], []
    test_loss_total = 0
    test_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            loss = criterion(logits, ensure_2d(batch.y))
            
            test_logits.extend(logits.cpu().numpy().flatten())
            test_labels.extend(ensure_2d(batch.y).cpu().numpy().flatten())
            test_loss_total += loss.item()
            test_batches += 1
    
    test_loss = test_loss_total / test_batches
    
    test_preds_default = (np.array(test_logits) > 0).astype(int)
    test_acc_default = (test_preds_default == np.array(test_labels)).mean()
    cm_test_default = confusion_matrix(test_labels, test_preds_default)
    
    test_preds_optimal = (np.array(test_logits) > optimal_threshold).astype(int)
    test_acc_optimal = (test_preds_optimal == np.array(test_labels)).mean()
    cm_test_optimal = confusion_matrix(test_labels, test_preds_optimal)
    
    # Print results
    print("\ntrain metrics (default threshold 0.0):")
    print(f"  loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
    
    print("\nvalidation metrics (default threshold 0.0):")
    print(f"  loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")
    
    print(f"\nvalidation metrics (optimal threshold {optimal_threshold:.4f}):")
    print(f"  accuracy: {val_acc_optimal:.4f}")
    
    print("\ntest metrics (default threshold 0.0):")
    print(f"  loss: {test_loss:.4f}, accuracy: {test_acc_default:.4f}")
    print(f"  confusion matrix:\n{cm_test_default}")
    
    print(f"\ntest metrics (optimal threshold {optimal_threshold:.4f}):")
    print(f"  loss: {test_loss:.4f}, accuracy: {test_acc_optimal:.4f}")
    print(f"  confusion matrix:\n{cm_test_optimal}")
    
    print(f"\nclassification report (test set, optimal threshold):")
    print(classification_report(test_labels, test_preds_optimal, 
                              target_names=['no disease', 'disease']))
    
    # Save results
    results = {
        'train_logits': train_logits,
        'train_labels': train_labels,
        'val_logits': val_logits,
        'val_labels': val_labels,
        'test_logits': test_logits,
        'test_labels': test_labels,
        'optimal_threshold': optimal_threshold,
        'f1_scores': f1_scores,
        'cm_train': cm_train,
        'cm_val_optimal': cm_val_optimal,
        'cm_test_default': cm_test_default,
        'cm_test_optimal': cm_test_optimal,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc_optimal': val_acc_optimal,
        'test_loss': test_loss,
        'test_acc_optimal': test_acc_optimal,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
    }
    
    torch.save(results, output_dir / 'threshold_tuning_results.pt')
    
    # Plots
    print("\ngenerating plots...")
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accs,
        'val_accuracies': val_accs
    }
    plot_training_history(history, save_path=output_dir / 'training.png')
    plot_confusion_matrix(cm_train, title=f"train (accuracy: {train_acc:.4f})",
                         save_path=output_dir / 'cm_train.png')
    plot_confusion_matrix(cm_test_optimal,
                         title=f"test (accuracy: {test_acc_optimal:.4f}, threshold: {optimal_threshold:.4f})",
                         save_path=output_dir / 'cm_test_optimal.png')
    
    # Summary
    elapsed = time.time() - start_time
    
    
    print("training complete")
   
    print(f"training time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"\nfinal results (threshold: {optimal_threshold:.4f}):")
    print(f"  test accuracy: {test_acc_optimal*100:.2f}%")
    print(f"  test loss: {test_loss:.4f}")
    print(f"  disease detection: {test_preds_optimal.sum()}/{int(np.array(test_labels).sum())} cases")
    print(f"\nmodel saved to: {output_dir / 'best_model.pt'}")
    print(f"results saved to: {output_dir / 'threshold_tuning_results.pt'}")
   


if __name__ == "__main__":
    main()