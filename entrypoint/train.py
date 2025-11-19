import sys
import os

# Get the absolute path of the directory containing this script (entrypoint)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project's root directory (one level up)
project_root = os.path.dirname(script_dir)
# Add the project root to Python's path
sys.path.insert(0, project_root)


import torch
import torch.nn.functional as F
from src.model import MultiTaskGNN  # Import the model 

# Configuration
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
EPOCHS = 300
HIDDEN_CHANNELS = 64
ATTENTION_HEADS = 8
DROPOUT_RATE = 0.6

def train(model, data, optimizer, criterion):
    """Performs a single training step."""
    model.train()
    optimizer.zero_grad()
    
    # Get model predictions
    pred_ad, pred_pd, pred_ftd, pred_als, _ = model(data)
    
    # Use the training mask to select the nodes for loss calculation
    train_mask = data.train_mask
    
    # Calculate loss for each task ONLY on the training nodes
    loss_ad = criterion(pred_ad[train_mask], data.y[train_mask, 0].unsqueeze(1))
    loss_pd = criterion(pred_pd[train_mask], data.y[train_mask, 1].unsqueeze(1))
    loss_ftd = criterion(pred_ftd[train_mask], data.y[train_mask, 2].unsqueeze(1))
    loss_als = criterion(pred_als[train_mask], data.y[train_mask, 3].unsqueeze(1))
    
    # Total loss is the sum of the individual task losses
    total_loss = loss_ad + loss_pd + loss_ftd + loss_als
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluates the model on a given data split (validation or test)."""
    model.eval()
    pred_ad, pred_pd, pred_ftd, pred_als, _ = model(data)
    
    # Calculate MSE loss for each task on the given mask
    loss_ad = F.mse_loss(pred_ad[mask], data.y[mask, 0].unsqueeze(1))
    loss_pd = F.mse_loss(pred_pd[mask], data.y[mask, 1].unsqueeze(1))
    loss_ftd = F.mse_loss(pred_ftd[mask], data.y[mask, 2].unsqueeze(1))
    loss_als = F.mse_loss(pred_als[mask], data.y[mask, 3].unsqueeze(1))
    
    total_val_loss = loss_ad + loss_pd + loss_ftd + loss_als
    return total_val_loss.item()


def main():
    """Main function to run the training and evaluation process."""
    print("Starting the GNN Training Process")
    
    #  2. Load Data 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = 'data/02-preprocessed/processed_graph.pt'
    data = torch.load(data_path)

    data = data.to(device)

    print("\nLoaded data object:")
    print(data)

    # 3. Initialize Model and Optimizer 
    model = MultiTaskGNN(
        in_channels=data.num_node_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=4, # This is fixed for our 4 tasks
        heads=ATTENTION_HEADS,
        dropout=DROPOUT_RATE
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.MSELoss() # Mean Squared Error is a good choice for regression
    
    print("\nModel architecture:")
    print(model)

    #  4. Training Loop 
    best_val_loss = float('inf')
    best_model_state = None

    print("\nStarting Training")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, data, optimizer, criterion)
        val_loss = evaluate(model, data, data.val_mask)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict() # Save the model weights
            print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}  (New best model!)")
        else:
            print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 5. Final Evaluation on Test Set 
    # Load the best performing model
    model.load_state_dict(best_model_state)
    
    test_loss = evaluate(model, data, data.test_mask)
    print("\n Training Complete")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    # 6. Save the Best Model 
    model_save_path = 'models/best_model.pt'
    torch.save(best_model_state, model_save_path)
    print(f"\nBest model weights saved to '{model_save_path}'")

if __name__ == '__main__':
    main()
