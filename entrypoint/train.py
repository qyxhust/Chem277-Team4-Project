import sys
import os
import copy
import matplotlib.pyplot as plt

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
LEARNING_RATE = 0.002
WEIGHT_DECAY = 5e-4
EPOCHS = 300
HIDDEN_CHANNELS = 128
ATTENTION_HEADS = 16
DROPOUT_RATE = 0.6
MASK_RATE = 0.5

def apply_mask(data, mask_rate):
    """Applies a mask to the data."""
    num_nodes, num_features = data.size()
    mask = torch.rand(num_nodes, num_features,device=data.device) < mask_rate

    x_masked = data.clone()
    x_masked[mask] = 0

    return x_masked, mask

def train(model, data, optimizer, criterion):
    """Performs a single training step."""
    model.train()
    optimizer.zero_grad()
    
    # all nodes are masked and take part in the training
    x_masked, mask_matrix = apply_mask(data.x, MASK_RATE)
    
    # construct the batch
    original_x = data.x
    data.x= x_masked

    # reconstruct the feature 
    reconstructed_x = model(data)

    data.x = original_x

    #compute the loss
    loss = criterion(reconstructed_x[mask_matrix], original_x[mask_matrix])

    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def evaluate(model, data, criterion):
    """Evaluates the model on a given data split (validation or test)."""
    model.eval()

    x_maksed, mask_matrix = apply_mask(data.x, MASK_RATE)

    original_x = data.x
    data.x = x_maksed
    reconstructed_x = model(data)
    data.x = original_x

    val_loss = criterion(reconstructed_x[mask_matrix], original_x[mask_matrix])

    return val_loss.item()


def main():
    """Main function to run the training and evaluation process."""
    print("Starting the self_supervised GNN Training Process")
    
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
        out_channels=data.num_node_features, # This is fixed for our 4 tasks
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
    
    # Lists to store loss values
    train_losses = []
    val_losses = []

    print("\nStarting Training")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, data, optimizer, criterion)
        val_loss = evaluate(model, data, criterion)
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict() # Save the model weights
            print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}  (New best model!)")
        else:
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Plotting the loss curve
    print("\nPlotting loss curve...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('Self-Supervised GNN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print("Loss curve saved to 'loss_curve.png'")

    # 5. Final Evaluation on Test Set 
    # Load the best performing model
    model.load_state_dict(best_model_state)
    
    # Re-initialize criterion to be safe (in case it was overwritten, though unlikely here)
    criterion = torch.nn.MSELoss()
    
    test_loss = evaluate(model, data, criterion)
    print("\n Training Complete")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    # 6. Save the Best Model 
    model_save_path = 'models/best_model.pt'
    torch.save(best_model_state, model_save_path)
    print(f"\nBest model weights saved to '{model_save_path}'")

if __name__ == '__main__':
    main()
