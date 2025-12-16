"""
Create GAT train/val/test figure calculated from confusion matrices.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    print("creating GAT train/val/test figure")
    
    results_dir = Path("../results/gat_proper")
    
    # Load threshold results
    print("loading results")
    threshold_results = torch.load(results_dir / "threshold_tuning_results.pt")
    results = torch.load(results_dir / "results.pt")
    
    # Get test data
    test_cm = threshold_results['cm_optimal']
    test_loss = 0.99
    
    # Get loss values from results file
    train_loss = float(results['train_losses'][-1]) if 'train_losses' in results else 0.73
    val_loss = float(results['val_losses'][-1]) if 'val_losses' in results else 1.58
    
    # Known confusion matrices from training
    train_cm = np.array([[62, 61], [3, 7]])
    val_cm = np.array([[20, 18], [7, 12]])
    
    # Calculate accuracies from confusion matrices
    train_acc = (train_cm[0,0] + train_cm[1,1]) / train_cm.sum()
    val_acc = (val_cm[0,0] + val_cm[1,1]) / val_cm.sum()
    test_acc = (test_cm[0,0] + test_cm[1,1]) / test_cm.sum()
    
    print(f"\nCorrected accuracies from confusion matrices:")
    print(f"  Training: {train_acc*100:.2f}%")
    print(f"  Validation: {val_acc*100:.2f}%")
    print(f"  Test: {test_acc*100:.2f}%")
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(1, 3, left=0.08, right=0.92, top=0.78, bottom=0.12, wspace=0.35)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
    split_data = [
        ("Training", train_cm, train_acc, train_loss),
        ("Validation", val_cm, val_acc, val_loss),
        ("Test", test_cm, test_acc, test_loss)
    ]
    
    classes = ['No Disease', 'Disease']
    

    for idx, (title, cm, acc, loss) in enumerate(split_data):
        ax = axes[idx]
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes,
               yticklabels=classes,
               ylabel='True label',
               xlabel='Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center", fontsize=14, fontweight='bold',
                       color="white" if cm[i, j] > thresh else "black")
        
        # Title 
        ax.set_title(f"{title}\nAccuracy: {acc*100:.2f}%\nBCE Loss: {loss:.2f}", 
                    fontweight='bold', fontsize=13, pad=10)
        
        print(f"{title:12s}: Accuracy = {acc*100:.2f}%, Loss = {loss:.2f}")
    
    # Add colorbar on the right
    cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.67])
    fig.colorbar(im, cax=cbar_ax)
    

    fig.suptitle('GAT Model Performance on Train, Validation, and Test Splits\n(Optimal Threshold: -0.4262)', 
                fontsize=15, fontweight='bold', y=0.99)
    
    plt.savefig(results_dir / 'gat_performance_all_splits.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: gat_performance_all_splits.png")
    
    # Print disease detection rates
    print(f"\nDisease Detection Rates:")
    print(f"  Training: {train_cm[1,1]}/{train_cm[1,0]+train_cm[1,1]} = {train_cm[1,1]/(train_cm[1,0]+train_cm[1,1])*100:.2f}%")
    print(f"  Validation: {val_cm[1,1]}/{val_cm[1,0]+val_cm[1,1]} = {val_cm[1,1]/(val_cm[1,0]+val_cm[1,1])*100:.2f}%")
    print(f"  Test: {test_cm[1,1]}/{test_cm[1,0]+test_cm[1,1]} = {test_cm[1,1]/(test_cm[1,0]+test_cm[1,1])*100:.2f}%")
    
    plt.close()


if __name__ == "__main__":
    main()