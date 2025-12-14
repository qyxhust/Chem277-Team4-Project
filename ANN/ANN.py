import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sigmoid, Softmax, BCELoss,  Dropout
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools

np.random.seed(42)
torch.manual_seed(42)

# get dataset
def get_tensors(filepath, device): 
    '''
    Creates x and y tensors based on GNPC data labels for each disease and protein abundance values.

    Parameters
    ----------
    filepath : str
        Location of cleaned data CSV.

    Returns
    -------
    data_obj : PyTorch Data Object
    '''
    # load in data object
    data = torch.load(filepath, map_location=device, weights_only=False)
    x_np = data.x.cpu().numpy() # features
    y_np = data.y.cpu().numpy() # labels (binary)
    y_re = y_np.reshape(-1, 1) # reshape to 2D to match input size

    # create training, validation, and testing masks (70/15/15 split)
    N = x_np.shape[0]
    split_labels = np.random.choice([0, 1, 2], N, p = [0.7, 0.15, 0.15]) # 0 is train, 1 is val, 2 is test

    train_mask_np = (split_labels == 0) # need np array to fit x_train

    train_mask = torch.tensor(train_mask_np, dtype=torch.bool).to(device)
    val_mask = torch.tensor(split_labels==1, dtype=torch.bool).to(device)
    test_mask = torch.tensor(split_labels==2, dtype=torch.bool).to(device)

    # fit only training data
    scaler = StandardScaler()
    x_train = x_np[train_mask_np]
    scaler.fit(x_train)
    
    # scale all features
    x_scaled = scaler.transform(x_np)

    # address class imbalance - get weight for BCE loss
    '''Citation: Tantai H. Use Weighted Loss Function to Solve Imbalanced Data Classification Problems. 
    Medium. Published February 27, 2023. Accessed December 13, 2025. 
    https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75 '''
    disease = y_np.sum()
    no_disease = len(y_np) - disease

    weight_class0 = len(y_np) / (no_disease * 2) 
    weight_class1 = len(y_np) / (disease * 2)

    weight = torch.tensor([weight_class1]).to(device) # only pass in weight for majority class

    # create x and y tensors
    x = torch.tensor(x_scaled, dtype=torch.float32).to(device)
    y = torch.tensor(y_re, dtype=torch.long).to(device)

    # create data object
    data_obj = Data(
        x=x,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask, 
        weight = weight
    )

    return data_obj

class my_ANN(torch.nn.Module):

    def __init__(self, num_features, num_neurons, dropout=0.5):
        '''
        Initializes an instances of the my_ANN class.

        Parameters
        ----------
        num_features : int
            Number of features in the feature tensor (X).
        n_neurons : int
            Number of dimensions in the space that input features are transformed to.
        '''
        super(my_ANN, self).__init__()

        self.num_features = num_features
        self.num_neurons = num_neurons
        
        self.dense1 = Linear(self.num_features, self.num_neurons)
        self.relu = ReLU()
        self.dropout = Dropout(p=dropout) # add if needed
        self.dense2 = Linear(self.num_neurons, 1)
        #self.sigmoid = Sigmoid()

    def forward(self, x): # lr: start w 0.001
        '''
        Forward pass for the 2-layer ANN. 

        Parameters
        ----------
        x : PyTorch tensor
            Features tensor.

        Returns
        -------
        y_prob : Pytorch tensors
            Disease status probability, between 0 and 1.
        '''        

        x1 = self.dense1(x)
        x2 = self.relu(x1)
        x3 = self.dropout(x2)
        x4 = self.dense2(x3)

        return x4
       
class FitModel():
    def __init__(self, my_model, learning_rate: float = 0.01, weight_decay: float = 1e-4):
        '''
        Initializes an instance of the FitModel class. 

        Parameters
        ----------
        my_model : an instance of the my_ANN class

        learning_rate : float
            Hyperparameter that sets the step size for updating model weights during training.
        '''
        
        # Adam for optimization
        self.optimizer = torch.optim.Adam(my_model.parameters(), lr = learning_rate, weight_decay=weight_decay)
        self.model = my_model

    def get_scaled_BCE(self, dataset, mask):
        '''
        Calculates the binary cross entry loss for the relevant data split.

        Parameters
        ----------
        dataset : PyTorch Data Object
            Data object containing features and labels.

        mask : PyTorch tensor
            Datapoints used for training, testing, or validation.

        Returns
        -------
        BCE_loss : PyTorch tensor
            The binary cross entropy loss.

        Citation for last 2 lines: Tantai H. Use Weighted Loss Function to Solve Imbalanced Data Classification Problems. 
        Medium. Published February 27, 2023. Accessed December 13, 2025. 
        https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75
        
        '''

        x = dataset.x.to(device)
        out1 = self.model(x) 
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight = dataset.weight) # logits loss incorporates sigmoid activation
        BCE_loss = criterion(out1[mask], dataset.y[mask].float()) # calculate BCE loss
        return BCE_loss

    def get_metrics(self, dataset, mask):
        ''' 
        Calculates BCE loss and accuracy for the relevant data split..

        Parameters
        ----------
        dataset : PyTorch Data Object
            Data object containing features and labels.

        mask : PyTorch tensor
            Datapoints used for training, testing, or validation.
        '''

        out1 = self.model(dataset.x) # get predictions

        # calculate BCE
        loss = self.get_scaled_BCE(dataset, mask)

        # get probabilities (output of sigmoid)
        y_true = dataset.y[mask].float()
        y_pred = out1[mask]

        # get class (all values > 0.5 are converted to True, all < 0.5 converted to False for y_true)
        # y_pred is raw logits - use threshold of 0.0
        y_true_binary = (y_true > 0.5).float()
        y_pred_binary = (y_pred > 0.0).float()

        # calculate accuracy
        accuracy = (y_pred_binary == y_true_binary).sum() / y_true_binary.size()[0]

        # create dictionary to return
        metrics = {
            'loss' : loss.item(), 
            'accuracy' : accuracy.item(),
            'y_true' : y_true_binary.cpu().numpy(), # move back to CPU and convert to NumPy for confusion matrix plotting
            'y_pred' : y_pred_binary.cpu().numpy()
        }

        return metrics

    def run(self, dataset, N_epochs : int = 100):
        '''
        Runs the training loop, validation, and testing on the best model identified by validation loss.

        Parameters
        ----------
        dataset : PyTorch Data Object
            Data object containing features and labels.

        N_epochs : int
            Number of times to run the training loop. 

        Returns
        -------
        loss : float
            Training loss for the final epoch.
        best_val_loss : float
            Best validation loss found over N_epochs.

        Citation for early stopping: Cyborg. What is Early Stopping in Deep Learning?. 
        Medium (2024). Published April 5, 2024. 
        https://cyborgcodes.medium.com/what-is-early-stopping-in-deep-learning-eeb1e710a3cf
        '''

        best_val_loss = float('inf')

        # add the below for early stopping to prevent overfitting
        stop_after = 10
        no_improvement = 0

        # intialize lists so we can plot MSE over epochs
        train_losses = []
        val_losses = []
        epoch_nums = []

        # initialize dictionaries to store metrics for best model
        best_train_metrics = {}
        best_val_metrics = {}

        for epoch in range(N_epochs):

            # Zero gradients
            self.model.train()
            self.optimizer.zero_grad()

            # Compute loss
            loss = self.get_scaled_BCE(dataset, dataset.train_mask)

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # get loss and accuracy
            train_metrics = self.get_metrics(dataset, dataset.train_mask)

            with torch.no_grad():
                # get metrics (validation) 
                val_metrics = self.get_metrics(dataset, dataset.val_mask)
                
                # get val loss 
                val_loss = self.get_scaled_BCE(dataset, dataset.val_mask)

                # add losses & current epoch to lists
                train_losses.append(loss.item())
                val_losses.append(val_loss.item())
                epoch_nums.append(epoch)

                # compare current loss to best loss
                if val_loss.item() < best_val_loss:
                    no_improvement = 0
                    best_val_loss = val_loss
                    best_train_metrics = train_metrics
                    best_val_metrics = val_metrics

                    # save best model
                    torch.save(self.model.state_dict(), 'best_ANN.pt')

                    print(f"Epoch {epoch:03d} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f} * NEW BEST")
                else:
                    no_improvement += 1
                    print(f"Epoch {epoch:03d} | Train: {loss.item():.4f} | Val: {val_loss.item():.4f}")

                if no_improvement >= stop_after:
                    break

        # get metrics (test)

        # load best model
        self.model.load_state_dict(torch.load('best_ANN.pt'))

        with torch.no_grad():
            self.model.eval()

            best_train_metrics = self.get_metrics(dataset, dataset.train_mask)
            best_val_metrics = self.get_metrics(dataset, dataset.val_mask)
            test_metrics = self.get_metrics(dataset, dataset.test_mask)
            
            test_loss = self.get_scaled_BCE(dataset, dataset.test_mask)

        print(f"Best val loss: {best_val_loss.item()}")   
        print(f"Test loss for best model: {test_loss.item()}")

        # print metrics for all splits 
        print("Training Metrics:")
        print(best_train_metrics)

        print("Val Metrics:")
        print(best_val_metrics)

        print("Test Metrics:")
        print(test_metrics)

        # plot train and val mse over epochs
        self.plot(train_losses, val_losses, epoch_nums)

        # plot confusion matrices for each split for best model
        self.plot_confusion(best_train_metrics, best_val_metrics, test_metrics)

        return loss.item(), best_val_loss
        
    def hp_tuning(self, num_features, dataset):
        '''
        Performs hyperparameter tuning for the ANN by iterating through combinations of learning rate and number of neurons.
        
        Parameters
        ----------

        num_features : int
            Number of features in the feature tensor (X).

        dataset : PyTorch Data Object
            Data object containing features and labels.

        Returns
        -------
        best_params : dictonary
            Combination of parameters that produced the lowest validation loss.

        Citation: Van den Berg, T. Parameter Grid-searching with Python's itertools. SITMO Machine Learning | Quantitative Finance (2020). 
        Published December 29, 2020. https://www.sitmo.com/grid-searching-for-optimal-hyperparameters-with-itertools/
        '''

        # define parameter space (learning rate, dropout rate, n_neurons)

        param_grid = {
            'learning_rate' : [0.001, 0.0005, 0.0001], 
            'n_neurons' : [8, 16, 32],
            'dropout_rate' : [0.3, 0.5, 0.7], 
            'weight_decay' : [5e-3, 1e-3, 5e-4]
            }

        best_loss = float('inf') # initialize with value of +infinity
        best_params = None

        for lr, n_neurons, dr, wd in itertools.product(param_grid['learning_rate'], param_grid['n_neurons'], 
                                                   param_grid['dropout_rate'], param_grid['weight_decay']): 

            # initialize model instance to push to GPU
            model = my_ANN(num_features=num_features, num_neurons=n_neurons, dropout=dr)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # initialize FitModel instance and push to GPU
            hp_tuner = FitModel(model, learning_rate=lr, weight_decay=wd)

            # get validation losses
            losses = hp_tuner.run(dataset, N_epochs=50)
            current_loss_t = losses[1]
            current_loss = current_loss_t.item()

            # track best validation loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = {'lr' : lr, 'n_neurons' : n_neurons, 'dropout_rate' : dr, 'weight_decay' : wd} 
        print(f'Best params: {best_params}')
        print(f'Best val loss: {best_loss}')

        return best_params
    
    def plot(self, train_losses, val_losses, epoch_nums):
        '''
        Plots MSE loss over epochs for training and validation.

        Parameters
        ----------
        train_losses : list
            List containing training losses for all epochs.
        val_losses : list
            List containing validation losses for all epochs.
        epoch_nums : list
            List containing epoch numbers. 
        '''

        plt.figure(figsize=(8,6))
        plt.plot(epoch_nums, train_losses, label = 'Training BCE Loss')
        plt.plot(epoch_nums, val_losses, label = 'Validation BCE Loss')

        plt.title("Loss Over Epochs for ANN", fontsize=18)
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("BCE Loss", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.yscale("log")
        plt.legend()
        plt.savefig('loss_over_epochs.png')

    def plot_confusion(self, train_metrics, val_metrics, test_metrics):
        '''
        Plots confusion matrices for training, validation, and test sets.

        Parameters
        ----------
        train_metrics : dict
            Loss, accuracy, predictions, and true labels for training data.

        val metrics : dict
            Loss, accuracy, predictions, and true labels for validation data.

        test_metrics : dict
            Loss, accuracy, predictions, and true labels for test data.
        '''
        all_metrics = {
            'Training' : train_metrics,
            'Validation' : val_metrics,
            'Test' : test_metrics
        }

        fig, axs = plt.subplots(1, 3, figsize=(15,5))

        for i, (split, metric) in enumerate(all_metrics.items()):
            accuracy = metric['accuracy']
            loss = metric['loss']
            cm = confusion_matrix(metric['y_true'], metric['y_pred'])
            disp = ConfusionMatrixDisplay(cm, display_labels=['No Disease', 'Disease'])
            disp.plot(cmap=plt.cm.Blues, ax=axs[i])
            axs[i].set_title(f'{split}\nAccuracy : {(accuracy*100):.2f}%\nBCE Loss: {loss:.2f}', fontsize=14)
            axs[i].set_xlabel(axs[i].get_xlabel(), fontsize=14)
            axs[i].set_ylabel(axs[i].get_ylabel(),fontsize=14)
            axs[i].tick_params(axis = 'x', labelsize=12)
            axs[i].tick_params(axis = 'y', labelsize=12)

        plt.tight_layout()
        plt.savefig('confusion_matrices.png')

if __name__ == '__main__':

    # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = get_tensors('/home/devle/chem_277B/Chem277-Team4-Project/ANN/binary_label_data.pt', device)

    # run model
    my_model = my_ANN(num_features=data.x.shape[1], num_neurons=16, dropout=0.5) # based on hyperparameter search
    my_model = my_model.to(device)
    My_Fit = FitModel(my_model, learning_rate=0.001, weight_decay=0.005) # based on hyperparameter search
    My_Fit.run(data)
    #My_Fit.hp_tuning(num_features=data.x.shape[1], dataset=data)