# CHEM 277B Final Project
# Topic: Predicting Neurodegenerative Disease Status From Proteomics Data Using an Artificial Neural Network (ANN) and a Graph Attention Network (GAT)
Contributors: Anna Devlen, Andrea Kim, Frank Qi, Yasemin Sucu, Leonard Wei

This repository contains code to predict neurodegenerative disease status using data from a large proteomics database maintained by the Global Neurodegenerative Proteomics Consortium (GNPC). Two models were trained for this purpose: a two-layer Artificial Neural Network (ANN) and a two-layer Graph Attention Network (GAT). The GAT incorporates a Protein-Protein Interaction (PPI) network built using the STRING database.

The repository contents are as follows:

**data:** The data folder contains the file `load.py` that generates a PyTorch Geometric Data object where data can be accessed for training and `uniprot_util` which contains helper functions to interact with UniProt's API. There is no need to run these files unless you want to change data filter parameters. Training data is provided in the Google Drive link at the bottom. 


**src:/ANN:** The ANN folder contains the file ANN.py, which defines the ANN model and performs training, hyperparameter tuning, and plotting. This file requires the download of the torch data object 'binary_data.pt' from the Google Drive link below.


**src:/GAT:** The GAT folder contains the implementation of a Graph Attention Network that incorporates protein-protein interactions from the STRING database. The folder includes:

- `gat_model.py`: Defines the GAT model architecture with two graph attention layers
- `train_gat.py`: Training script with threshold optimization on the validation set
- `dataset.py`: Data loading utilities for patient graphs
- `restructure_to_patient_graphs.py`: Script to construct patient-level graphs from the PPI network

The training script implements threshold optimization using F1-score maximization to improve disease detection while balancing overall accuracy. This approach achieves a 2.7-fold improvement in disease detection sensitivity compared to the ANN baseline (65.52% vs 24.1% recall).

To run the GAT training:
1. Download `patient_graphs_dataset` and `binary_label_data_fixed.pt` from the Google Drive link below
2. Place files in the data directory
3. Run: `python src/GAT/train_gat_proper.py`

The script outputs confusion matrices, training history plots, and comprehensive results including logits for threshold analysis.

Data files that were too large to upload to GitHub are available for download from Google Drive: [LINK](https://drive.google.com/drive/folders/1y6sxCVYQeF1sogsOQQJlNr-0LhlxfL2f?usp=sharing)
