# CHEM 277B Final Project
# Topic: Predicting Neurodegenerative Disease Status From Proteomics Data Using an Artificial Neural Network (ANN) and a Graph Attention Network (GAT)
Contributors: Anna Devlen, Andrea Kim, Frank Qi, Yasemin Sucu, Leonard Wei

This repository contains code to predict neurodegenerative disease status using data from a large proteomics database maintained by the Global Neurodegenerative Proteomics Consortium (GNPC). Two models were trained for this purpose: a two-layer Artificial Neural Network (ANN) and a two-layer Graph Attention Network (GAT). The GAT incorporates a Protein-Protein Interaction (PPI) network built using the STRING database.

The repository contents are as follows:

**data:** The data folder contains the file `load.py` that generates a PyTorch Geometric Data object where data can be accessed for training and `uniprot_util` which contains helper functions to interact with UniProt's API. There is no need to run these files unless you want to change data filter parameters. Training data is provided in the Google Drive link at the bottom. 

**entrypoint:**

**models:**

**notebooks:**

**plots:**

**src:** The ANN folder contains the file ANN.py, which defines the ANN model and performs training, hyperparameter tuning, and plotting. This file requires the download of the torch data object 'binary_data.pt' from the Google Drive link below.

Data files that were too large to upload to GitHub are available for download from Google Drive: [LINK](https://drive.google.com/file/d/1nlMvfXWzfrRaQgN2R3YtEnHS7OiGsCJb/view?usp=sharing)
