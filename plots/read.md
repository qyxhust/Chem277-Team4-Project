In this branch, I made some minor modifications to the GAT model. The original model masked the entire node, but I modified it to mask at the feature level, and examined the reasoning and reconstruction results of the masked features in the node.

You can see the model's training results and training process in the image.(loss_curve / train_result)

1.Following the downstream task, I compared the differences in dimensionality reduction using UMAP between the original data and the data reconstructed by GAT. The data reconstructed by GAT showed a more concentrated distribution by disease type compared to the metadata. This characteristic reflects that the original data only modeled through data correlation, while GAT also learned the topological structure from the string library. The more concentrated clustering also reflects better classification performance (not yet validated by the model).
(cluster_gnn / cluster_raw)

2.We also compared the data obtained from gat inference with the original data and selected the top 10 with the largest differences. These proteins are likely related to the disease. Among them, ACHE and VAT1 were indeed proven to be related to AD. (However, I think these models might be more convincing if they were initially trained using protein concentrations rather than protein statistics, which is also the direction I will take to improve the models in the future.)

3.The model is compatible with inputs of different data shapes. Due to its built-in inference capabilities, it has some compatibility with missing data dimensions for data of varying shapes. The data we obtained from https://proteome-phenome-atlas.com/ only includes AD and PD dimensions, lacking protein data. We can use this data to import into the model. (However, excessive missing data may lead to poor inference performance.)
(inference/imputation_comparision)