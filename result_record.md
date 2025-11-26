Training done!
Best val loss: 0.0606
Test loss: 0.0613

Train metrics (scaled space):
  AD: MSE=0.0134, R²=0.9866
  PD: MSE=0.0148, R²=0.9852
  FTD: MSE=0.0117, R²=0.9883
  ALS: MSE=0.0059, R²=0.9941

Val metrics (scaled space):
  AD: MSE=0.0287, R²=0.9744
  PD: MSE=0.0146, R²=0.9853
  FTD: MSE=0.0113, R²=0.9891
  ALS: MSE=0.0080, R²=0.9926

Test metrics (scaled space):
  AD: MSE=0.0241, R²=0.9789
  PD: MSE=0.0180, R²=0.9827
  FTD: MSE=0.0133, R²=0.9865
  ALS: MSE=0.0058, R²=0.9941
Baseline (AD) – MSE=1.1436, R²=-0.0007
Baseline (PD) – MSE=1.0460, R²=-0.0018
Baseline (FTD) – MSE=0.9851, R²=-0.0014
Baseline (ALS) – MSE=0.9944, R²=-0.0000
Saved run-specific model to models/run_20251125_163527/best_model.pt
Also updated models/best_model.pt for downstream analysis


Trying different clustering parameters
  size= 15: 79 clusters, silhouette=0.5820 (id=min15_cfg0)
  size= 25: 37 clusters, silhouette=0.3059 (id=min25_cfg1)
  size= 35: 16 clusters, silhouette=0.3149 (id=min35_cfg2)
  size= 50:  2 clusters, silhouette=0.3627 (id=min50_cfg3)
  size= 75:  2 clusters, silhouette=0.3627 (id=min75_cfg4)
  size=100:  2 clusters, silhouette=0.3910 (id=min100_cfg5)

All HDBSCAN configs sorted by silhouette:
     config_id  min_cluster_size  n_clusters  silhouette
0   min15_cfg0                15          79    0.581998
5  min100_cfg5               100           2    0.390968
3   min50_cfg3                50           2    0.362714
4   min75_cfg4                75           2    0.362714
2   min35_cfg2                35          16    0.314874
1   min25_cfg1                25          37    0.305908
Saved silhouette summary plot to plots/run_20251125_171032/hdbscan_silhouette_by_min_cluster_size.png

Generating detailed outputs for top 6 HDBSCAN configs
