# PONYTA
PONYTA: prioritization of phenotype-related genes from mouse KO events using PU learning on a biological network

<img width="1226" alt="image" src="https://github.com/user-attachments/assets/31120b78-f071-4d85-a86a-99bef05cd055">

## Environment
Install `conda` environment

```
conda env create -f ponyta.yaml
```

## Usage

- `main.py` : Run PU learning on network for gene prioritization. For each iteration of positive-unlabeled (PU) learning, it outputs prioritized gene ranks output for each fold.
  ```
  python main.py --ko_gene KO_gene_name --deg_output_file /path/to/DEGoutput --np_output /path/to/NPoutput --deg_num --deg_num 100 --np_num 50 --nx_network /path/to/Networkxfile --csv_network /path/to/CSVfile
  ```
  - `--ko_gene` : Name of knock-out (KO) gene. 
  - `--deg_output_file` : Path to differentially expressed genes (DEGs) analysis output file for KO gene. DEG output considered as already sorted as adjusted p-value ascending order, and DEGs are in first column.
  - `--np_output_file` : Path to network propagation (NP) output file to used. If it is not provided (default : None), it will automatically perform network propagation and use its output.
  - `--deg_num` : Number of DEGs to use as positive genes for PU learning. (default : 100)
  - `--np_num` : Number of NP genes to use as positive genes for PU learning. (default : 50)
  - `--nx_network` : Path to Networkx network file. Wil used for PU learning.
  - `--csv_network` : Path to CSV adjacency matrix. Will used for Network Propagation.

  - `--gnn_type` : Type of graph neural network (GNN) to use. GAT(Graph attention network) for default. Otherwise, either GCN (Graph convolution network) and GIN (Graph isomorphism network) can be utilized.
  - `--gnn_epoch` : Number of GNN iteration for each iteration of PU learning. It can be early-stopped depends on value of `early_stopping_epochs` designated. (default : 50)
  - `--train_epoch`: Number of iterations for training PU learning on graph. (default : 30)
  - `--iter` : Number of total iteration of PONYTA. (default : 10)
  - `--num_folds` : Number of folds for cross-validation. (default : 5)
  - `--hidden_channels` : Dimension of node embedding during GNN training. (default : 256)
 
- `rank_main.py` : Run processing on rank outputs obtained from `main.py`. It first aggregated gene rank outputs using average rank aggregation. Aggregated gene rank subsequently used for extracting gene ranks for phenotype-related genes, considering as exceptional case when phenotype-related genes involved in positive genes used for PU learning which is top-ranked subset genes from either DEG analysis output (`deg_output_file`) and NP output (`np_output_file`). Finally, partial AUC value considering top _n_ negative genes ($AUC_n$) is evaluated to quantify ability of the algorithm on prioritization of phenotype-related genes.
  ```
  python rank_main.py --ponyta_output /path/to/PONYTAoutput --pr_genes_path /path/to/txtfile --deg_output_path /path/to/DEGoutput --np_output_path /path/to/NPoutput --nx_network_path /path/to/Networkxfile --np_pos_genes_count 50 --n_auc_min 5 --n_auc_max 300 --agg_method weight_dibra
  ```
  - `--ponyta_output` : Path to PONYTA output folder. The output folder should have output folders from each iteration.
  - `--pr_genes_path` : Path to phenotype-related genes _txt_ file. The _txt_ file considered to have vertical list of phenotype-related gene names, with each gene name on a separate line.
  - `--deg_output_path` : Path to differentially expressed genes (DEGs) analysis output file for KO gene. DEG output considered as already sorted as adjusted p-value ascending order, and DEGs are in first column.
  - `--np_output_path` : Path to network propagation (NP) output file to used.
  - `--nx_network_path` : Path to Networkx file.
  - `--np_pos_genes_count` : Number of top-ranked NP genes used for positive genes.
  - `--n_auc_min` : Minimum value for _n_, number of top-ranked negative genes to consider for partial AUC value calculation. (default : 5)
  - `--n_auc_max` : Maximum value for _n_, number of top-ranked negative genes to consider for partial AUC value calculation. (default : 300)
  - `--agg_method`: Rank aggregation method to use. One of Weighted dibra, Median rank aggregation, Linear combination with Borda normalization and Majoritarian method with Condorcet Winners method can be given. (default : weight_dibra)
