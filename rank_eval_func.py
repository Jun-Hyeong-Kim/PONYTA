import pandas as pd
import numpy as np

import os

from tqdm import tqdm

import sys

def partial_roc(rank_output_path, n, T):

    
    try:
        rank = pd.read_csv(rank_output_path)
    except:
        ### In case NO Affected genes output are left ####
        rank = pd.DataFrame([{'Affected Gene':'-',
                            'Gene Rank':9999999999999,
                            'DEG Rank':99999999999,
                            'NP Rank':9999999999}])

    if max(pos_gene_ranks) < n: 
        pos_gene_ranks.append(9999999999)
    
    rank = pd.read_csv(rank_output_path)
    
    pos_gene_ranks = rank['Gene Rank'].to_list()
    
    neg_rank = 1
    neg_gene_ranks = []
    for pos_gene_rank in pos_gene_ranks:
        while neg_rank < pos_gene_rank:
            neg_gene_ranks.append(neg_rank)
            neg_rank+=1
            if len(neg_gene_ranks) == n:
                break
        if len(neg_gene_ranks) == n:
            break
        ## when i == pos_gene_rank
        neg_rank+=1
        
    sigTi = 0
    Ti = 0
    pos_rank_idx = 0
    Ti_list = []
    for neg_rank in neg_gene_ranks:
        Ti = len([rank for rank in pos_gene_ranks if rank < neg_rank])
        Ti_list.append(Ti)
        sigTi += Ti
    return sigTi / (n * T)


def rank_evaluation(ko_output_path, np_pos_genes_count, n_auc_min, n_auc_max, pr_genes_path, nx_network_path, aggregation_method):

    
    output = os.path.join(ko_output_path, f'pr_genes_rank_{aggregation_method}.csv')
    
    
    rank_eval_dict = {'Model': 'PONYTA',
                      'NP Genes #':np_pos_genes_count}

    # Load STRING network to count phenotype-related genes number within the network for partial AUC calculation
    g = nx.read_graphml(nx_network_path)
    nodes_list = set(list(g.nodes))

    pr_genes = pd.read_csv(pr_genes_path, header=None)[0].to_list()
    pr_genes = [gene.strip().upper() for gene in pr_genes]
    pr_genes = set(pr_genes)
    pr_genes_in_nx = nodes_list.intersection(pr_genes)
    
    total_num_pr = len(pr_genes_in_nx)
    
    for n in range(n_auc_mim, n_auc_max, 1):
        rank_eval_dict[f'Partial AUC ({n})'] = partial_roc(output, n = n, T = total_num_pr)
    
    rank_eval_df = pd.DataFrame([rank_eval_dict])
    
        
    rank_eval_df.to_csv(os.path.join(ko_output_path,f'partial_AUC_range{n_auc_min}_{n_auc_max}_{aggregation_method}.csv'), index=False)