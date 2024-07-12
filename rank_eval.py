import pandas as pd
import numpy as np

import os

from tqdm import tqdm

import sys

def partial_roc(rank_output_path, n):
    
    # n: negative_gene_nums_to_consider
    
    rank = pd.read_csv(rank_output_path)
    T = rank.shape[0]
    
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


def rank_evaluation(ko_output_path, np_pos_genes_count, n_auc_min, n_auc_max):
    
    output = os.path.join(ko_output_path, 'pr_genes_rank_avg.csv')
    
    
    rank_eval_dict = {'Model': 'PULSAR',
                      'NP Genes #':np_pos_genes_count}
    
    for n in range(n_auc_mim, n_auc_max, 1):
        rank_eval_dict[f'Partial AUC ({n})'] = partial_roc(output, n=n)
    
    rank_eval_df = pd.DataFrame([rank_eval_dict])
    
    
    save_dir_path = ko_output_path
    
        
    rank_eval_df.to_csv(os.path.join(save_dir_path,f'partial_AUC_range{n_auc_min}_{n_auc_max}.csv'), index=False)
