import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm

import os

import networkx as nx
from sklearn.model_selection import train_test_split

import sys


def convert_to_int(value):
    try:
        return str(int(float(value)))
    except ValueError:
        return '-'

        
def rank_pr_genes(ko_output_path, pr_genes_path, deg_output_path, np_output_path, nx_network_path, np_pos_genes_count):

    pr_genes = pd.read_csv(pr_genes_path, header=None)[0].to_list()
    pr_genes = [gene.strip() for gene in pr_genes]
    pr_genes_list = list(set(pr_genes))
    
    g = nx.read_graphml(nx_network_path)
    nodes_list = list(g.nodes)
    
        
    deg_list = pd.read_csv(deg_output).index.to_list()
    deg_list = [deg.upper() for deg in deg_list]
    
    pos_deg = [gene for gene in deg_list if gene in nodes_list]
    
    if len(pos_deg) > deg_num:
        pos_deg = [gene for gene in deg_list if gene in nodes_list][:deg_num]
    
    np_output = pd.read_csv(np_output_path, index_col=0)
    np_output_genes_all = np_output['protein'].to_list()
    # np_pos_genes_count = np_pos_genes_count
    
    np_p_genes = []
    for gene in np_output_genes_all:
    
        if gene in pos_deg:
            
            continue
            
        np_p_genes.append(gene)
    
        if len(np_p_genes) == int(np_pos_genes_count):
            
            break
    
    gene_rank = pd.read_csv(os.path.join(ko_output_path, 'gene_rank_avg.csv'))
    
    tmp_gene_rank = []
    for gene in gene_rank['gene'].to_list():
        if gene in pr_genes_list:
            tmp_gene_rank.append(gene)
            continue
        elif gene in pos_deg or gene in np_p_genes:
            continue
        else:
            tmp_gene_rank.append(gene)
            continue
    
    '''
    1. Phenotype-related genes rank within gene rank avg output WITHOUT considering exception when Phenotype-related genes itself involved in DEG & NP positive genes 
    '''
    raw_df = pd.DataFrame()
    
    for pr_gene in pr_genes_list:
    
        if pr_gene not in nodes_list:
            continue
        elif pr_gene in gene_rank['gene'].to_list():
            raw_gene_rank = gene_rank['gene'].to_list().index(pr_gene)+1
        else:
            raw_gene_rank = '-'
        
        ### Raw DEG 전체 내부에서 rank ###
        if pr_gene in deg_list:
            deg_rank_from_all = deg_list.index(pr_gene) + 1
        else:
            deg_rank_from_all = '-'
    
        ### NP 전체 rank 내부에서 Rank ###
        np_rank_from_all = np_output_genes_all.index(pr_gene) + 1 
        
        raw_tmp = pd.DataFrame([{'Affected Gene':pr_gene.capitalize(), 'Raw Gene Rank':raw_gene_rank,
                                'DEG Rank':deg_rank_from_all, 'NP Rank':np_rank_from_all}])
        raw_df = pd.concat([raw_df, raw_tmp], axis=0)
        
    # Convert the column to numeric, treating '-' as NaN
    raw_df['Raw Gene Rank'] = pd.to_numeric(raw_df['Raw Gene Rank'], errors='coerce')
    
    # Sort the DataFrame based on the 'your_column' values
    raw_df = raw_df.sort_values(by='Raw Gene Rank', na_position='last', ascending=True)
    raw_df['Raw Gene Rank'] = raw_df['Raw Gene Rank'].apply(convert_to_int)
    
    
    
    '''
    2. Phenotype-related genes rank within gene rank avg output considering exception when Phenotype-related genes itself involved in DEG & NP positive genes 
    '''
    
    non_degnp_rank = []
    
    for gene in gene_rank['gene'].to_list():
        if gene in pos_deg or gene in np_p_genes:
            continue
        else:
            non_degnp_rank.append(gene)
    
    non_degnp_df = pd.DataFrame()
    
    for pr_gene in pr_genes_list:
        if pr_gene not in nodes_list:
            continue
        elif pr_gene in non_degnp_rank:
            gene_rank = non_degnp_rank.index(pr_gene) + 1
            
            if pr_gene in deg_list:
                deg_rank_from_all = deg_list.index(pr_gene) + 1
            else:
                deg_rank_from_all = '-'
                
            np_rank_from_all = np_output_genes_all.index(pr_gene) + 1
    
            non_degnp_tmp = pd.DataFrame([{'Affected Gene':pr_gene.capitalize(), 'Gene Rank':gene_rank,
                                          'DEG Rank':deg_rank_from_all,'NP Rank':np_rank_from_all}])
            non_degnp_df = pd.concat([non_degnp_df, non_degnp_tmp], axis=0)
        else:
            continue
    
    try:
        # Convert the column to numeric, treating '-' as NaN
        non_degnp_df['Gene Rank'] = pd.to_numeric(non_degnp_df['Gene Rank'], errors='coerce')
        
        # Sort the DataFrame based on the 'your_column' values
        non_degnp_df = non_degnp_df.sort_values(by='Gene Rank', na_position='last', ascending=True)
        non_degnp_df['Gene Rank'] = non_degnp_df['Gene Rank'].apply(convert_to_int)
    except:
        pass
    
    raw_df.to_csv(os.path.join(ko_output_path,'raw_pr_genes_rank_avg.csv'), index=False)
    non_degnp_df.to_csv(os.path.join(ko_output_path,'pr_genes_rank_avg.csv'), index=False)
