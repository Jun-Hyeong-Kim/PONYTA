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

        
def rank_pr_genes(ko_output_path, pr_genes_path, deg_output_path, np_output_path, nx_network_path, deg_num, np_pos_genes_count, aggregation_method):

    # Read phenotype-related genes lists
    pr_genes = pd.read_csv(pr_genes_path, header=None)[0].to_list()
    pr_genes = [gene.strip() for gene in pr_genes]
    pr_genes_list = list(set(pr_genes))

    # Load STRING network
    g = nx.read_graphml(nx_network_path)
    nodes_list = list(g.nodes)

    # Read DEG output file to obtain positive genes from DEG analysis 
    deg_list = pd.read_csv(deg_output).index.to_list()
    deg_list = [deg.upper() for deg in deg_list]
    
    pos_deg = [gene for gene in deg_list if gene in nodes_list]
    
    if len(pos_deg) > deg_num:
        pos_deg = [gene for gene in deg_list if gene in nodes_list][:deg_num]

    # Read NP output file to obtain positive genes from NP
    np_output = pd.read_csv(np_output_path, index_col=0)
    np_output_genes_all = np_output['protein'].to_list()
    
    np_p_genes = []
    for gene in np_output_genes_all:

        # Does NOT count genes that already in positive genes set from DEG analysis 
        if gene in pos_deg:
            
            continue
            
        np_p_genes.append(gene)
    
        if len(np_p_genes) == int(np_pos_genes_count):
            
            break

    # Read aggregated gene list
    gene_rank = pd.read_csv(os.path.join(ko_output_path, f'gene_rank_{aggregation_method}.csv'))
    
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
    1. Phenotype-related genes rank within aggregated genes ranked output WITHOUT considering exception when Phenotype-related genes itself involved in DEG & NP positive genes 
    '''
    raw_df = pd.DataFrame()

    if aggregation_method in ['weight_dibra', 'linear_borda', 'majoritarian']:

        for pr_gene in pr_genes_list:
        
            if pr_gene not in nodes_list:
                continue
            elif pr_gene in gene_rank['Voter'].to_list():
                raw_gene_rank = gene_rank['Voter'].to_list().index(pr_gene)+1
            else:
                raw_gene_rank = '-'
            
            ### Rank within DEG analysis output ###
            if pr_gene in deg_list:
                deg_rank_from_all = deg_list.index(pr_gene) + 1
            else:
                deg_rank_from_all = '-'
        
            ### Rank within NP output ###
            np_rank_from_all = np_output_genes_all.index(pr_gene) + 1 
            
            raw_tmp = pd.DataFrame([{'Affected Gene':pr_gene.capitalize(), 'Raw Gene Rank':raw_gene_rank,
                                    'DEG Rank':deg_rank_from_all, 'NP Rank':np_rank_from_all}])
            raw_df = pd.concat([raw_df, raw_tmp], axis=0)
            
        # Convert the column to numeric, treating '-' as NaN
        raw_df['Raw Gene Rank'] = pd.to_numeric(raw_df['Raw Gene Rank'], errors='coerce')
        
        # Sort the DataFrame based on the 'your_column' values
        raw_df = raw_df.sort_values(by='Raw Gene Rank', na_position='last', ascending=True)
        raw_df['Raw Gene Rank'] = raw_df['Raw Gene Rank'].apply(convert_to_int)
        
    else:
    
        for pr_gene in pr_genes_list:
        
            if pr_gene not in nodes_list:
                continue
            elif pr_gene in gene_rank['gene'].to_list():
                raw_gene_rank = gene_rank['gene'].to_list().index(pr_gene)+1
            else:
                raw_gene_rank = '-'
            
            ### Rank within DEG analysis output ###
            if pr_gene in deg_list:
                deg_rank_from_all = deg_list.index(pr_gene) + 1
            else:
                deg_rank_from_all = '-'
        
            ### Rank within NP output ###
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
    2. Phenotype-related genes rank within aggregated genes ranked output considering exception when Phenotype-related genes itself involved in DEG & NP positive genes 
    '''

    if aggregation_method in ['weight_dibra', 'linear_borda', 'majoritarian']:

        non_degnp_rank = []
        
        for gene in gene_rank['Voter'].to_list():
            if gene in pos_deg or gene in np_p_genes:
                continue
            else:
                non_degnp_rank.append(gene)
                
        
        non_degnp_df = pd.DataFrame()

        non_degnp_rank = gene_rank[gene_rank['Voter'].isin(non_degnp_rank)]
        non_degnp_rank.reset_index(drop=True, inplace=True)
        non_degnp_rank.index = non_degnp_rank.index + 1

        
        for pr_gene in pr_genes_list:
            
            if pr_gene not in nodes_list:
                continue
                
            elif pr_gene in non_degnp_rank['Voter'].to_list():

                # Mean rank for genes with same score (rank)
                gene_rank = np.mean(non_degnp_rank[non_degnp_rank['Score']==non_degnp_rank[non_degnp_rank['Voter']==affected_gene]['Score'].item()].index)

                
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
        
        raw_df.to_csv(os.path.join(ko_output_path,f'raw_pr_genes_rank_{aggregation_method}.csv'), index=False)
        non_degnp_df.to_csv(os.path.join(ko_output_path,f'pr_genes_rank_{aggregation_method}.csv'), index=False)

    else:

        # when aggregation method is median aggregation
        
        non_degnp_rank = []
        
        for gene in gene_rank['gene'].to_list():
            if gene in pos_deg or gene in np_p_genes:
                continue
            else:
                non_degnp_rank.append(gene)
        
        non_degnp_df = pd.DataFrame()

        non_degnp_rank = gene_rank[gene_rank['gene'].isin(non_degnp_rank)]
        non_degnp_rank.reset_index(drop=True, inplace=True)
        non_degnp_rank.index = non_degnp_rank.index + 1
        
        for pr_gene in pr_genes_list:
            if pr_gene not in nodes_list:
                continue
                
            elif pr_gene in non_degnp_rank['gene'].to_list():

                # Mean rank for genes with same score (rank)
                gene_rank = np.mean(non_degnp_rank[non_degnp_rank['rank']==non_degnp_rank[non_degnp_rank['gene']==affected_gene]['rank'].item()].index)

                
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
        
        raw_df.to_csv(os.path.join(ko_output_path,'raw_pr_genes_rank_median.csv'), index=False)
        non_degnp_df.to_csv(os.path.join(ko_output_path,'pr_genes_rank_median.csv'), index=False)
