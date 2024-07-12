import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm

import os
import sys

import networkx as nx
from sklearn.model_selection import train_test_split

# from rank_aggregation_func import rank_aggregation
import rank_aggregation_func
from rank_prgenes_func import convert_to_int, rank_pr_genes
from rank_eval_func import partial_roc, rank_evaluation



parser = argparse.ArgumentParser()

# rank aggregation
parser.add_argument('--ponyta_output', type=str, help='Path to PONYTA output folder')

# rank PR-genes
parser.add_argument('--pr_genes_path', type=str, help='Path to phenotype-related genes txt file')

parser.add_argument('--deg_output_path', type=str, help='DEG output path')
parser.add_argument('--np_output_path', type=str, help='NP output path')
parser.add_argument('--nx_network_path', type=str, help='Path to networkx file')

parser.add_argument('--np_pos_genes_count', type=int, help='number of np genes to use')

# rank evaluation
parser.add_argument('--n_auc_min', type=int, default=5, help='minimum value for n')
parser.add_argument('--n_auc_max', type=int, default=300, help='maximum value for n')


parser.add_argument('--agg_method', type=str, default='weight_dibra', choices=['weight_dibra', 'median', 'linear_borda', 'majoritarian'], help='type of aggregation method')


args = parser.parse_args()


# Rank Aggregation

ko_output = [folder for folder in os.listdir(args.ponyta_output) if 'npgenes' in folder][0]
ranks_output = os.path.join(args.ponyta_output, ko_output)


if args.agg_method in ['weight_dibra', 'linear_borda', 'majoritarian']:
    rank_aggregation.rank_aggregation(output_path = ranks_output, agg_method = args.agg_method)
elif args.agg_method == 'median':
    rank_aggregation.rank_aggregation_median(ranks_output)


# Ranks for Phenotype-related genes

rank_pr_genes(ranks_output, args.pr_genes_path, args.deg_output_path, args.np_output_path, args.nx_network_path, args.np_pos_genes_count)


# Rank Evaluation
rank_evaluation(ranks_output, np_pos_genes_count, args.n_auc_min, args.n_auc_max)