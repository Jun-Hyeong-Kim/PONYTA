import os
import sys
import math
import copy
import pickle
import numpy as np
import networkx as nx    
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

import argparse

import torch
from torch import nn
from torch.nn import Linear, ReLU, Module, BCELoss
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, dense_to_sparse
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.nn import GCNConv, GINConv, GATv2Conv, global_mean_pool, global_add_pool
from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU, 
                      Sequential, BatchNorm1d as BN)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
from graph import Graph
from networks import LBP, GCN, GIN, GAT
from utils import seed_everything, to_LBP_graph, init_weights, get_loss, get_val_loss




def train(model, data, beliefs, p_true_labels, p_node_index, u_node_index, p_val_node_index, n_val_node_index, num_epoches, optimizer, device):     

    best_loss = 10000000
    early_stop_counter = 0
    early_stopping_epochs = args.early_stopping_epochs
    
    for epoch in range(num_epoches):
        
        model.train()
        output = model(data)
        output = F.log_softmax(output, dim=1)
        
        loss = get_loss(output, beliefs, p_true_labels, p_node_index, u_node_index, device)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        
        with torch.no_grad():
            output_val = model(data)
            valid_loss = get_val_loss(output, beliefs, p_true_labels, p_val_node_index, n_val_node_index, u_node_index, device, training=False)
            
        if valid_loss > best_loss:
            early_stop_counter += 1
            
        else:
            best_loss = valid_loss
            model_state_file = os.path.join(output_dir, f'model_state_fold{fold_num+1}.csv')
            torch.save(model.state_dict(), model_state_file)
            early_stop_counter = 0

        if early_stop_counter >= early_stopping_epochs:
            print(f'Early Stopping at gnn {epoch} epoch')
            break
            
    return loss.item(), model_state_file
    
@torch.no_grad()
def test(model, data, beliefs, p_true_labels, val_p_idx, val_n_idx, device):
    
    model.eval()
    test_output = model(data).detach().cpu()
    test_output = F.log_softmax(test_output, dim=1)
    
    test_loss = get_loss(test_output, beliefs, p_true_labels, val_p_idx, val_n_idx, device, training=False)
    
    accuracy, precision, recall, f1 = metric(model, data, val_p_idx, val_n_idx)
    accuracy_pos = pos_metric(model, data, val_p_idx)
    accuracy_neg = neg_metric(model, data, val_n_idx)
    
    return test_loss.item(), accuracy, precision, recall, f1, accuracy_pos, accuracy_neg, test_output
    
@torch.no_grad()
def calculate_new_prior(model, data, u_node):
    
    model.eval() 
    output = model(data)[u_node].detach().cpu()
    output_softmax = F.softmax(output, dim=1)
    pred = output_softmax[:, 0] > 0.9
    
    return sum(pred.numpy()) / len(u_node)

def metric(model, data, val_p_idx, val_n_idx):


    val_idx = torch.cat((val_p_idx, val_n_idx), dim=0)
    
    val_final = model(data)[val_idx].detach().cpu()
    val_final = F.log_softmax(val_final, dim=1)
    val_final = torch.argmax(val_final, dim=1)

    val_p_true = torch.zeros(val_p_idx.shape[0])
    val_n_true = torch.ones(val_n_idx.shape[0])
    val_true = torch.cat((val_p_true, val_n_true), dim=0)

    accuracy = accuracy_score(val_true, val_final)
    precision = precision_score(val_true, val_final)
    recall = recall_score(val_true, val_final)
    f1 = f1_score(val_true, val_final)

    return accuracy, precision, recall, f1

def pos_metric(model, data, val_p_idx):
    
    val_p_final = model(data)[val_p_idx].detach().cpu()
    val_p_final = F.log_softmax(val_p_final, dim=1)
    val_p_final = torch.argmax(val_p_final, dim=1)

    val_p_true = torch.zeros(val_p_idx.shape[0], dtype=torch.int64)

    accuracy_pos = accuracy_score(val_p_true, val_p_final)

    return accuracy_pos

def neg_metric(model, data, val_n_idx):
    
    val_n_final = model(data)[val_n_idx].detach().cpu()
    val_n_final = F.log_softmax(val_n_final, dim=1)
    val_n_final = torch.argmax(val_n_final, dim=1)

    val_n_true = torch.ones(val_n_idx.shape[0], dtype=torch.int64)

    accuracy_neg = accuracy_score(val_n_true, val_n_final)

    return accuracy_neg
    
def exponential_function(x):
    return np.exp(x)
    
def run_PU(data, output_dir, hidden_channels, train_p_idx, train_u_idx, val_p_idx, val_n_idx, train_epoch, seed_num=42, gpu=0):


    device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
    
    ## Set learning hyperparameters
    num_GCN_epoches = gnn_epoch
    
    lbp_graph = to_LBP_graph(data, device)
    
    p_true_labels = pos_indices.to(device)

    lbp = LBP(lbp_graph, train_p_idx, num_states=2, device=device) 

    # Define node embeddings 
    node_embeds = torch.nn.Embedding(len(lbp.graph.features), hidden_channels).to(device)

    if gnn_type == 'GCN':
        model = GCN(hidden_channels = hidden_channels, num_layers = num_layers, node_embedding = node_embeds, dropout=dropout)
    elif gnn_type == 'GIN':
        model = GIN(hidden_channels = hidden_channels, num_layers = num_layers, node_embedding = node_embeds, dropout=dropout)
    elif gnn_type == 'GAT':
        model = GAT(hidden_channels = hidden_channels, node_embedding=node_embeds, heads=8, dropout=dropout)
    else:
        print(f"GNN could not import successfully")
        sys.exit()       
    
    model.apply(init_weights)
    model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_count = 0
    loss = 100000000.0
    test_loss = 100000000.0
    test_accuracy = 0.0
    train_loss_list = []
    test_loss_list = []

    metric_output = pd.DataFrame()
    
    for _ in range(train_epoch):
        
        if train_count == 0:
            prior = 0

        beliefs = lbp(prior, THRESHOLD=1e-6, EPSILON=1e-10) 
        model_saved_path = None
        
        train_new_loss, model_saved_path = train(model, data, beliefs, p_true_labels, train_p_idx, train_u_idx, val_p_idx, val_n_idx, num_GCN_epoches, optimizer, device)

        model.load_state_dict(torch.load(model_saved_path))
        prior = calculate_new_prior(model, data, train_u_idx)
        test_loss, accuracy, precision, recall, f1, accuracy_pos, accuracy_neg, test_eval_output = test(model, data, beliefs, p_true_labels, val_p_idx, val_n_idx, device)

        test_accuracy = accuracy_pos
        train_old_loss = train_new_loss

        log = 'Epoch: {:03d}, prior: {:.2f}, old loss: {:.4f}, new loss: {:.4f}, test loss: {:.4f}'

        print(log.format(train_count, prior, train_old_loss, train_new_loss, test_loss))
        
        train_loss_list.append(train_new_loss)
        test_loss_list.append(test_loss)

        tmp = {'accuracy_all':accuracy,
              'precision_all':precision,
              'recall_all':recall,
              'f1_all':f1,
              'accuracy_positive':accuracy_pos,
              'accuracy_negative':accuracy_neg}

        tmp = pd.DataFrame(list(tmp.values()), index=tmp.keys(), columns=[f'iteration_{train_count+1}'])
        
        metric_output = pd.concat([metric_output, tmp], axis=1)
        
        train_count += 1

    
    final_beliefs = beliefs


    
    test_eval_label = torch.argmax(test_eval_output, dim=1)
    
    val_true_degs = [nodes_list[idx] for idx in val_deg_p_idx]
    val_true_nps = [nodes_list[idx] for idx in val_np_p_idx]

    val_pred_degs = [nodes_list[idx] for idx in val_deg_p_idx if test_eval_label[idx] == 0]
    val_pred_nps = [nodes_list[idx] for idx in val_np_p_idx if test_eval_label[idx] == 0]

    sorted_tensor, indices = torch.sort(test_eval_output[:,0], descending=True)
    
    rank_nodes = [nodes_list[idx] for idx in indices]
    rank_nodes_val = [test_eval_output[idx][0] for idx in indices]
    
    rank_df = pd.DataFrame()
    for i in range(len(rank_nodes)):
        tmp = pd.DataFrame([{'gene':rank_nodes[i], 'pos_val':float(rank_nodes_val[i])}])
        rank_df = pd.concat([rank_df, tmp], axis=0)
    

    
    return metric_output, prior, final_beliefs, train_loss_list, test_loss_list, val_true_degs, val_true_nps, val_pred_degs, val_pred_nps, rank_df



parser = argparse.ArgumentParser(description='PU learning')
parser.add_argument('--gpu', type=str, help = 'gpu')
parser.add_argument('--early_stopping_epochs', type=int, default=5, help='Early stopping epochs to stop trainning on GNN')
parser.add_argument('--seed_num', type=int, default=42, help='seed num')

parser.add_argument('--iter', type=int, default=10, help= 'Number of iterations of the algorithm')

parser.add_argument('--num_folds', type=int, default=5, help = 'Number of folds for cross validation')

parser.add_argument('--ko_gene', type=str, default=None, help= 'KO gene name')


parser.add_argument('--train_epoch', type=int, default=30, help='Number of iterations for training PU learning on graph')
parser.add_argument('--hidden_channels', type=int, default=256, help='Dimension of node embedding during GNN training')
parser.add_argument('--num_layers', type=int, default=2, help='Number of GNN layers in case using GNN types other than GAT')
parser.add_argument('--gnn_epoch', type=int, default=50, help='Number of GNN iteration for each iteration of PU learning')
parser.add_argument('--nx_network', type=str, help='Path to Networkx network. Will used for PU learning')
parser.add_argument('--csv_network', type=str, help='Path to CSV adjacency matrix. Will used for Network Propagation')

parser.add_argument('--deg_output_file', type=str, help='Path to DEG output file for KO gene. DEG output considered as already sorted as adjusted p-value ascending order, and DEGs are in first column')
parser.add_argument('--np_output_file', type=str, default = None, help = 'NP output to use. If it is not provided, it will automatically perform network propagation and use its output')
parser.add_argument('--deg_num', type=int, default=50, help='Number of DEG to use as positive genes')
parser.add_argument('--np_num', type=int, default=50, help='Number of NP genes to use as positive genes')
parser.add_argument('--gnn_type', type=str, default='GAT', help='Type of GNN to use')

parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate during training')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout portion during GNN training')


args = parser.parse_args()

# Reading hyperparameter inputs
np_output_file = args.np_output_file
train_epoch = args.train_epoch
seed_num = args.seed_num
gpu = args.gpu
deg_num = args.deg_num
np_pos_genes_count = args.np_num
hidden_channels = args.hidden_channels
num_layers = args.num_layers
gnn_epoch = args.gnn_epoch
gnn_type = args.gnn_type
lr = args.lr
dropout = args.dropout


if args.ko_gene != None:

    ko_gene = args.ko_gene
    ko_gene = ko_gene.upper()
    
else:
    print('Enter KO gene name')
    sys.exit()

print(f'KO gene name: {ko_gene}')


net = args.nx_network
grn_csv = args.csv_network


seed_everything(seed=seed_num)

g = nx.read_graphml(net)
nodes_list = list(g.nodes)


deg = args.deg_output_file



# Read in DEG output file
deg_list = pd.read_csv(deg, index_col=0).index.to_list()
deg_list = [deg.upper() for deg in deg_list]



# Using only top-ranked DEGs
pos_indices = torch.tensor([nodes_list.index(gene) for gene in deg_list if gene in nodes_list])

origin_pos_len = pos_indices.shape[0]

if pos_indices.shape[0] > deg_num:
    pos_indices = torch.tensor([nodes_list.index(gene) for gene in deg_list if gene in nodes_list][:deg_num])
    print(f'Slice pos_indices tensor to length of {pos_indices.shape[0]}')
else:
    print(f'pos_indices tensor length remain same since oringal length is {pos_indices.shape[0]}')


    
### NX to PYG for use in PU learning
no_data_graph = nx.Graph()
no_data_graph.add_nodes_from(g.nodes())
no_data_graph.add_edges_from(g.edges(data=True))

data = from_networkx(no_data_graph)
data.x = torch.arange(data.num_nodes)


for iteration in range(args.iter):


    dir_name = f'iteration_{iteration}'

    
    if origin_pos_len > deg_num:
        
        output_dir = os.path.join(f'./output/', f'{ko_gene}_degnum{deg_num}_npgenes{np_pos_genes_count}', dir_name)
        
    else:
        
        output_dir = os.path.join(f'./output/', f'{ko_gene}_degnum{pos_indices.shape[0]}_npgenes{np_pos_genes_count}', dir_name)
        
    os.makedirs(output_dir, exist_ok=True)

    # Save hyperparameter output
    hyperparameter_output_path = os.path.join(output_dir, 'hyperparameter.txt')

    with open(hyperparameter_output_path, 'w') as file:

        file.write(f"DEG_output_path: {deg}\n")
        file.write(f"NX_input_path: {net}\n")
        file.write(f"CSV_input_path: {grn_csv}\n")
        file.write(f"KO Gene: {ko_gene}\n")
        file.write(f"Hidden Channels: {hidden_channels}\n")
        file.write(f"Num Layers: {num_layers}\n")
        file.write(f"GNN Epoch: {gnn_epoch}\n")
        file.write(f"NP Positive Genes Count: {np_pos_genes_count}\n")
        file.write(f"GNN Type: {gnn_type}\n")
        file.write(f"Learning Rate: {lr}\n")
        file.write(f"Dropout: {dropout}\n")
        if np_output_file != None:
            file.write(f"NP Output: {np_output_file}\n")
        file.write(f"Train Epoch: {train_epoch}\n")
        file.write(f"Seed Num: {seed_num}\n")
        file.write(f'DEG_input #: {pos_indices.shape[0]}\n')
        file.write(f'Early stopping epoches #: {args.early_stopping_epochs}\n')

    
    '''
    Network Propagation using DEGs as seed nodes
    '''    
    
    print('Preparing NP input...')

    if np_output_file == None:
    
        print('Performing NP...')
        
        # Run network propagation as positive degs
        import run_NP as netprop
                
        # Using KO gene and positive DEG genes as seed genes
        seed_degs = [nodes_list[i] for i in pos_indices.tolist()]
        np_output = netprop.main_propagation(grn_csv, ko_gene, seed_degs)
        np_output_path = os.path.join(output_dir, f'{ko_gene}_np_output.csv')
        np_output.to_csv(np_output_path)
        
        with open(hyperparameter_output_path, 'a') as file:
            file.write(f"NP Output: {np_output_path}\n")
            
        np_output_file = np_output_path
    
    # '''
    # For later iterations than the first, NP output file obtained from the first iterations is used
    # OR, in case NP output file provided in first place, it uses the NP output directly rather than performing Network propagation
    # '''    
    
    else:
    
        np_output = pd.read_csv(np_output_file)
    
        with open(hyperparameter_output_path, 'a') as file:
            file.write(f"NP Output: {np_output_file}\n")
            
        print(f'Successfully read in NP output for {ko_gene} to use')


    # Using only top-ranked NP genes as positive genes

    np_output_genes = np_output['protein'].to_list()
    seed_degs = [nodes_list[i] for i in pos_indices.tolist()]

    np_p_genes = []
        
    for gene in np_output_genes:

        # Exclude gene when gene already in DEG positive genes
        if gene in seed_degs:
            continue
            
        np_p_genes.append(gene)

        if len(np_p_genes) == np_pos_genes_count:
            break      


    '''
    For n Fold Cross Validation, make n different cross validation train/test indicies set
    ** DEG split and NP input split independently and concatenated for usage **
    '''    

    num_folds = args.num_folds

    # Initialize StratifiedKFold

    stratified_kfold_deg = StratifiedKFold(n_splits=num_folds, shuffle=True)
    stratified_kfold_np = StratifiedKFold(n_splits=num_folds, shuffle=True)
    
    splits_deg = []
    splits_np = []
    
    # Iterate through the folds deg
    for fold, (train_deg_index, val_deg_index) in enumerate(stratified_kfold_deg.split(pos_indices, torch.zeros_like(pos_indices))):
        data_train_deg, data_val_deg = pos_indices[train_deg_index], pos_indices[val_deg_index]
        splits_deg.append((data_train_deg, data_val_deg))

    # Iterate through the folds np
    all_np_p_idx = torch.tensor([index for index, gene in enumerate(nodes_list)
                                if gene in np_p_genes])
    for fold, (train_np_index, val_np_index) in enumerate(stratified_kfold_np.split(all_np_p_idx, torch.zeros_like(all_np_p_idx))):
        data_train_np, data_val_np = all_np_p_idx[train_np_index], all_np_p_idx[val_np_index]
        splits_np.append((data_train_np, data_val_np))

    # Combine each train, test folds for deg and np
    i = 0
    splits = []
    for sp_deg in splits_deg:
        sp_train = torch.cat((sp_deg[0], splits_np[i][0]))
        sp_val = torch.cat((sp_deg[1], splits_np[i][1]))
        splits.append((sp_train, sp_val))
        i += 1

    
    '''
    Iterate for each fold
    '''

    for fold_num in range(num_folds):

        
        print(f'{iteration}th iteration - ko gene: {ko_gene}, np genes : {np_pos_genes_count}, fold: {fold_num}')


        ### For usage in metric calculation, declare each variable ###
        train_deg_p_idx = splits_deg[fold_num][0]
        val_deg_p_idx = splits_deg[fold_num][1]
        train_np_p_idx = splits_np[fold_num][0]
        val_np_p_idx = splits_np[fold_num][1]
        
        train_p_idx = splits[fold_num][0]
        train_u_idx = torch.tensor([index for index,gene in enumerate(nodes_list)
                                   if index not in train_p_idx])
        
        val_p_idx =  splits[fold_num][1]
        val_n_idx = torch.tensor([index for index, gene in enumerate(nodes_list)
                                 if index not in train_p_idx and index not in val_p_idx])
          
        
        metric_output, prior, final_beliefs, train_loss_list, test_loss_list, test_true_degs, test_true_nps, test_pred_degs, test_pred_nps, rank_df = run_PU(data, output_dir, hidden_channels, train_p_idx, train_u_idx, val_p_idx, val_n_idx, train_epoch = train_epoch, seed_num = seed_num, gpu=gpu)

        # Save final belief matrix as csv file
        
        belief_file_path = os.path.join(output_dir, f'belif_matrix_fold{fold_num+1}.csv')
        beliefs_cpu = final_beliefs.cpu()
        beliefs_df = pd.DataFrame(beliefs_cpu.numpy(), columns=['positive', 'negative'])
        beliefs_df = beliefs_df.round(3)
        beliefs_df.T.to_csv(belief_file_path)

        
        # Save metric dataframe as csv file
        
        metric_file_path = os.path.join(output_dir, f'metric_output_fold{fold_num+1}.csv')
        metric_output.loc['train_loss'] = train_loss_list
        metric_output.loc['val_loss'] = test_loss_list
        metric_output = metric_output.round(3)
        metric_output.T.to_csv(metric_file_path)

        # Save predicted genes ranking csv file
                
        rank_path = os.path.join(output_dir, f'gene_rank_fold{fold_num+1}.csv')
        rank_df['pos_val'] = rank_df['pos_val'].apply(exponential_function)
        rank_df = rank_df.rename(columns={'pos_val': 'pos_prob'})
        rank_df.to_csv(rank_path, index=False)

        
