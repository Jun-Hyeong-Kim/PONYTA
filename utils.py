import os
import time
import math
import copy
import random
import numpy as np
import pickle
import networkx as nx

import torch
from torch.linalg import norm
from torch import nn
from torch.nn import Linear
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

# from bpn.graph import Graph
from graph import Graph
# from bpn.networks import LBP, GCN, GCN_Net
# from networks import LBP, GCN, GCN_Net
from networks import LBP
# from bpn.loss_function import Soft_NLL_Loss

from sklearn.metrics import pairwise

def seed_everything(seed=73):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) #####
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 



def init_weights(m):
    if isinstance(m, Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
        
def to_LBP_graph(data, device):
    
    temp_nodes = data.x
    edge_ls = []
    for i in range(data.edge_index.shape[1]):
        if data.edge_index[0][i] < data.edge_index[1][i]:
            edge_ls.append([int(data.edge_index[0][i]), int(data.edge_index[1][i])])
    temp_graph = Graph(temp_nodes, np.array(edge_ls)).to(device)
    
    return temp_graph

def p_label_vectorize(positive_node_index):
    return torch.zeros(len(positive_node_index), dtype=torch.int64)

def n_label_vectorize(negative_node_index):
    return torch.ones(len(negative_node_index), dtype=torch.int64)

def load_CC_data(file_dir, space_name, seed_num):
    seed_everything(seed=seed_num)
    ## load adjacency matrix and node feature
    ad = np.load('adjacency.npy')
    ft = torch.tensor(np.load('feature_matrix.npy'))
    ad = ad + np.eye(ft.shape[0])
    G = nx.from_numpy_matrix(ad)
    A = nx.adjacency_matrix(G)
    edge_index, edge_attr = dense_to_sparse(torch.tensor(A.todense()))
    data = Data(x=ft, edge_index=edge_index, edge_attr=edge_attr)
    input_shape = data.x.shape[1]
    
    approved_node_idx = list(range(219, ft.shape[0]))
    withdrawn_node_idx = list(range(219))
    # select half of withdrawn node as spy node (treat as unlabeled node)
    spy_node_idx = random.sample(withdrawn_node_idx, round(len(withdrawn_node_idx)/2))
    withdrawn_node_idx = list(set(withdrawn_node_idx).difference(spy_node_idx))
    # Make unlabeled node index with half of withdrawn node and whole approved node
    unlabeled_node_idx = approved_node_idx + spy_node_idx
    
    test_spy_node_idx = random.sample(spy_node_idx, round(len(spy_node_idx)/2))
    val_spy_node_idx = list(set(spy_node_idx).difference(test_spy_node_idx))
    with open('true_negative_node_index.pkl', 'rb') as f:
        tn_node_idx = pickle.load(f)
    test_tn_node_idx = random.sample(tn_node_idx, round(len(tn_node_idx)/2))
    val_tn_node_idx = list(set(tn_node_idx).difference(test_tn_node_idx))
    return data, input_shape, torch.tensor(withdrawn_node_idx), torch.tensor(unlabeled_node_idx), torch.tensor(val_spy_node_idx), torch.tensor(val_tn_node_idx), torch.tensor(test_spy_node_idx), torch.tensor(test_tn_node_idx)
        
def net_generate(model, 
                 features, 
                 net_method, 
                 knn_num, 
                 net_save_dir=None, 
                 seed_num=None):
    model.eval()
    output = model(features)
    c = pairwise.cosine_similarity(output.cpu().detach().numpy())
    if net_method == 'threshold':
        thr = np.sort(c.flatten())[::-1][:int(len(c.flatten())*0.05)][-1]
        ad = (c >= thr)*1
        np.fill_diagonal(ad, 0)
    elif net_method == 'knn':
        np.fill_diagonal(c, 0)
        
        def k_n_graph(row):
            thr = np.sort(row)[::-1][:knn_num][-1]
            return (row >= thr)*1
        
        ad = np.apply_along_axis(k_n_graph, 1, c)
        
    if save_dir != None:
        save_dir = '../../Results/GBPU_graph/'+save_dir
        isExist = os.path.exists(save_dir)
        if not isExist:
            os.makedirs(save_dir)
            print("The new directory is created!")
            
        np.save(save_dir+'/adjacency_'+str(iter_num)+'.npy', ad)
        np.save(save_dir+'/node_features_'+str(iter_num)+'.npy', output.cpu().detach().numpy())
        return None
    else:
        ad = ad + np.eye(2163)
        G = nx.from_numpy_matrix(ad)
        A = nx.adjacency_matrix(G)
        edge_index, edge_attr = dense_to_sparse(torch.tensor(A.todense()))
        data = Data(x=output, edge_index=edge_index, edge_attr=edge_attr)
        return data, ad

def generate_index_set(seed_num):
    seed_everything(seed=seed_num)
  
    approved_idx = list(range(219, 2163))
    withdrawn_idx = list(range(219))
    # select half of withdrawn node as spy node (treat as unlabeled node)
    spy_idx = random.sample(withdrawn_idx, round(len(withdrawn_idx)/2))
    withdrawn_idx = list(set(withdrawn_idx).difference(spy_idx))
    # Make unlabeled node index with half of withdrawn node and whole approved node
    unlabeled_idx = approved_idx + spy_idx
    
    test_spy_idx = random.sample(spy_idx, round(len(spy_idx)/2))
    val_spy_idx = list(set(spy_idx).difference(test_spy_idx))
    with open('/data/project/joebrother/pu_learning/CC_data/true_negative_node_index.pkl', 'rb') as f:
        tn_idx = pickle.load(f)
    test_tn_idx = random.sample(tn_idx, round(len(tn_idx)/2))
    val_tn_idx = list(set(tn_idx).difference(test_tn_idx))
    approved_idx = list(set(approved_idx).difference(tn_idx))
    return torch.tensor(withdrawn_idx), torch.tensor(unlabeled_idx), torch.tensor(val_spy_idx), torch.tensor(val_tn_idx), torch.tensor(test_spy_idx), torch.tensor(test_tn_idx), torch.tensor(approved_idx)

        
def cosine_similarity(tensor_a, tensor_b):
    c = torch.dot(tensor_a,tensor_b)/(torch.linalg.norm(tensor_a)*torch.linalg.norm(tensor_b))
    return c


def get_loss(output, beliefs, p_true_labels, p_node_index, u_node_index, device, training = True):


    p_true = torch.zeros(p_node_index.shape[0])
    p_true = p_true.long()
    p_true = p_true.to(device)

    # pred tensor for positive nodes
    
    p_pred = output[p_node_index]

    # Belief matrix values for unlabeled idx (for training)
    u_belief = beliefs[u_node_index]
    u_belief = torch.argmax(u_belief, dim=1)

    u_pred = output[u_node_index]


    if training == False:

        p_true = p_true.detach().cpu()
        u_belief = u_belief.detach().cpu()
    
    p_loss = F.nll_loss(p_pred, p_true)
    u_loss = F.nll_loss(u_pred, u_belief)

    loss = p_loss + u_loss
    
    return loss

def get_val_loss(output, beliefs, p_true_labels, p_node_index, n_node_index, u_node_index, device, training=True):
    
    p_true = torch.zeros(p_node_index.shape[0])
    p_true = p_true.long()
    p_true = p_true.to(device)
    
    p_pred = output[p_node_index]
    
    n_true = torch.ones(n_node_index.shape[0])
    n_true = n_true.long()
    n_true = n_true.to(device)
    
    n_pred = output[n_node_index]
    
    if training == False:
        
        p_pred = p_pred.detach().cpu()
        p_true = p_true.detach().cpu()
    
    pos_loss = F.nll_loss(p_pred, p_true)
    neg_loss = F.nll_loss(n_pred, n_true)
    
    return pos_loss + neg_loss
    
    