import sys
import numpy as np
import networkx as nx
import argparse
import sklearn.preprocessing
from scipy.stats import spearmanr
import pandas as pd
import os
from tqdm import tqdm

CONV_THRESHOLD = 0.000001

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")

def isNum(x):
        try:
                float(x)
                return True
        except:
                return False
            
            
class Walker:
        """ Class for multi-graph walk to convergence, using matrix computation.

        Random walk with restart (RWR) algorithm adapted from:

        Kohler S, Bauer S, Horn D, Robinson PN. Walking the interactome for
        prioritization of candidate disease genes. The American Journal of Human
        Genetics. 2008 Apr 11;82(4):949-58.

        Attributes:
        -----------
                og_matrix (np.array) : The column-normalized adjacency matrix
                                                           representing the original graph LCC, with no
                                                           nodes removed
                tsg_matrix (np.array): The column-normalized adjacency matrix
                                                           representing the tissue-specific graph LCC, with
                                                           unexpressed nodes removed as specified by
                                                           low_list.
                restart_prob (float) : The probability of restarting from the source
                                                           node for each step in run_path (i.e. r in the
                                                           original Kohler paper RWR formulation)
                og_prob (float)          : The probability of walking on the original graph
                                                           for nodes that are expressed (so, we walk on the
                                                           TSG with probability 1 - og_prob)
                normalize (bool)         : Whether normalizing p0 to [0,1]
        """
        def __init__(self, original_ppi, low_list=[], remove_nodes=[], constantWeight=False, absWeight=False, addBidirectionEdge=False):
            self._build_matrices(original_ppi, low_list, remove_nodes, constantWeight, absWeight, addBidirectionEdge)
            self.dic_node2idx = dict([(node,i) for i, node in enumerate(self.OG.nodes())])
        
        
        def run_exp(self, seed2weight, restart_prob, og_prob=None, normalize=False, node_list=[]):
            # NP for one sample
            '''Run a multi-graph random walk experiment, and print results.
            
            Parameters:
            -----------
                    seed2weight (dictionary): The source node indices (i.e. a list of Entrez gene IDs)
                    
                    restart_prob (flost): As above
                    og_prob (float): As above
                    normalize (bool): As above
            '''
            self.restart_prob = restart_prob
            self.og_prob = og_prob
            
            # set up the starting probability vector
            criteria_p = self._set_up_p0(seed2weight)
            
            # mask TG with 0
            p_0 = self._set_up_p0(seed2weight)
            if normalize == True:
                p_0 /= np.sum(p_0) # normalize
            diff_norm = 1
            # this needs to be a deep copy, since we're reusing p_0 later
            p_t = np.copy(p_0)
            
            # arr_p includes all p_t for tracing
            arr_p = np.empty((len(p_t), 1))
            arr_p[:,0] = p_t
            
            while (diff_norm > CONV_THRESHOLD):
                    # first, calculate p^(t + 1) from p^(t)
                    p_t_1 = self._calculate_next_p(p_t, p_0)
                    if normalize == True:
                            p_t_1 /= np.sum(p_t_1) # normalize
                    # calculate L1 norm of difference between p^(t + 1) and p^(t),
                    # for checking the convergence condition
                    diff_norm = np.linalg.norm(np.subtract(p_t_1, p_t), 1)
                    # then, set p^(t) = p^(t + 1), and loop again if necessary
                    # no deep copy necessary here, we're just renaming p
                    p_t = p_t_1
                    # append p_t to arr_p
                    arr_p = np.c_[arr_p, p_t]
                    if arr_p.shape[1] >= 50000:
                            break
            print('%d iterated'%(arr_p.shape[1]))
            # now, generate and print a rank list from the final prob vector
            if node_list:#if I want to get propagation result only from selected node list
                    gene_idx = dict(zip(self.OG.nodes(), range(len(self.OG.nodes()))))
                    output = []
                    for node in node_list:
                            i = gene_idx[node]
                            output.append([node,arr_p[i,-1],arr_p[i,:].tolist()])
                    return output
            else:
                    gene_idx = dict(zip(self.OG.nodes(), range(len(self.OG.nodes()))))
                    output = []
                    for node in sorted(self.OG.nodes()):
                            i = gene_idx[node]
                            output.append([node,arr_p[i,-1],arr_p[i,:].tolist()])
                    return output

        def _generate_prob_list(self, p_t, node_list):
                gene_probs = dict(zip(self.OG.nodes(), p_t.tolist()))
                for node in node_list:
                        yield node, gene_probs[node]
                        
        def _generate_rank_list(self, p_t):
                """ Return a rank list, generated from the final probability vector.

                Gene rank list is ordered from highest to lowest probability.
                """
                gene_probs = zip(self.OG.nodes(), p_t.tolist())

                for s in sorted(gene_probs, key=lambda x: x[0]):
                        yield s[0], s[1]


        def _calculate_next_p(self, p_t, p_0):
                """ Calculate the next probability vector. """
                if self.tsg_matrix is not None:
                        no_epsilon = np.squeeze(np.asarray(np.dot(self.tsg_matrix, p_t) *
                                                                        (1 - self.og_prob)))
                        epsilon = np.squeeze(np.asarray(np.dot(self.og_matrix, p_t) *
                                                                          (self.og_prob)))
                        no_restart = np.add(epsilon, no_epsilon) * (1 - self.restart_prob)
                else:
                        epsilon = np.squeeze(np.asarray(np.dot(self.og_matrix, p_t)))
                        no_restart = epsilon * (1 - self.restart_prob)
                restart = p_0 * self.restart_prob
                return np.add(no_restart, restart)
            
        def _set_up_p0(self, seed2weight, set_TF=None):
            '''Set up and return the 0th probability vector'''
            
            p_0 = [0]*self.OG.number_of_nodes()
            weightSum = 0.0
            for seed, weight in seed2weight.items():
                if seed not in self.dic_node2idx:
                    continue
                weightSum += seed2weight[seed]
            for seed, weight in seed2weight.items():
                if set_TF != None:
                    if seed not in set_TF:
                        continue
                if seed not in self.dic_node2idx:
                    continue
                idx = self.dic_node2idx[seed]
                p_0[idx] = seed2weight[seed]
            return np.array(p_0)
        
        
        
        
        
        def _build_matrices(self, original_ppi, low_list, remove_nodes, constantWeight, absWeight, addBidirectionEdge):
            '''Build column-normalized adjacecny matrix for each graph.
            
            NOTE: there are column-normalized adjacecny matrices (not nx graphs), used to compute each p-vector
            '''
            original_graph = self._build_og(original_ppi, constantWeight, absWeight, addBidirectionEdge)
            
            if remove_nodes: 
                original_graph.remove_nodes_from(remove_nodes)
                original_graph.max(nx.connected_component_subgraphs(original_graph),
                                  key = len)
            
            self.OG = original_graph
            og_not_normalized = nx.to_numpy_array(original_graph)
            self.og_matrix = self._normalize_cols(np.transpose(og_not_normalized))
            
            self.tsg_matrix = None
            
            
        
        
        def _build_og(self, original_ppi, constantWeight=False, absWeight=False, addBidirectionEdge=False):
            '''Build the original graph, without any nodes removed'''
            

            G=nx.DiGraph()
            edge_list=[]
            

            print(original_ppi)
            i=0

                
            edge_list = [(row[0], row[1], float(row[2])) for row in original_ppi.itertuples(index=False)]

            G.add_weighted_edges_from(edge_list)
            
            return G
        

        def _normalize_cols(self, matrix):
            '''Normalize the columns of the adjacecny matrix'''
            return sklearn.preprocessing.normalize(matrix, norm='l1', axis=0)

def network_reconstruction(grn):
    '''
    Input: GRN network pandas dataframe,
    Output: 3 columns form (source gene, target gene, weight)
    '''
    grn_3c = grn.stack().reset_index()
    grn_3c.columns=['source','target','weight']
    grn_3c = grn_3c[grn_3c['weight']>0]
    
    return grn_3c

def main_propagation(grn, ko_gene, seed_degs=None, restart_prob=0.1, constantWeight=False, absoluteWeight=False, addBidirectionEdge=False, normalize=False):

    grn = pd.read_csv(grn, sep=',').fillna(0)
    grn.set_index(grn.columns[0], inplace=True)

    if seed_degs != None:
                
        if ko_gene not in seed_degs:
            
            seeds = [ko_gene]
            seeds = seeds + seed_degs

        else:
            
            seeds = seed_degs
    else:            
        
        seeds = [ko_gene]
        
    
    lst_columnName = ['0']
    lst_seed = []
    lst_weights = []
    
    weight = [1.0]
    
    for seed in seeds:
        lst_seed.append(seed)
        lst_weights.append(weight)
    arr_weights=np.array(lst_weights)
    
    # run the experiments, and write a rank list to stdout
    
    dic_node2weights = {}
    set_nodes = set()
    lst_wk = []
    network_name = []
    
    # 3 column network (source gene, target gene, weight)
    input_graph = network_reconstruction(grn)
    

    network_name.append(input_graph)
    wk = Walker(input_graph, constantWeight = str2bool(constantWeight), absWeight=str2bool(absoluteWeight), addBidirectionEdge = str2bool(addBidirectionEdge))
    set_nodes |= set(wk.OG.nodes())
    lst_wk.append(wk)
    
    column_name = []
    
    for idx, wk in enumerate(lst_wk):
        for j in range(arr_weights.shape[1]): # iterate # of samples times
            if len(network_name) > 1:
                column_name.append(lst_columnName[j] + '_'+ network_name[idx])
            else:
                column_name.append(lst_columnName[j])
            if sum(np.abs(arr_weights[:,j])) == 0.0:
                for node in set_nodes:
                    if node not in dic_node2weights:
                        dic_node2weights[node]=[]
                    dic_node2weight[node].append(0.0)
                continue
            seed2weight = dict()
            for ii in range(len(lst_seed)):
                seed2weight[lst_seed[ii]] = arr_weights[ii,j]
            lst_node_weight = wk.run_exp(seed2weight, restart_prob, normalize=str2bool(normalize))
            set_tmpNodes = set()
            for node, weight, all_weight in lst_node_weight:
                if node not in dic_node2weights:
                    dic_node2weights[node] = []
                dic_node2weights[node].append(weight)
                set_tmpNodes.add(node)
            for node in set_nodes-set_tmpNodes:
                if node not in dic_node2weights:
                    dic_node2weights[node] = []
                dic_node2weights[node].append(0.0)
    
    np_result_df = []
    for node, weights in dic_node2weights.items():
        assert len(weights) == 1, f"There are more than 1 number of weights: {weights}"
        tmp = {
            'protein': node,
            'np_score': float(weights[0])
        }
        np_result_df.append(tmp)

    #Soring in descending manner using 'rank' function
    
    np_result_df = pd.DataFrame(np_result_df)
    np_result_df = np_result_df.sort_values('np_score', ascending = False)
    np_result_df = np_result_df.to_dict('records')



    return pd.DataFrame(np_result_df)
