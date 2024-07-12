import math
import torch
from torch import Tensor
from torch import nn

from graph import Graph

from torch.nn import (ModuleList, Linear, Conv1d, MaxPool1d, Embedding, ReLU, 
                      Sequential, BatchNorm1d as BN)

from torch_geometric.nn import GCNConv, GINConv, GATv2Conv
from torch.nn import functional as F



## Loopy Belief Propagation
class LBP(nn.Module):
    """
    Loopy Belief Propagation.
    """
    def __init__(self,
                 graph: Graph,
                 trn_nodes: Tensor,
                 num_states: int,
                 device: torch.device,
                 epsilon: float = 0.9,
                 diffusion: int = 10):
        """
        Initializer.
        """
        super(LBP, self).__init__()

        self.num_states = 2 # +1 or -1
        self.diffusion = diffusion
        self.epsilon = epsilon
        self.device = device

        self.softmax = nn.Softmax(dim=1)
        self.graph = graph
        self.trn_nodes = trn_nodes
        self.num_edges = self.graph.num_edges()
        self.features = self.graph.get_features()
        self.potential = torch.exp(
            (torch.ones(self.num_states, device=device) - torch.eye(self.num_states, device=device))*(1.0-self.epsilon) 
            + torch.eye(self.num_states, device=device)*self.epsilon
        )

    def _init_messages(self) -> Tensor:
        """
        Initialize (or create) a message matrix.
        """
        size = (self.num_edges * 2, self.num_states)
        return torch.ones(size, device=self.device) / self.num_states

    def _update_messages(self, messages: Tensor, beliefs: Tensor) -> Tensor:
        """
        Update the message matrix with using beliefs.
        """
        new_beliefs = beliefs[self.graph.src_nodes]
        rev_messages = messages[self.graph.rev_edges]
        new_msgs = torch.mm(new_beliefs / rev_messages, self.potential)
        new_msgs = new_msgs / new_msgs.sum(dim=1, keepdim=True)
        return new_msgs

    def _compute_beliefs(self, priors: Tensor, messages: Tensor, EPSILON) -> Tensor:
        """
        Compute new beliefs based on the current messages.
        """
        beliefs = torch.log(torch.clamp(priors, min=EPSILON))
        log_msgs = torch.log(torch.clamp(messages, min=EPSILON))
        beliefs.index_add_(0, self.graph.dst_nodes, log_msgs)
        return self.softmax(beliefs)

    def propagate(self, priors: Tensor, THRESHOLD, EPSILON):
        """
        Propagate the priors produced from the classifier.
        """
        
        messages = self._init_messages()
        new_messages = messages.clone().detach()
        count = 0
        while (new_messages - messages).abs().mean().item() > THRESHOLD or count == 0:
            messages = new_messages.clone().detach()
            beliefs = self._compute_beliefs(priors, messages, EPSILON)
            new_messages = self._update_messages(messages, beliefs)
            count += 1
            if count >= 10: ##############
                break      ##############
            
        beliefs = self._compute_beliefs(priors, new_messages, EPSILON)
        print('number of iteration : ', count)
        
        return beliefs
    
    def create_node_priors(self, class_prior: float):
        
        num_nodes = self.num_nodes()
        size = (num_nodes, self.num_states)
        priors = torch.tensor([class_prior, 1.0-class_prior], device=self.device, dtype=torch.float32).repeat(num_nodes, 1)
        priors[self.trn_nodes] = torch.tensor([1.0, 0.0], device=self.device)
        
        return priors

    def num_nodes(self) -> int:
        """
        Count the number of nodes in the current graph.
        """
        return self.graph.num_nodes()
    
    def forward(self, class_prior, THRESHOLD, EPSILON) -> Tensor:
        """
        Run the loopy belief propagation of this model.
        """
        
        priors = self.create_node_priors(class_prior)
        # print(f'priors: {priors}') #####
        beliefs = self.propagate(priors, THRESHOLD, EPSILON)
        # print(f'beliefs: {beliefs}') #####
        
        return beliefs


class GCN(torch.nn.Module):

    def __init__(self, hidden_channels, num_layers, 
                 node_embedding, dropout = 0.1):

        super(GCN, self).__init__()

        self.node_embedding = node_embedding
        
        self.convs = ModuleList()

        initial_channels = hidden_channels
        # initial_channels += node_embedding.embedding_dim

        self.convs.append(GCNConv(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)

    def reset_parameters(self):

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):

        x = self.node_embedding(data.x)

        for conv in self.convs[:-1]:

            x = conv(x, data.edge_index, data.edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training = self.training)

        x = self.convs[-1](x, data.edge_index, data.edge_weight)

        x = self.lin1(x)  # Use self.lin1 instead of self.final_linear
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin2(x)  # Use self.lin2 for the final linear layer

        return x


class GIN(torch.nn.Module):
    
    def __init__(self, hidden_channels, node_embedding, num_layers=3, dropout=0.5, train_eps=False, jk = True):
        
        super(GIN, self).__init__()
        
        self.node_embedding = node_embedding
        self.jk = jk


        initial_channels = hidden_channels
        
        self.conv1 = GINConv(
            Sequential(
                Linear(initial_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                BN(hidden_channels),
            ),
            train_eps=train_eps)
        
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        BN(hidden_channels),
                    ),
                    train_eps=train_eps))

        self.dropout = dropout
        
        if self.jk:
            self.lin1 = Linear(num_layers * hidden_channels, hidden_channels)
        else:
            self.lin1 = Linear(hidden_channels, hidden_channels)
            
        self.lin2 = Linear(hidden_channels, 2)
        

    def forward(self, data):
        
        x = self.node_embedding(data.x)

        x = self.conv1(x, data.edge_index)

        xs = [x]

        for conv in self.convs:
            x = conv(x, data.edge_index)
            xs += [x]

        x = torch.cat(xs, dim=1)
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, node_embedding, dropout=0.5, heads=8):

        super(GAT, self).__init__()
        self.node_embedding = node_embedding

        self.gat1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads)
        self.gat2 = GATv2Conv(hidden_channels*heads, 2, heads=1)
 
    # def forward(self, x, edge_index):
    def forward(self, data):
    
        x = self.node_embedding(data.x)

        x = F.dropout(x, p=0.6, training=self.training) # Preventing overfitting
        x = self.gat1(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, data.edge_index)

        return x

