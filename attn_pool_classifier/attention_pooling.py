import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.nn import GCNConv
from torch_geometric.graphgym.config import cfg
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_layer, register_network, register_pooling, register_head
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.models.gnn import FeatureEncoder

class GatedAttention(nn.Module):
    def __init__(self, M: int, L: int):
        super(GatedAttention, self).__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = 1  # current implementation only supports 1 here

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

    def forward(self, x, batch):
        # Feature extraction handled a priori
        H = x  # KxM

        A_V = self.attention_V(H)  # KxL
        A_U = self.attention_U(H)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        
        # FOR UN-BATCHED IMPLEMENTATION
        # A = F.softmax(A, dim=1)  # softmax over K
        # Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM
        
        # SPECIFIC TO BATCHED IMPLEMENTATION (where K = num_nodes_in_graph_1 + num_nodes_in_graph_2 + num_nodes_in_graph_3 + ...)
        # Apply softmax and weighted avg separately for each graph in the batch
        node_counts = batch.bincount().tolist()
        A_split = torch.split(A, node_counts, dim=1)
        H_split = torch.split(H, node_counts, dim=0)

        Z_split = [torch.mm(F.softmax(a, dim=1), h) for a, h in zip(A_split, H_split)]
        Z = torch.cat(Z_split, dim=0)  # Concatenate to get the final embeddings

        return Z
