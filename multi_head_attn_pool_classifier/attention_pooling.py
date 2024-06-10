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
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP

class GatedAttention(nn.Module):
    def __init__(self, M: int, L: int, attn_branches: int = 1):
        super(GatedAttention, self).__init__()
        self.M = M
        self.L = L
        self.ATTENTION_BRANCHES = attn_branches

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
        
        # If there are multiple attention branches, we need to concatenate along the feature dimension
        if self.ATTENTION_BRANCHES > 1:
            Z_split = [z.view(-1, self.ATTENTION_BRANCHES * self.M) for z in Z_split]
        
        Z = torch.cat(Z_split, dim=0)  # Concatenate to get the final embeddings

        return Z
    
@register_network('multi_head_attn_gnn')
class MultiHeadAttnGNN(torch.nn.Module):
    r"""A general Graph Neural Network (GNN) model.

    The GNN model consists of three main components:

    1. An encoder to transform input features into a fixed-size embedding
       space.
    2. A processing or message passing stage for information exchange between
       nodes.
    3. A head to produce the final output features/predictions.

    The configuration of each component is determined by the underlying
    configuration in :obj:`cfg`.

    Args:
        dim_in (int): The input feature dimension.
        dim_out (int): The output feature dimension.
        **kwargs (optional): Additional keyword arguments.
    """
    def __init__(self, dim_in: int, dim_out: int, **kwargs):
        super().__init__()
        GNNStage = register.stage_dict[cfg.gnn.stage_type]
        GNNHead = register.head_dict[cfg.gnn.head]

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        # This is set to zero
        if cfg.gnn.layers_mp > 0:
            self.mp = GNNStage(dim_in=dim_in, dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
            
        # Attention pooling and post classifier
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner*cfg.attn.attention_branches, dim_out=dim_out)

        self.apply(init_weights)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
