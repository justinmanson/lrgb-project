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
        self.ATTENTION_BRANCHES = 1

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
        # Apply softmax and weighted avg separately for each graph in the batch
        node_counts = batch.bincount().tolist()
        A_split = torch.split(A, node_counts, dim=1)
        H_split = torch.split(H, node_counts, dim=0)

        Z_split = [torch.mm(F.softmax(a, dim=1), h) for a, h in zip(A_split, H_split)]
        Z = torch.cat(Z_split, dim=0)  # Concatenate to get the final embeddings

        return Z

# @register_network('mil_classifier')
# class MILClassifier(torch.nn.Module):
#     r"""Multi Instance Learning Classifier.

#     The model consists of three main components:

#     1. An encoder to transform input features into a fixed-size embedding
#        space.
#     2. A gated attention unit to perform pooling of node features based on
#        a weighted average with attention scores.
#     3. A head to produce the final output features/predictions.

#     The configuration of each component is determined by the underlying
#     configuration in :obj:`cfg`.

#     Args:
#         dim_in (int): The input feature dimension.
#         dim_out (int): The output feature dimension.
#         **kwargs (optional): Additional keyword arguments.
#     """
#     def __init__(self, dim_in: int, dim_out: int, **kwargs):
#         super().__init__()

#         # GNNStage = register.stage_dict[cfg.gnn.stage_type]
#         # GNNHead = register.head_dict[cfg.gnn.head]

#         self.encoder = FeatureEncoder(dim_in)
#         dim_in = self.encoder.dim_in

#         self.pooling_fn = AttentionPooling(L=cfg.gnn.dim_inner, M=2*cfg.gnn.dim_inner)
#         self.classifier = nn.Linear(cfg.gnn.dim_inner, dim_out)

#         # if cfg.gnn.layers_pre_mp > 0:
#         #     self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
#         #                            cfg.gnn.layers_pre_mp)
#         #     dim_in = cfg.gnn.dim_inner
#         # if cfg.gnn.layers_mp > 0:
#         #     self.mp = GNNStage(dim_in=dim_in, dim_out=cfg.gnn.dim_inner,
#         #                        num_layers=cfg.gnn.layers_mp)
#         # self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

#         self.apply(init_weights)

#     def forward(self, batch):
#         node_features = batch.x
#         node_embeddings = self.encoder(node_features)
#         pooled_node_embeddings = self.pooling_fn(node_embeddings)
#         output = self.classifier(pooled_node_embeddings)
#         return output
        
#         # for module in self.children():
#         #     batch = module(batch)
#         # return batch


# @register_head('mil')
# class MILGraphHead(torch.nn.Module):
#     r"""A GNN prediction head for graph-level prediction tasks.
#     A post message passing layer (as specified by :obj:`cfg.gnn.post_mp`) is
#     used to transform the pooled graph-level embeddings using an MLP.

#     Args:
#         dim_in (int): The input feature dimension.
#         dim_out (int): The output feature dimension.
#     """
#     def __init__(self, dim_in: int, dim_out: int):
#         super().__init__()
#         self.layer_post_mp = MLP(
#             new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
#                              has_act=False, has_bias=True, cfg=cfg))
#         self.pooling_fun = AttentionPooling(L=cfg.gnn.dim_inner, M=2*cfg.gnn.dim_inner)

#     def _apply_index(self, batch):
#         return batch.graph_feature, batch.y

#     def forward(self, batch):
#         graph_emb = self.pooling_fun(batch.x, batch.batch)
#         graph_emb = self.layer_post_mp(graph_emb)
#         batch.graph_feature = graph_emb
#         pred, label = self._apply_index(batch)
#         return pred, label


# class Attn_Net_Gated(nn.Module):
#     def __init__(self, L : int, M : int):
#         """
#         Attention Network with Sigmoid Gating (3 fc layers)
#         Args:
#             L: input feature dimension
#             M: hidden layer dimension
#         """
#         super(Attn_Net_Gated, self).__init__()

#         # First fully connected layer (V matrix in the paper)
#         self.fc1 = nn.Linear(L, M, bias=False) # test whether bias changes anything
#         # Second fully connected layer (w matrix in the paper)
#         self.fc2 = nn.Linear(M, 1, bias=False)
#         # Gating fully connected layer (U matrix in the paper)
#         self.fc_gate = nn.Linear(L, M, bias=False)

#     def forward(self, x):
#         """Forward path of the gated attention network
#         Args:
#             xin: (N, L) List of N patches and L features
#         Return:
#             A: (N, 1) Attention value for each patch
#         """

#         gating_weights = torch.sigmoid(self.fc_gate(x))
#         x = torch.tanh(self.fc1(x))
#         x = x * gating_weights
#         A = self.fc2(x)

#         A = torch.softmax(A, dim=0)
#         return A

# class AttentionPooling(nn.Module):
#     def __init__(self, L : int, M : int):
#         super().__init__()
#         # Intatiate the gated layer
#         self.attention_net = Attn_Net_Gated(L, M)

#     def forward(self, x, batch, attention_only : bool = False):
#         """Forward pass
#         Args:
#             x (torch.tensor): (N, L) Input feature over N patches and L features
#             attention_only (bool): Say whether to return the attention or not
#         Returns:
#             Y (torch.Tensor): (1, N) Output, if attention_only==False
#             A (torch.Tensor): (1, M) Attention values, if attention_only==True
#         """

#         # Attention pooling (weighted average based on learned attention scores)
#         A = self.attention_net(x)
#         Y = torch.sum(x * A, dim=0, keepdim=True)  # (1, L)

#         # Check if need to return attention
#         if attention_only:
#             return A
#         else:
#             return Y
