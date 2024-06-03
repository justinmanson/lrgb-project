import torch
import torch.nn as nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.nn import GCNConv
from torch_geometric.graphgym.register import register_layer

@register_layer('gcnconv_attn_vn')  # GCNconv layer with virtual node integration
class GCNConvAttnVN(torch.nn.Module):
    """
    A Graph Convolutional Network (GCN) layer.
    Here we assume that the last node of graph is a virtual node, fully connected to other nodes in graph.
    We apply a gated attention mechanism to pool node embeddings (MIL - see arXiv:1802.04712).
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GCNConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
        )
        self.pooling_fn = AttentionPooling(L=layer_config.dim_out, M=2*layer_config.dim_out)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        
        # Copy batch.x to avoid in-place modification
        new_x = batch.x.clone()

        for i in range(len(batch.ptr) - 1):
            node_embeddings = batch.x[batch.ptr[i]: batch.ptr[i+1]]  # extract all node embeddings from a single graph
            new_x[batch.ptr[i+1] - 1] = self.pooling_fn(node_embeddings)  # update virtual node embedding

        batch.x = new_x  # update batch.x with the new tensor
        return batch

class Attn_Net_Gated(nn.Module):
    def __init__(self, L : int, M : int):
        """
        Attention Network with Sigmoid Gating (3 fc layers)
        Args:
            L: input feature dimension
            M: hidden layer dimension
        """
        super(Attn_Net_Gated, self).__init__()

        # First fully connected layer (V matrix in the paper)
        self.fc1 = nn.Linear(L, M, bias=False) # test whether bias changes anything
        # Second fully connected layer (w matrix in the paper)
        self.fc2 = nn.Linear(M, 1, bias=False)
        # Gating fully connected layer (U matrix in the paper)
        self.fc_gate = nn.Linear(L, M, bias=False)

    def forward(self, x):
        """Forward path of the gated attention network
        Args:
            xin: (N, L) List of N patches and L features
        Return:
            A: (N, 1) Attention value for each patch
        """
        gating_weights = torch.sigmoid(self.fc_gate(x))
        x = torch.tanh(self.fc1(x))
        x = x * gating_weights
        A = self.fc2(x)
        A = torch.softmax(A, dim=0)
        return A
    
class AttentionPooling(nn.Module):
    def __init__(self, L : int, M : int):
        super().__init__()
        # Intatiate the gated layer
        self.attention_net = Attn_Net_Gated(L, M)

    def forward(self, x, attention_only : bool = False):
        """Forward pass
        Args:
            x (torch.tensor): (N, L) Input feature over N patches and L features
            attention_only (bool): Say whether to return the attention or not
        Returns:
            Y (torch.Tensor): (1, N) Output, if attention_only==False
            A (torch.Tensor): (1, M) Attention values, if attention_only==True
        """

        # Attention pooling (weighted average based on learned attention scores)
        A = self.attention_net(x)
        Y = torch.sum(x * A, dim=0, keepdim=True)  # (1, L)

        # Check if need to return attention
        if attention_only:
            return A
        else:
            return Y