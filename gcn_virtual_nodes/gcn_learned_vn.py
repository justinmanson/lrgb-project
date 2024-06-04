import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.nn import GCNConv
from torch_geometric.graphgym.register import register_layer


def reshape_node_embeddings(batch):
    """
    Concatenate the embeddings from two nodes connected by an edge.
    """
    # Extract node embeddings and edge indices
    node_embeddings = batch.x
    edge_index = batch.edge_index

    # Get source and target node indices
    src_node_indices = edge_index[0]
    tgt_node_indices = edge_index[1]

    # Gather node embeddings based on edge indices
    src_node_embeddings = node_embeddings[src_node_indices]
    tgt_node_embeddings = node_embeddings[tgt_node_indices]

    # Concatenate source and target node embeddings
    edge_node_embeddings = torch.cat([src_node_embeddings, tgt_node_embeddings], dim=1)

    return edge_node_embeddings

@register_layer('gcnconv_learned_vn')
class GCNConvLearnedVN(torch.nn.Module):
    r"""A Graph Convolutional Network (GCN) layer."""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GCNConv(
            layer_config.dim_in,
            layer_config.dim_out,
            bias=layer_config.has_bias,
        )

        self.edge_weight_predictor = nn.Linear(layer_config.dim_in * 2, 1)

    def forward(self, batch):
        """
        Our aim is to learn edge weights to guide the message-passing mechanism of the GCN.
        This is done by training a linear layer that receives the concatenation of two node embeddings
        connected by an edge and outputs a weight for this corresponding edge.

        We explore two approaches,
        Option 1: Pass all edges of the graph, virtual or not, through this linear layer and learn an edge weight.
        Option 2: Separate virtual edges from non-virtual edges. Set real edge weights to 1 and pass the concatenated
                  node embeddings connected by virtual edges through linear layer to learn edge weights.
        """

        # Option 1:
        # Attribute edge weights
        batch.edge_weight = torch.sigmoid(self.edge_weight_predictor(reshape_node_embeddings(batch)).squeeze(-1))
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_weight)
        return batch

        # Option 2:

        # The issue with this option is that extracting virtual edges from non-virtual edges is a slow and inneficient process.

        # edge_weights = []
        # for i in range(batch.ptr.size(0) - 1):
        #     start, end = batch.ptr[i], batch.ptr[i + 1]
        #     num_nodes = end - start - cfg.vn.num_vns

        #     # Identify virtual node indices
        #     virtual_node_indices = torch.arange(end - cfg.vn.num_vns, end, device=batch.x.device)
            
        #     # Identify virtual node edges
        #     is_virtual_edge = (batch.edge_index[0] >= end - cfg.vn.num_vns) | (batch.edge_index[1] >= end - cfg.vn.num_vns)

        #     # Split edge indices for this specific graph
        #     graph_edges_mask = (batch.edge_index[0] >= start) & (batch.edge_index[0] < end) & (batch.edge_index[1] >= start) & (batch.edge_index[1] < end)
        #     edge_index = batch.edge_index[:, graph_edges_mask]
        #     edge_node_embeddings = reshape_node_embeddings(batch, edge_index)

        #     # Assign weights
        #     num_edges = edge_index.size(1)
        #     virtual_edge_weights = torch.ones(num_edges, device=batch.x.device)
        #     if is_virtual_edge[graph_edges_mask].any():
        #         virtual_edge_weights[is_virtual_edge[graph_edges_mask]] = torch.sigmoid(self.edge_weight_predictor(edge_node_embeddings[is_virtual_edge[graph_edges_mask]]).squeeze(-1))
            
        #     edge_weights.append(virtual_edge_weights)

        # batch.edge_weight = torch.cat(edge_weights, dim=0)

        # batch.x = self.model(batch.x, batch.edge_index, batch.edge_weight)
        # return batch
    