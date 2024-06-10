import torch.nn as nn
# import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
# from torch_geometric.graphgym.register import register_stage
import torch
from .example import GNNLayer
from param_calcs import get_k_neighbourhoods

sort_and_removes_dupes = lambda mylist : sorted(list(dict.fromkeys(mylist)))

def init_DRewGCN(model, dim_in, dim_out, num_layers, skip_first_hop=False):
  """The (nu)DRew-GCN param initialiser, used for drew_gnn"""
  model.num_layers, use_weights = num_layers, cfg.agg_weights.use
  model.nu = cfg.nu if cfg.nu != -1 else float('inf')
  W_kt = {}
  if use_weights: alpha_t = []
  t0 = 1 if skip_first_hop else 0
  for t in range(t0, num_layers):
    d_in = dim_in if t == 0 else dim_out
    k_neighbourhoods = get_k_neighbourhoods(t)
    for k in k_neighbourhoods:
      W_kt["k=%d, t=%d" % (k,t)] = GNNLayer(d_in, dim_out) # regular GCN layers
    # if use_weights: alpha_t.append(torch.nn.Parameter(torch.randn(len(k_neighbourhoods)), requires_grad=True)) # random init from normal dist
    if use_weights: alpha_t.append(torch.nn.Parameter(torch.ones(len(k_neighbourhoods)), requires_grad=True)) # unity init
  model.W_kt = nn.ModuleDict(W_kt)
  if use_weights: model.alpha_t = nn.ParameterList(alpha_t)
  return model