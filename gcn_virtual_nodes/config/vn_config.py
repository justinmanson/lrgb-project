from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

def set_cfg_vn(cfg):
    """Configuration for Virtual Nodes.
    """

    # Virtual nodes argument group
    cfg.vn = CN()

    # Number of virtual nodes to add to each graph
    cfg.vn.num_vns = 1


register_config('cfg_vn', set_cfg_vn)