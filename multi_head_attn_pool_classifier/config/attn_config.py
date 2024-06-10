from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN

def set_cfg_attn(cfg):
    """Configuration for Virtual Nodes.
    """

    # Virtual nodes argument group
    cfg.attn = CN()

    # Number of attention heads to stack
    cfg.attn.attention_branches = 1


register_config('cfg_attn', set_cfg_attn)