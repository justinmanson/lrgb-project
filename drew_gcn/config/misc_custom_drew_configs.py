from yacs.config import CfgNode as CN

from torch_geometric.graphgym.register import register_config


def set_misc_custom_drew_configs(cfg):
    r'''
    For utilising relational edge labels 
    '''

    # ----------------------------------------------------------------------- #
    # Customized options
    # ----------------------------------------------------------------------- #

    # Select device: 'cpu', 'cuda:0', 'auto'
    cfg.device = 'auto'

    

register_config('misc_drew', set_misc_custom_drew_configs)