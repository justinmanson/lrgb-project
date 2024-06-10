import os
import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist) 
# from custom_config import (cfg, dump_cfg,
#                     set_cfg, load_cfg,
#                     makedirs_rm_exist) 
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.register import register_pooling
from tqdm import tqdm
from torch_geometric.data import Data

from utils import train, test, Args, save_result_logs
from graphgps.make_k_hop_edges import make_k_hop_edges

# import custom configs
from config import *
import graphgps  # noqa, register custom modules

EXPERIMENT_NAME = "drew_gcn"

def remove_edge_attrs(dataset):
    """Removes edge attrs from dataset for experiments which don't use them"""
    dataset.data.edge_attr = None
    if any([dataset.get(i).edge_attr is not None for i in range(len(dataset))]):
        print('Removing edge attrs so GDC preprocessing can be performed...')
        count = 0
        for i in tqdm(range(len(dataset))): 
            if dataset.get(i).edge_attr is not None:
                count += 1
                dataset._data_list[i] = Data(x=dataset.get(i).x,
                                            edge_index=dataset.get(i).edge_index,
                                            edge_attr=None,
                                            y=dataset.get(i).y)
        assert not any([dataset.get(i).edge_attr is not None for i in range(len(dataset))])
    return dataset

def update_drew_edge_attributes(cfg, dataset, split):
    multi_hop_stages = [
            'sp_gnn',
            'drew_gnn',
            'k_gnn',
    ]
    multi_hop_models = ['drew_gated_gnn', 'drew_gin']
    use_drew = any([
        (cfg.gnn.stage_type in multi_hop_stages),
        ('delay' in cfg.gnn.stage_type),
        (cfg.model.type in multi_hop_models),
    ])

    if use_drew or ('noedge' in cfg.gnn.layer_type) or ('peptides' in cfg.dataset.name):
        dataset = remove_edge_attrs(dataset)

        k_max = min(cfg.gnn.layers_mp, cfg.k_max)
        dataset = make_k_hop_edges(dataset, k_max, cfg.dataset.format, cfg.dataset.name, split)
    else:
        print("If not drew, then why are we running this script?")

    return dataset


def main():
    # For full path
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(script_dir, 'pept-func_DRew-GCN_bs=0128_d=042_L=23.yaml')

    # Hardcode the configuration file
    args = Args(config_file)  # format to make load_cfg() work
    
    # Set and load config
    set_cfg(cfg)
    load_cfg(cfg, args)

    cfg.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'  # add to config - needed for create_model()
    device = torch.device(cfg.accelerator)  # my prefered notation
    print(f'Using device: {device}')

    # Load training and testing sets
    train_dataset = LRGBDataset(root="data", name="peptides-func", split='train')
    val_dataset = LRGBDataset(root="data", name="peptides-func", split='val')
    test_dataset = LRGBDataset(root="data", name="peptides-func", split='test')

    # DREW SPECIFIC
    train_dataset = update_drew_edge_attributes(cfg, train_dataset, 'train')
    val_dataset = update_drew_edge_attributes(cfg, val_dataset, 'val')
    test_dataset = update_drew_edge_attributes(cfg, test_dataset, 'test')

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    # Initialize GCN model
    model = create_model(dim_out=10)  # uses graphgym GNN() module  (torch_geometric > graphgym > models > gnn.py)

    # Optimizer (asserts since I overwrode with my own)
    assert cfg.optim.optimizer == "adamW", "We implement 'adamW' but cfg specifies other option"
    assert cfg.optim.scheduler == "reduce_on_plateau", "We implement 'reduce_on_plateau' but cfg specifies other option"
    optimizer = AdamW(model.parameters(), lr=cfg.optim.base_lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min', 
        factor=cfg.optim.reduce_factor, 
        patience=cfg.optim.schedule_patience, 
        min_lr=cfg.optim.min_lr
    )

    # Directory for saving model checkpoints
    model_save_dir = 'model_weights'
    os.makedirs(model_save_dir, exist_ok=True)

    early_stopping_patience = 50  # Number of epochs to wait for improvement before stopping
    no_improve_epochs = 0
    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(1, cfg.optim.max_epoch + 1):
        train_loss = train(model, train_loader, optimizer, device)
        val_acc, val_loss = test(model, val_loader, device)
        scheduler.step(val_loss)

        # Early stopping and checkpoint logic
        if val_acc > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'{EXPERIMENT_NAME}_weights.pt'))
            print(f'New best model saved at epoch {epoch} with validation accuracy {val_acc:.4f}')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            print(f'Early stopping triggered after {early_stopping_patience} epochs without improvement.')
            break

        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    # End of training
    print(f'Best {EXPERIMENT_NAME} model: Epoch {best_epoch} with Validation Accuracy {best_val_acc:.4f}')

    # Reload the best model
    model.load_state_dict(torch.load(os.path.join(model_save_dir, f'{EXPERIMENT_NAME}_weights.pt')))

    # Test the best model
    test_acc, test_loss = test(model, test_loader, device)
    print(f'Test Accuracy of the best model: {test_acc:.4f}')

    # Save run statistics in both local and shared directory
    save_result_logs(EXPERIMENT_NAME, best_epoch, best_val_acc, test_acc)


if __name__ == "__main__":
    main()