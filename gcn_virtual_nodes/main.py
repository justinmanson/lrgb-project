import os
import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch_geometric.graphgym.config import (cfg, set_cfg, load_cfg) 
from torch_geometric.graphgym.model_builder import create_model
from utils import Args, train, test, add_vn, add_edge_weights

# import custom configs
from config import *
from gcn_attn_vn import *  # allows us to register gcnconv_attn_vn layer
from gcn_learned_vn import *  # registers gcnconv_learned_vn

def main():
    # For full path
    script_dir = os.path.dirname(os.path.realpath(__file__))  # allows us to run script from vscode terminal for debugging
    config_file = os.path.join(script_dir, 'peptides-func-GCN.yaml')

    # Hardcode the configuration file
    # config_file = 'peptides-func-GCN.yaml'
    args = Args(config_file)  # format to make load_cfg() work
    
    # Set and load config
    set_cfg(cfg)
    load_cfg(cfg, args)

    cfg.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'  # add to config - needed for create_model()
    device = torch.device(cfg.accelerator)  # my prefered notation
    print(f'Using device: {device}')

    # Load training and testing sets
    train_dataset = LRGBDataset(root="data", name="peptides-func", split='train')
    test_dataset = LRGBDataset(root="data", name="peptides-func", split='test')

    # Add virtual node, fully connected to other nodes in graph
    train_dataset = [add_vn(data, num_virtual_nodes=cfg.vn.num_vns) for data in train_dataset]
    test_dataset = [add_vn(data, num_virtual_nodes=cfg.vn.num_vns) for data in test_dataset]

    # Add edge weights - 1 for pre-existing edges and random weight for virtual edges
    train_dataset = [add_edge_weights(data, num_virtual_nodes=cfg.vn.num_vns) for data in train_dataset]
    test_dataset = [add_edge_weights(data, num_virtual_nodes=cfg.vn.num_vns) for data in test_dataset]

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    # Initialize GCN model
    """
    Options:
    - gcn_conv: uses random weights for virtual edges and applies standard message passing.
    - gcn_learned_vn: learns weights for edges on top of standard message passing.
    - gcn_attn_vn: instead of applying standard message passing for virtual nodes,
                   pools node embeddings using attention scores to compute weighted average.

    Results:
    We find that the first option (simply apply random weights to virtual edges) works best.
    """
    model = create_model(dim_out=10)  # uses graphgym GNN() module  (torch_geometric > graphgym > models > gnn.py)

    # Optimizer (asserts since I only hardcoded the following option)
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

    # Best model tracking
    best_accuracy = 0
    best_epoch = 0

    # Directory for saving model checkpoints
    model_save_dir = 'model_weights'
    os.makedirs(model_save_dir, exist_ok=True)

    early_stopping_patience = 50  # Number of epochs to wait for improvement before stopping
    no_improve_epochs = 0
    best_accuracy = 0
    best_epoch = 0

    for epoch in range(1, cfg.optim.max_epoch + 1):
        loss = train(model, train_loader, optimizer, device)
        test_acc, test_loss = test(model, test_loader, device)
        scheduler.step(test_loss)

        # Early stopping and checkpoint logic
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'gcn_weights.pt'))
            print(f'New best model saved at epoch {epoch} with accuracy {test_acc:.4f}')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            print(f'Early stopping triggered after {early_stopping_patience} epochs without improvement.')
            break

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {test_acc:.4f}')

    print(f'Best GCN model: Epoch {best_epoch} with Accuracy {best_accuracy:.4f}')


if __name__ == "__main__":
    main()