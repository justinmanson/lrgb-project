import os
import torch
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist) 
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian
from functools import partial
from transform.posenc_stats import get_lap_decomp_stats
from torch.optim import AdamW

from utils import train, test, Args, save_result_logs

# import custom configs
from config import *
from model import *
from encoder import *
from transform import *


EXPERIMENT_NAME = "san"
    

def compute_laplacian_eigen(data, normalization='sym', max_freqs=10, eigvec_norm=None):
    """Compute eigenvalues and eigenvectors of the graph Laplacian."""
    N = data.num_nodes if hasattr(data, 'num_nodes') else data.x.shape[0]
    L = to_scipy_sparse_matrix(
        *get_laplacian(data.edge_index, normalization=normalization, num_nodes=N)
    )
    
    evals, evects = np.linalg.eigh(L.toarray())
    
    max_freqs=cfg.posenc_LapPE.eigen.max_freqs
    eigvec_norm=cfg.posenc_LapPE.eigen.eigvec_norm

    data.EigVals, data.EigVecs = get_lap_decomp_stats(
        evals=evals, evects=evects,
        max_freqs=max_freqs,
        eigvec_norm=eigvec_norm
    )

    return data


def main():
    # For full path
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.join(script_dir, 'peptides-func-SAN.yaml')

    # Hardcode the configuration file
    # config_file = 'peptides-func-SAN.yaml'
    args = Args(config_file)  # format to make load_cfg() work
    
    # Set and load config
    set_cfg(cfg)
    load_cfg(cfg, args)

    cfg.accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'  # add to config - needed for create_model()
    device = torch.device(cfg.accelerator)  # my prefered notation
    print(f'Using device: {device}')

    pre_transform_func = partial(
        compute_laplacian_eigen, 
        normalization=cfg.posenc_LapPE.eigen.laplacian_norm, 
        max_freqs=cfg.posenc_LapPE.eigen.max_freqs, 
        eigvec_norm=cfg.posenc_LapPE.eigen.eigvec_norm
    )

    # Load training and testing sets
    train_dataset = LRGBDataset(root="data", name="peptides-func", split='train', pre_transform=pre_transform_func)
    val_dataset = LRGBDataset(root="data", name="peptides-func", split='val', pre_transform=pre_transform_func)
    test_dataset = LRGBDataset(root="data", name="peptides-func", split='test', pre_transform=pre_transform_func)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)

    # Initialize SAN model
    model = create_model(dim_out=10)

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