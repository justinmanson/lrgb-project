import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops
import torch_geometric.graphgym.register as register
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch.optim import AdamW, Optimizer
from torch_geometric.data import Dataset
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import numpy as np
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist) 
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian
from functools import partial
from transform.posenc_stats import get_lap_decomp_stats
from typing import Tuple

# import custom configs
from config import *
from model import *
from encoder import *
from transform import *
    
def train(model: torch.nn.Module, loader: DataLoader, optimizer: Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    loss_function = torch.nn.BCEWithLogitsLoss()
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data)  # the SANGraphHead module returns a pred, label tuple
        loss = loss_function(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    loss_function = torch.nn.BCEWithLogitsLoss()
    for data in tqdm(loader):
        data = data.to(device)
        out, _ = model(data)  # the SANGraphHead module returns a pred, label tuple
        loss = loss_function(out, data.y.float())
        total_loss += loss.item()
        preds = torch.sigmoid(out).detach().cpu().numpy()
        labels = data.y.detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    # Calculate mean Average Precision
    ap_scores = [average_precision_score(all_labels[:, i], all_preds[:, i]) for i in range(all_labels.shape[1])]
    mean_ap = np.mean(ap_scores)
    return mean_ap, total_loss / len(loader)


class Args:
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.opts = []  # You can add any additional options here if needed


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
    test_dataset = LRGBDataset(root="data", name="peptides-func", split='test', pre_transform=pre_transform_func)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
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

        # Early Stopping and Checkpoint Logic
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'san_weights.pt'))
            print(f'New best model saved at epoch {epoch} with accuracy {test_acc:.4f}')
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_patience:
            print(f'Early stopping triggered after {early_stopping_patience} epochs without improvement.')
            break

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {test_acc:.4f}')

    print(f'Best SAN model: Epoch {best_epoch} with Accuracy {best_accuracy:.4f}')


if __name__ == "__main__":
    main()