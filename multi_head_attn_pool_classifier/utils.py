import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.optim import Optimizer
from typing import Tuple
import numpy as np
from sklearn.metrics import average_precision_score
import os
import pandas as pd

class Args:
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.opts = []  # You can add any additional options here if needed

def add_vn(data, num_virtual_nodes=1):
    """Add virtual nodes to the end of the graph, fully connected with other nodes in the graph."""
    num_nodes = data.num_nodes
    num_existing_edges = data.edge_index.size(1)

    # Assuming virtual node features are zeros
    virtual_node_features = torch.zeros((num_virtual_nodes, data.x.size(1)), dtype=data.x.dtype)
    data.x = torch.cat([data.x, virtual_node_features], dim=0)

    # Create edges between virtual nodes and all other nodes
    virtual_edges = []
    for v in range(num_virtual_nodes):
        vn_idx = num_nodes + v
        for i in range(num_nodes):
            virtual_edges.append([vn_idx, i])
            virtual_edges.append([i, vn_idx])

    virtual_edges = torch.tensor(virtual_edges, dtype=torch.int64).t()
    data.edge_index = torch.cat([data.edge_index, virtual_edges], dim=1)

    # Initialize edge attributes for virtual edges
    if data.edge_attr is not None:
        virtual_edge_attr = torch.zeros((virtual_edges.size(1), data.edge_attr.size(1)), dtype=data.edge_attr.dtype)
        data.edge_attr = torch.cat([data.edge_attr, virtual_edge_attr], dim=0)

    return data

def add_edge_weights(data, num_virtual_nodes=1):
    if data.edge_weight is None:
        num_existing_edges = data.edge_index.size(1) - 2 * num_virtual_nodes * (data.num_nodes - num_virtual_nodes)
        # Fixed weights for pre-existing edges
        existing_edge_weights = data.edge_weight
        if existing_edge_weights is None:
            existing_edge_weights = torch.ones(num_existing_edges, dtype=torch.float)
        
        # Random weights for virtual node edges
        num_virtual_edges = data.edge_index.size(1) - num_existing_edges
        virtual_edge_weights = torch.rand(num_virtual_edges, dtype=torch.float)
        
        # Concatenate fixed and learnable weights
        data.edge_weight = torch.cat([existing_edge_weights, virtual_edge_weights], dim=0)
    return data

def train(model: torch.nn.Module, loader: DataLoader, optimizer: Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    loss_function = torch.nn.BCEWithLogitsLoss()
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data)  # GNNGraphHead() module returns a pred, label tuple
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
        out, _ = model(data)  # GNNGraphHead() module returns a pred, label tuple
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

def save_result_logs(experiment_name, best_epoch, best_val_acc, test_acc):
    result_log_dir = '../result_logs'
    os.makedirs(result_log_dir, exist_ok=True)
    result_log_file = os.path.join(result_log_dir, f'{experiment_name}_logs.csv')

    stats = {
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc
    }

    df = pd.DataFrame([stats])
    df.to_csv(result_log_file, index=False)
    df.to_csv(f'{experiment_name}_logs.csv', index=False)

    print(f'Statistics saved to {result_log_file}')