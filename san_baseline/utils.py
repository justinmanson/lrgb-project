import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.optim import Optimizer
from typing import Tuple
import numpy as np
from sklearn.metrics import average_precision_score

class Args:
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.opts = []  # You can add any additional options here if needed

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