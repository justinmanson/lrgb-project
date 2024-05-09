import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch.optim import AdamW, Optimizer
from torch_geometric.data import Dataset
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 256)
        self.conv2 = GCNConv(256, 256)
        self.conv3 = GCNConv(256, 256)
        self.conv4 = GCNConv(256, 256)
        self.conv5 = GCNConv(256, 256)
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv5(x, edge_index))
        x = global_mean_pool(x, batch)  # Pool node features to get graph-level representation
        x = self.fc(x)
        return x
    
def train(model: torch.nn.Module, loader: DataLoader, optimizer: Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    loss_function = torch.nn.BCEWithLogitsLoss()
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_preds = []
    all_labels = []
    for data in tqdm(loader):
        data = data.to(device)
        out = model(data)
        preds = torch.sigmoid(out).detach().cpu().numpy()
        labels = data.y.detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    # Calculate mean Average Precision
    ap_scores = [average_precision_score(all_labels[:, i], all_preds[:, i]) for i in range(all_labels.shape[1])]
    mean_ap = np.mean(ap_scores)
    return mean_ap


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load training and testing sets
    train_dataset = LRGBDataset(root="data/peptides-func", name="peptides-func", split='train')
    test_dataset = LRGBDataset(root="data/peptides-func", name="peptides-func", split='test')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize models
    gcn = GCN(train_dataset.num_features, train_dataset.num_classes).to(device)

    # Optimizer GCN
    optimizer = AdamW(gcn.parameters(), lr=0.001, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, min_lr=1e-5)

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

    for epoch in range(1, 501):
        loss = train(gcn, train_loader, optimizer, device)
        test_acc = test(gcn, test_loader, device)
        scheduler.step(test_acc)

        # Early Stopping and Checkpoint Logic
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(gcn.state_dict(), os.path.join(model_save_dir, f'gcn_weights.pt'))
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