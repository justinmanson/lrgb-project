# Graph Classification Using LRGB Dataset

This project utilizes the LRGB dataset to test and compare various architectures designed to model long-range interactions within graphs. Our focus is on a graph classification task using the peptides functional dataset from within the LRGB dataset.

## Project Structure

The repository is structured into several directories, each containing different models and components of our project. Below is a detailed description of each directory:

### Directories

#### `attn_pool_classifier`
This directory contains a model implementing a gated attention mechanism to compute scores/weights for each node embedding. The aggregated/pool all nodes into a single embedding. This is followed by a 2-layer MLP classifier. This non-graph / non-GNN approach serves as a baseline or sanity check to validate the necessity of using GNNs for this classification problem.

#### `multi_head_attn_pool_classifier`
This directory features a model using a multi-headed gated attention mechanism to compute scores/weights for each node embedding, then aggregating/pooling all nodes into multiple (exactly 10) embeddings. By using multiple channels, the compression is reduced, preserving more information from the original node embeddings and enhancing the classification model.

#### `gcn_baseline`
This directory includes a baseline GCN (Graph Convolutional Network) implementation, as utilized in the original LRGB paper experiments.

#### `gcn_virtual_nodes`
This directory contains a GCN model enhanced with virtual nodes. A few virtual nodes are added to the graph, fully connected to the rest of the graph. Pre-existing edge weights are set to 1, while virtual edges have random weights. Standard GCN is then applied with these new virtual connections to mitigate oversquashing of information and improve long-range interaction (LRI) modeling capabilities.

#### `drew_gcn`
This directory contains a GCN model that uses dynamic rewiring on top of the GCN to improve LRI modeling.

#### `san_baseline`
This directory features a Spectral Attention Network (SAN) baseline, implemented exactly as described in the original LRGB paper experiments.

#### `source_repositories`
This directory contains the source repositories that served as the basis for our work.

#### `requirements.txt`
This file lists the dependencies required to set up the virtual environment and run our code.

## Results

We evaluate the performance of each architecture type on the test set and report the average precision. The results will be updated in the table below:

| Architecture                      | Average Precision |
|--------------------------------   |-------------------|
| `attn_pool_classifier`            |      0.5256       |
| `multi_head_attn_pool_classifier` |      0.5837       |
| `gcn_baseline`                    |      0.5799       |
| `gcn_virtual_nodes`               |      0.6702       |
| `drew_gcn`                        |      0.6692       |
| `san_baseline`                    |      0.6620       |

## Getting Started

To get started with this project, please follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/justinmanson/lrgb-project
   cd graph-classification-lrgb
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Navigate to the desired model directory and run the **main.py** script to train and evaluate the model.



## Acknowledgments

We would like to thank the authors of the LRGB paper and the maintainers of the source repositories used as the foundation for our work.