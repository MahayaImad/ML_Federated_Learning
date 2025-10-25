# FL-comparison

## Description

Comparison framework for Federated Learning methods evaluating **FedAvg**, **FedProx**, **IFCA**, and **Agglomerative Clustering** on three datasets: **MNIST**, **CIFAR10**, and **MALNET**.

## Key Features

- **4 FL Methods**: FedAvg (Standard), FedProx, IFCA, Agglomerative Clustering (DJS-based)
- **3 Datasets**: MNIST, CIFAR10, MALNET (Android malware)
- **Data-driven Clustering**: Jensen-Shannon divergence for Agglomerative method
- **IID/Non-IID**: Flexible data distribution (Dirichlet α=0.5)
- **Comprehensive Metrics**: Accuracy, loss, training time
- **Client Statistics**: Detailed per-client CSV reports

## Methods Overview

### 1. FedAvg (Standard)
Classical Federated Averaging with weighted aggregation by client data size.

### 2. FedProx
FedAvg with proximal term to handle system heterogeneity:
- Adds regularization: `(μ/2)||w - w_global||²`
- Better convergence in non-IID settings

### 3. IFCA (Iterative Federated Clustering Algorithm)
Learns multiple cluster models and assigns clients based on loss:
- K cluster models trained simultaneously
- Dynamic client-to-cluster assignment
- Best model selected for evaluation

### 4. Agglomerative Clustering (2 Variants)
**Data-driven clustering** using Jensen-Shannon divergence:

#### 4a. agglomerative-inter (With Inter-Cluster)
- Computes client label distribution similarity (histograms)
- Forms clusters automatically using DJS threshold
- **Cyclic inter-cluster communication** each round for knowledge sharing
- Three-level hierarchical aggregation: Client → Cluster → Inter-Cluster → Global
- **Best for**: Fast convergence, highly non-IID data
- **Trade-off**: Higher communication cost

#### 4b. agglomerative-global (Global Only)
- Same clustering as above (DJS-based)
- **NO inter-cluster communication**
- Two-level aggregation: Client → Cluster → Global
- **Best for**: Limited communication budget, well-separated clusters
- **Trade-off**: Slower convergence

## Installation

```bash
# Install dependencies
pip install tensorflow numpy matplotlib scikit-learn scipy

# Using Nvidia Containers (Docker required)
# Windows:
docker run --rm -it --gpus all -v ${PWD}:/app nvcr.io/nvidia/tensorflow:24.07-tf2-py3 bash

# Linux:
docker run --rm -it --gpus all -v $(pwd):/app nvcr.io/nvidia/tensorflow:24.07-tf2-py3 bash

# Navigate to project
cd /app/FL-comparison
```

## Quick Start

```bash
# Single method - FedAvg
python main.py --dataset mnist --method fedavg --clients 20 --rounds 20

# Single method - FedProx
python main.py --dataset cifar10 --method fedprox --clients 20 --rounds 20 --mu 0.01

# Single method - IFCA
python main.py --dataset mnist --method ifca --clients 20 --rounds 20 --num-clusters 5

# Single method - Agglomerative (fixed subset)
python main.py --dataset cifar10 --method agglomerative --clients 20 --rounds 20 \
    --js-threshold 0.5 --selection-ratio 0.3 --subset-mode fixed

# Agglomerative with random subset per round
python main.py --dataset mnist --method agglomerative --clients 30 --rounds 25 \
    --js-threshold 0.5 --selection-ratio 0.3 --subset-mode random

# Compare ALL methods
python main.py --dataset mnist --method all --clients 20 --rounds 20 --verbose
```

## Command-Line Arguments

### Required Arguments
| Argument | Options | Description |
|----------|---------|-------------|
| `--dataset` | mnist, cifar10, malnet | Dataset to use |
| `--method` | fedavg, fedprox, ifca, agglomerative-inter, agglomerative-global, all | FL method (or "all" to compare) |

### General Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--clients` | int | 20 | Number of clients |
| `--rounds` | int | 20 | Communication rounds |
| `--local-epochs` | int | 5 | Local training epochs |
| `--batch-size` | int | 32 | Batch size |
| `--iid` | flag | False | IID data distribution |
| `--gpu` | int | 0 | GPU id (-1 for CPU) |
| `--seed` | int | 42 | Random seed |
| `--verbose` | flag | False | Verbose output |

### Agglomerative Specific
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--js-threshold` | float | 0.5 | JS divergence threshold for clustering |
| `--selection-ratio` | float | 0.3 | Client selection ratio per cluster (0-1) |
| `--subset-mode` | fixed/random | fixed | Fixed subset or random per round |

**Subset Modes:**
- `fixed`: Select top clients by data size (consistent across rounds)
- `random`: Random selection each round (more exploration)

### FedProx Specific
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mu` | float | 0.01 | Proximal term coefficient |

### IFCA Specific
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num-clusters` | int | 5 | Number of cluster models |

## Agglomerative Clustering Details

### Jensen-Shannon Divergence (DJS)
Measures similarity between client data distributions:

```
JS(P||Q) = √[0.5 × KL(P||M) + 0.5 × KL(Q||M)]
where M = 0.5 × (P + Q)
```

### Threshold Selection
- `--js-threshold 0.2`: Many small clusters (fine-grained)
- `--js-threshold 0.5`: Balanced (recommended)
- `--js-threshold 0.7`: Few large clusters (coarse)

### Client Selection Strategies
- `--selection-ratio 0.3`: Top 30% of clients per cluster
- `--selection-ratio 1.0`: All clients
- `--subset-mode fixed`: Consistent selection (by data size)
- `--subset-mode random`: Random selection each round

## Output Files

### Results Directory (`results/`)
- **Text format**: `results_[dataset]_[method]_[timestamp].txt`
  - Configuration details
  - Round-by-round metrics
  - Final summary
- **JSON format**: `results_[dataset]_[method]_[timestamp].json`
  - Structured data for analysis
- **CSV format**: `client_stats_[dataset]_[method]_[timestamp].csv`
  - Per-client statistics
  - Class distribution
  - Training metrics

### Visualizations Directory (`visualizations/`)
- **Comparison plots**: `comparison_[dataset]_[timestamp].png`
  - 4-panel visualization:
    1. Accuracy evolution
    2. Loss evolution
    3. Time per round
    4. Final performance summary

## MALNET Dataset Integration

### Expected Data Format
The MALNET dataset should provide:
- Feature vectors (e.g., from static analysis of Android APKs)
- Labels for malware families (20 classes)

### Implementation Steps
1. Implement `load_malnet_data()` in `data_preparation.py`
2. Expected return format:
```python
def load_malnet_data():
    # Load your MALNET data
    x_train = ... # Shape: (n_samples, n_features)
    y_train = ... # Shape: (n_samples, 20) - one-hot encoded
    x_test = ...
    y_test = ...
    return x_train, y_train, x_test, y_test
```

3. Adjust input dimensions in `config.py` if needed
4. Run experiments:
```bash
python main.py --dataset malnet --method all --clients 30 --rounds 25
```

## Example Workflows

### Experiment 1: Compare All Methods on MNIST
```bash
python main.py --dataset mnist --method all --clients 20 --rounds 30 --verbose
```

### Experiment 2: Agglomerative with Different Strategies
```bash
# Fixed subset (top clients)
python main.py --dataset cifar10 --method agglomerative \
    --clients 30 --rounds 20 --js-threshold 0.5 --selection-ratio 0.3 --subset-mode fixed

# Random subset each round
python main.py --dataset cifar10 --method agglomerative \
    --clients 30 --rounds 20 --js-threshold 0.5 --selection-ratio 0.3 --subset-mode random
```

### Experiment 3: FedProx with Different Mu Values
```bash
python main.py --dataset mnist --method fedprox --clients 20 --rounds 20 --mu 0.001
python main.py --dataset mnist --method fedprox --clients 20 --rounds 20 --mu 0.01
python main.py --dataset mnist --method fedprox --clients 20 --rounds 20 --mu 0.1
```

### Experiment 4: IFCA with Different Cluster Numbers
```bash
python main.py --dataset cifar10 --method ifca --clients 30 --rounds 25 --num-clusters 3
python main.py --dataset cifar10 --method ifca --clients 30 --rounds 25 --num-clusters 5
python main.py --dataset cifar10 --method ifca --clients 30 --rounds 25 --num-clusters 10
```

## Project Structure

```
FL-comparison/
├── main.py                      # Main entry point
├── config.py                    # Configuration parameters
├── data_preparation.py          # Data loading and splitting
├── models.py                    # Neural network models
├── client.py                    # Federated client implementation
├── train_fedavg.py             # FedAvg training logic
├── train_fedprox.py            # FedProx training logic
├── train_ifca.py               # IFCA training logic
├── train_agglomerative.py      # Agglomerative training logic
├── utils.py                     # Utility functions
├── README.md                    # This file
├── METHODS.md                   # Detailed method descriptions
├── INTER_CLUSTER_AGGREGATION.md # Inter-cluster aggregation explanation
├── results/                     # Training results
│   ├── *.txt                   # Text results
│   ├── *.json                  # JSON results
│   └── *.csv                   # Client statistics
└── visualizations/              # Plots and visualizations
    └── *.png                   # Comparison plots
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{fl_comparison_2025,
  title={FL-comparison: Federated Learning Methods Comparison Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/FL-comparison}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]