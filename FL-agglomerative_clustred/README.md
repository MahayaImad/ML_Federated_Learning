# FL-Agglomerative Clustred

## Description

Compares four federated learning architectures: **Vanilla FL**, **Hierarchical**, **Drop-in**, and **Agglomerative** (JS divergence-based clustering). Evaluates performance and communication efficiency on MNIST, CIFAR-10, and CIFAR-100.

## Key Features

- **4 FL Architectures**: Vanilla FL, Hierarchical, Drop-in, Agglomerative clustering
- **3 Datasets**: MNIST, CIFAR-10, CIFAR-100
- **Data-driven Clustering**: Jensen-Shannon divergence for client grouping
- **IID/Non-IID**: Flexible data distribution (Dirichlet α=0.5)
- **Complete Metrics**: Accuracy, communication cost, training time

## Quick Start

```bash
# Install dependencies
pip install tensorflow numpy matplotlib scikit-learn scipy

# Using Nvidia Containers (Docker must be installed)
Windows: docker run --rm -it --gpus all -v ${PWD}:/app nvcr.io/nvidia/tensorflow:24.07-tf2-py3 bash
Linux: docker run --rm -it --gpus all -v ${pwd}:/app nvcr.io/nvidia/tensorflow:24.07-tf2-py3 bash
Then : /app 

# Vanilla FL baseline
python main.py --hierarchy-type vanilla --dataset mnist --clients 20 --rounds 20

# Hierarchical FL
python main.py --hierarchy-type hierarchical --dataset cifar10 \
    --clients 20 --edge-servers 5 --rounds 20
    
# Drop-in FL
python main.py --hierarchy-type drop-in --dataset cifar10 \
    --clients 30 --edge-servers 6 --rounds 10
    
# Agglomerative clustering (data-driven)
python main.py --hierarchy-type agglomerative --dataset cifar100 \
    --clients 30 --js-threshold 0.5 --selection-ratio 0.3 --rounds 25

# Compare all methods
python main.py --hierarchy-type compare --dataset mnist \
    --clients 20 --edge-servers 5 --rounds 20 --verbose
```

## FL Architectures

### 1. Vanilla FL
Direct client-server communication with FedAvg aggregation.

### 2. Hierarchical FL
Two-tier: clients → edge servers → cloud.

### 3. Drop-in Hierarchical
Hierarchical + shared clients across edges for robustness.

### 4. Agglomerative FL
**Data-driven clustering** using Jensen-Shannon divergence:
- Computes client label distribution similarity
- Forms clusters automatically (JS threshold)
- Selects top clients per cluster (by data size)
- Optimal for non-IID scenarios

##  Command-Line Arguments

| Argument | Options | Default      | Description |
|----------|---------|--------------|-------------|
| `--hierarchy-type` | vanilla, hierarchical, drop-in, agglomerative, compare | **required** | FL architecture |
| `--dataset` | mnist, cifar10, cifar100 | **required** | Dataset to use |
| `--clients` | int | 20           | Number of clients |
| `--edge-servers` | int | 5            | Edge servers (hierarchical/drop-in) |
| `--js-threshold` | float | 0.5          | JS clustering threshold |
| `--selection-ratio` | float | 0.3          | % clients per cluster |
| `--rounds` | int | 20           | Communication rounds |
| `--local-epochs` | int | 5            | Local training epochs |
| `--batch-size` | int | 64            | Batch size |
| `--iid` | flag | False        | IID distribution |
| `--gpu` | int | 0            | GPU id (-1=CPU) |

## Evaluated Metrics

### Performance Metrics
- **Final Accuracy**: Test set accuracy after training
- **Accuracy History**: Per-round accuracy evolution
- **Training Stability**: Variance in accuracy across rounds

### Communication Metrics
- **Edge Communication**: Client ↔ Edge server transfers
- **Global Communication**: Edge server ↔ Cloud transfers
- **Total Communication Cost**: Combined network overhead
- **Communication Efficiency**: Accuracy per communication unit

### Computational Metrics
- **Round Time**: Duration per communication round
- **Total Training Time**: End-to-end training duration
- **Aggregation Time**: Time spent on model aggregation
- **Average Time per Round**: Mean round duration


### Generated Files

#### Results Directory (`results/`)
- **Text format**: `[timestamp]_[dataset]_[type]_[config].txt`
  - Detailed metrics per round
  - Final summary statistics


#### Visualizations Directory (`visualizations/`)
- **Comparison plots**: 4-panel charts showing:
  1. Accuracy evolution over rounds
  2. Training time per round
  3. Communication cost per round
  4. Summary statistics

## Agglomerative Clustering

Uses **Jensen-Shannon divergence** to measure client data similarity:

```
JS(P||Q) = √[0.5 × KL(P||M) + 0.5 × KL(Q||M)]
where M = 0.5 × (P + Q)
```

**Threshold Selection:**
- `--js-threshold 0.2`: Many small clusters (fine-grained)
- `--js-threshold 0.5`: Balanced (recommended)
- `--js-threshold 0.7`: Few large clusters (coarse)

**Client Selection:**
- `--selection-ratio 0.3`: Top 30% by data volume
- `--selection-ratio 1.0`: All clients

##  Output Files

**Results** (`results/`):
- `[timestamp]_[dataset]_[type].txt`: Detailed metrics

**Visualizations** (`visualizations/`):
- 4-panel plots: accuracy, time, communication, summary