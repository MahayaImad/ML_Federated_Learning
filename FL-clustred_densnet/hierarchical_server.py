
from aggregator import FedAvgAggregator
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from utils import jensen_shannon_distance, select_clients_by_samples


class EdgeServer:
    def __init__(self, edge_id, client_ids):
        self.edge_id = edge_id
        self.client_ids = client_ids
        self.aggregator = FedAvgAggregator()
        self.local_model = None

    def set_global_model(self, weights):
        """Reçoit les poids du modèle global"""
        if self.local_model is None:
            # Créer le modèle local lors de la première fois
            pass
        else:
            self.local_model.set_weights(weights)

    def aggregate_clients(self, client_updates):
        """Agrège les mises à jour des clients locaux"""
        edge_updates = []
        edge_sizes = []

        for client_id in self.client_ids:
            if client_id in client_updates:
                edge_updates.append(client_updates[client_id]['weights'])
                edge_sizes.append(client_updates[client_id]['data_size'])

        if edge_updates:
            self.local_model = self.aggregator.aggregate(edge_updates, edge_sizes)

        return self.local_model


class HierarchicalServer:
    def __init__(self, edge_servers):
        self.edge_servers = edge_servers
        self.global_aggregator = FedAvgAggregator()
        self.global_model = None

    def aggregate_edges(self):
        """Agrège les modèles des edge servers"""
        edge_models = []
        edge_sizes = []

        for edge_server in self.edge_servers:
            if edge_server.local_model is not None:
                edge_models.append(edge_server.local_model)
                edge_sizes.append(len(edge_server.client_ids))

        if edge_models:
            self.global_model = self.global_aggregator.aggregate(edge_models, edge_sizes)

        return self.global_model


def setup_vanilla_fl():
    """Vanilla FL has no hierarchy"""
    return None, None


def setup_standard_hierarchy(clients_data, num_edge_servers, verbose=False):
    """
    Standard hierarchical FL - no overlap between edge servers
    Each client belongs to exactly one edge server
    """
    total_clients = len(clients_data)
    clients_per_edge = total_clients // num_edge_servers
    edge_servers = []

    for i in range(num_edge_servers):
        start_idx = i * clients_per_edge
        end_idx = min(start_idx + clients_per_edge, total_clients)
        client_ids = list(range(start_idx, end_idx))
        edge_servers.append(EdgeServer(i, client_ids))

        if verbose:
            print(f"  Edge Server {i}: {len(client_ids)} clients : {client_ids}")

    return edge_servers, HierarchicalServer(edge_servers)


def setup_agglomerative_hierarchy(clients_data, js_threshold,
                                  selection_ratio, num_classes, verbose=False):
    """
    Agglomerative clustering based on Jensen-Shannon distance
    Groups similar clients together
    """
    # Step 1: Calculate label histograms
    client_histograms = []
    for x_train, y_train in clients_data:
        labels = np.argmax(y_train, axis=1)
        histogram = np.zeros(num_classes)
        unique, counts = np.unique(labels, return_counts=True)
        histogram[unique] = counts
        histogram = histogram / histogram.sum()
        client_histograms.append(histogram)

    # Step 2: Build distance matrix
    n_clients = len(client_histograms)
    js_matrix = np.zeros((n_clients, n_clients))

    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            js_dist = jensen_shannon_distance(client_histograms[i],
                                              client_histograms[j])
            js_matrix[i, j] = js_matrix[j, i] = js_dist

    if verbose:
        _print_js_statistics(js_matrix)

    # Step 3: Clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=js_threshold
    )
    cluster_labels = clustering.fit_predict(js_matrix)

    # Step 4: Create edge servers from clusters
    clusters = {}
    for client_idx, cluster_id in enumerate(cluster_labels):
        clusters.setdefault(cluster_id, []).append(client_idx)

    edge_servers = []
    for edge_id, client_ids in enumerate(clusters.values()):
        # Select top clients by sample size
        selected = select_clients_by_samples(
            client_ids, clients_data, selection_ratio
        )
        edge_servers.append(EdgeServer(edge_id, selected))

        if verbose:
            print(f"  Edge {edge_id}: {len(selected)}/{len(client_ids)} clients : {client_ids}")

    if verbose:
        print(f"  Total clusters: {len(clusters)}, JS threshold: {js_threshold}")

    return edge_servers, HierarchicalServer(edge_servers)


def setup_dropin_hierarchy(clients_data, num_edge_servers, verbose=False):
    """
    Drop-in hierarchical FL - clients can appear in multiple edge servers
    Adds redundancy for robustness
    """
    total_clients = len(clients_data)
    clients_per_edge = total_clients // num_edge_servers
    remaining = total_clients % num_edge_servers

    # Step 1: Base assignment (no overlap)
    base_assignments = []
    client_idx = 0

    for i in range(num_edge_servers):
        count = clients_per_edge + (1 if i < remaining else 0)
        client_ids = list(range(client_idx, client_idx + count))
        base_assignments.append(client_ids)
        client_idx += count

    # Step 2: Add drop-in duplicates
    edge_servers = []
    for i in range(num_edge_servers):
        client_ids = base_assignments[i].copy()

        # Get clients from other edges
        other_clients = [c for j, clients in enumerate(base_assignments)
                         if j != i for c in clients]

        # Add 2-3 random duplicates
        if other_clients:
            num_dups = min(3, len(other_clients))
            duplicates = np.random.choice(other_clients, num_dups, replace=False)
            client_ids.extend(duplicates.tolist())

        edge_servers.append(EdgeServer(i, client_ids))

        if verbose:
            print(f"  Edge {i}: {len(client_ids)} clients (with duplicates) : {client_ids}")

    return edge_servers, HierarchicalServer(edge_servers)


def _print_js_statistics(js_matrix):
    """Helper to print JS distance statistics"""
    js_values = js_matrix[np.triu_indices_from(js_matrix, k=1)]
    print(f"\n  JS Distance Stats:")
    print(f"    Min: {js_values.min():.4f}, Max: {js_values.max():.4f}")
    print(f"    Mean: {js_values.mean():.4f}, Median: {np.median(js_values):.4f}")