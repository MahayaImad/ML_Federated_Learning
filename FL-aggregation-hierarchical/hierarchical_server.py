# Ajouter à FL-aggregation-hierarchical/hierarchical_server.py
from aggregations.fedavg import FedAvgAggregator
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering


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
