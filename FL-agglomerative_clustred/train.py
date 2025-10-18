
import time
import numpy as np
from models import initialize_global_model, initialize_edge_models
from hierarchical_server import get_cyclic_partner
from utils import save_client_stats_csv
from config import SAVE_CLIENTS_STATS, VERBOSE

def train_vanilla_fl(clients, test_data, args):
    """Entraînement FL vanilla (FedAvg classique)"""
    print(" Train of Vanilla FL (FedAvg)...")

    global_model = initialize_global_model(args.dataset)

    results = {
        'accuracy_history': [],
        'communication_costs': [],
        'round_times': []
    }

    for round_num in range(args.rounds):
        start_time = time.time()

        if VERBOSE:
            print(f"  Round {round_num + 1}/{args.rounds}")

        # Local training
        client_updates = []
        client_sizes = []

        for client in clients:
            client.update_model(global_model)
            client.train_local(args.local_epochs)
            weights = client.local_model.get_weights()
            client_updates.append(weights)
            client_sizes.append(len(client.x_train))

        # Aggregation FedAvg
        global_weights = fedavg_aggregate(client_updates, client_sizes)
        global_model.set_weights(global_weights)

        # Evaluation
        x_test, y_test = test_data
        y_test = np.argmax(y_test, axis=1)
        _, accuracy = global_model.evaluate(x_test, y_test, verbose=0)

        # Métriques
        comm_cost = len(clients) * 2  # Upload + download pour chaque client
        round_time = time.time() - start_time

        results['accuracy_history'].append(accuracy)
        results['communication_costs'].append(comm_cost)
        results['round_times'].append(round_time)

        if VERBOSE:
            print(f"    Accuracy: {accuracy:.4f}, Comm Cost: {comm_cost}, Time: {round_time:.2f}s")

    # Save client statistics to CSV
    if SAVE_CLIENTS_STATS:
        save_client_stats_csv(clients, args)

    return results


def train_hierarchical(clients, test_data, edge_servers, hierarchical_server, args):

    print(f" Train of FL {args.hierarchy_type}...")

    global_model = initialize_global_model(args.dataset)
    initialize_edge_models(edge_servers, args.dataset, global_model)

    accuracy_history = []
    communication_costs = []
    round_times = []

    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        round_start_time = time.time()
        communication_cost = 0

        # ÉTAPE 1: Diffuser le modèle global vers les edge servers
        for edge_server in edge_servers:
            edge_server.set_global_model(global_model.get_weights())
            communication_cost += 1  # Global -> Edge

        # ÉTAPE 2: Chaque edge server entraîne ses clients et agrège localement
        edge_updates = []
        edge_weights = []

        for edge_server in edge_servers:
            # Clients de cet edge server
            edge_clients = [client for client in clients if client.client_id in edge_server.client_ids]

            if not edge_clients:
                continue

            # Entraînement local des clients de cet edge server
            client_updates = []
            client_data_sizes = []

            for client in edge_clients:
                # Mettre à jour avec le modèle de l'edge server
                client.update_model(edge_server.local_model)

                # Entraînement local
                client.train_local(args.local_epochs)
                communication_cost += 2  # Edge -> Client -> Edge

                client_updates.append(client.local_model.get_weights())
                client_data_sizes.append(len(client.x_train))

            # ÉTAPE 3: Agrégation au niveau de l'edge server
            if client_updates:
                edge_aggregated = fedavg_aggregate(client_updates, client_data_sizes)
                edge_server.local_model.set_weights(edge_aggregated)

                edge_updates.append(edge_aggregated)
                edge_weights.append(sum(client_data_sizes))  # Poids = total des données

        # ÉTAPE 4: Agrégation finale au serveur global
        if edge_updates:
            global_aggregated = fedavg_aggregate(edge_updates, edge_weights)
            global_model.set_weights(global_aggregated)

            # Communication Edge -> Global
            communication_cost += len(edge_servers)

        # ÉTAPE 5: Évaluation
        x_test, y_test = test_data
        y_test = np.argmax(y_test, axis=1)
        test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)

        round_time = time.time() - round_start_time

        accuracy_history.append(test_acc)
        communication_costs.append(communication_cost)
        round_times.append(round_time)

        print(f"  Accuracy: {test_acc:.4f}, Time: {round_time:.2f}s, Comm: {communication_cost}")

    # Save client statistics to CSV
    if SAVE_CLIENTS_STATS:
        save_client_stats_csv(clients, args)

    return {
        'method': 'hierarchical',
        'accuracy_history': accuracy_history,
        'communication_costs': communication_costs,
        'round_times': round_times
    }


def train_cyclic_agglomerative(clients, test_data, edge_servers, hierarchical_server, args):
    """
    Cyclic-Agglomerative training with dynamic edge server aggregation
    At each round, edge servers aggregate their clients + a cyclic partner edge server
    """
    print(f" Train of FL Cyclic-Agglomerative...")

    global_model = initialize_global_model(args.dataset)
    initialize_edge_models(edge_servers, args.dataset, global_model)

    accuracy_history = []
    communication_costs = []
    round_times = []

    num_edges = len(edge_servers)

    for round_num in range(args.rounds):
        print(f"Round {round_num + 1}/{args.rounds}")
        round_start_time = time.time()
        communication_cost = 0

        # ÉTAPE 1: Diffuser le modèle global vers les edge servers
        for edge_server in edge_servers:
            edge_server.set_global_model(global_model.get_weights())
            communication_cost += 1  # Global -> Edge

        # ÉTAPE 2: Chaque edge server entraîne ses clients
        edge_client_aggregates = {}  # Store aggregated client models per edge
        edge_data_sizes = {}

        for edge_server in edge_servers:
            edge_clients = [client for client in clients if client.client_id in edge_server.client_ids]

            if not edge_clients:
                continue

            # Entraînement local des clients
            client_updates = []
            client_data_sizes = []

            for client in edge_clients:
                client.update_model(edge_server.local_model)
                client.train_local(args.local_epochs)
                communication_cost += 2  # Edge -> Client -> Edge

                client_updates.append(client.local_model.get_weights())
                client_data_sizes.append(len(client.x_train))

            # Agrégation des clients de cet edge
            if client_updates:
                edge_client_aggregates[edge_server.edge_id] = fedavg_aggregate(client_updates, client_data_sizes)
                edge_data_sizes[edge_server.edge_id] = sum(client_data_sizes)

        # ÉTAPE 3: Agrégation cyclique entre edge servers
        edge_updates = []
        edge_weights = []

        for edge_server in edge_servers:
            edge_id = edge_server.edge_id

            if edge_id not in edge_client_aggregates:
                continue

            # Get cyclic partner
            partner_id = get_cyclic_partner(edge_id, round_num, num_edges)

            # Prepare models and weights for aggregation
            models_to_aggregate = [edge_client_aggregates[edge_id]]
            weights_to_aggregate = [edge_data_sizes[edge_id]]

            # Add partner's model if available
            if partner_id in edge_client_aggregates and partner_id != edge_id:
                models_to_aggregate.append(edge_client_aggregates[partner_id])
                weights_to_aggregate.append(edge_data_sizes[partner_id])
                communication_cost += 1  # Edge-to-edge communication

                if VERBOSE:
                    print(f"    Edge {edge_id} aggregating with partner Edge {partner_id}")

            # Aggregate current edge + cyclic partner
            edge_aggregated = fedavg_aggregate(models_to_aggregate, weights_to_aggregate)
            edge_server.local_model.set_weights(edge_aggregated)

            edge_updates.append(edge_aggregated)
            edge_weights.append(sum(weights_to_aggregate))

        # ÉTAPE 4: Agrégation finale au serveur global
        if edge_updates:
            global_aggregated = fedavg_aggregate(edge_updates, edge_weights)
            global_model.set_weights(global_aggregated)

            # Communication Edge -> Global
            communication_cost += len(edge_servers)

        # ÉTAPE 5: Évaluation
        x_test, y_test = test_data
        y_test = np.argmax(y_test, axis=1)
        test_loss, test_acc = global_model.evaluate(x_test, y_test, verbose=0)

        round_time = time.time() - round_start_time

        accuracy_history.append(test_acc)
        communication_costs.append(communication_cost)
        round_times.append(round_time)

        print(f"  Accuracy: {test_acc:.4f}, Time: {round_time:.2f}s, Comm: {communication_cost}")

    # Save client statistics to CSV
    if SAVE_CLIENTS_STATS:
        save_client_stats_csv(clients, args)

    return {
        'method': 'cyclic-agglomerative',
        'accuracy_history': accuracy_history,
        'communication_costs': communication_costs,
        'round_times': round_times
    }

def fedavg_aggregate(model_weights_list, data_sizes):
    """Agrégation FedAvg pondérée par la taille des données"""
    if not model_weights_list:
        return None

    total_size = sum(data_sizes)
    weights_avg = []

    # Weight each model by its data size
    for layer_idx in range(len(model_weights_list[0])):
        weighted_layer = sum(
            (data_sizes[i] / total_size) * model_weights_list[i][layer_idx]
            for i in range(len(model_weights_list))
        )
        weights_avg.append(weighted_layer)

    return weights_avg

