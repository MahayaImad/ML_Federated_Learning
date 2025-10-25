"""
FedAvg (Standard Federated Averaging) Training
"""

import time
import numpy as np
from models import initialize_global_model
from communication_cost import CommunicationTracker
from config import VERBOSE


def train_fedavg(clients, test_data, args):
    """
    Train using standard FedAvg algorithm

    Args:
        clients: List of FederatedClient objects
        test_data: (x_test, y_test) tuple
        args: Command line arguments

    Returns:
        results: Dictionary with training metrics
    """
    print("Training with FedAvg (Standard)...")

    # Initialize communication tracker
    comm_tracker = CommunicationTracker(args.dataset)

    # Initialize global model
    global_model = initialize_global_model(args.dataset)

    # Training history
    accuracy_history = []
    loss_history = []
    round_times = []
    communication_costs = []

    num_clients = len(clients)

    # Training loop
    for round_num in range(args.rounds):
        round_start = time.time()
        comm_tracker.reset_round()

        if args.verbose or round_num % 5 == 0:
            print(f"\nRound {round_num + 1}/{args.rounds}")

        # Get global model weights
        global_weights = global_model.get_weights()

        # Server broadcasts to all clients
        comm_tracker.record_client_download(num_clients)

        # Local training on all clients
        client_updates = []
        client_sizes = []

        for client in clients:
            # Update client model with global weights
            client.local_model.set_weights(global_weights)

            # Local training
            client.train_local(args.local_epochs)

            # Collect update
            client_updates.append(client.local_model.get_weights())
            client_sizes.append(len(client.x_train))

        # All clients upload to server
        comm_tracker.record_client_upload(num_clients)

        # FedAvg aggregation
        aggregated_weights = fedavg_aggregate(client_updates, client_sizes)
        global_model.set_weights(aggregated_weights)

        # Evaluation
        x_test, y_test = test_data
        y_test_labels = np.argmax(y_test, axis=1)
        loss, accuracy = global_model.evaluate(x_test, y_test_labels, verbose=0)

        # Finalize communication cost for this round
        round_comm_cost = comm_tracker.finalize_round()

        round_time = time.time() - round_start

        accuracy_history.append(accuracy)
        loss_history.append(loss)
        round_times.append(round_time)
        communication_costs.append(round_comm_cost)

        if args.verbose:
            print(f"  Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, Time: {round_time:.2f}s, Comm: {round_comm_cost:.2f} KB")

    # Print communication summary
    comm_tracker.print_summary()

    return {
        'accuracy_history': accuracy_history,
        'loss_history': loss_history,
        'round_times': round_times,
        'communication_costs': communication_costs,
        'communication_tracker': comm_tracker
    }


def fedavg_aggregate(model_weights_list, data_sizes):
    """
    FedAvg aggregation weighted by data sizes

    Args:
        model_weights_list: List of model weights
        data_sizes: List of data sizes for weighting

    Returns:
        Aggregated weights
    """
    if not model_weights_list:
        return None

    total_size = sum(data_sizes)
    weights_avg = []

    for layer_idx in range(len(model_weights_list[0])):
        weighted_layer = sum(
            (data_sizes[i] / total_size) * model_weights_list[i][layer_idx]
            for i in range(len(model_weights_list))
        )
        weights_avg.append(weighted_layer)

    return weights_avg