"""
FedLoss Training - Loss-Based Client Selection
Selects clients with highest loss for targeted training
"""

import time
import numpy as np
from models import initialize_global_model
from communication_cost import CommunicationTracker
from config import VERBOSE


def train_fedloss(clients, test_data, args):
    """
    Train using FedLoss algorithm

    Round 1: Train all clients
    Round 2+: Select clients with highest loss (bottom performers)

    Args:
        clients: List of FederatedClient objects
        test_data: (x_test, y_test) tuple
        args: Command line arguments

    Returns:
        results: Dictionary with training metrics
    """
    print(f"Training with FedLoss (selection_ratio={args.selection_ratio})...")
    print("Round 1: Training ALL clients to establish baseline")
    print(f"Round 2+: Selecting top {int(args.selection_ratio * 100)}% clients with HIGHEST loss")

    # Initialize communication tracker
    comm_tracker = CommunicationTracker(args.dataset)

    # Initialize global model
    global_model = initialize_global_model(args.dataset)

    # Training history
    accuracy_history = []
    loss_history = []
    round_times = []
    communication_costs = []
    client_losses_history = []  # Track client losses over rounds
    selected_clients_history = []  # Track which clients were selected

    num_clients = len(clients)

    # Track client losses (initialized to None)
    client_losses = {i: None for i in range(num_clients)}

    # Training loop
    for round_num in range(args.rounds):
        round_start = time.time()
        comm_tracker.reset_round()

        if args.verbose or round_num % 5 == 0:
            print(f"\nRound {round_num + 1}/{args.rounds}")

        # Get global model weights
        global_weights = global_model.get_weights()

        # SELECTION LOGIC
        if round_num == 0:
            # Round 1: Train ALL clients
            selected_clients = list(range(num_clients))
            if args.verbose:
                print(f"  Training ALL {num_clients} clients (baseline round)")
        else:
            # Round 2+: Select clients with HIGHEST loss
            num_to_select = max(1, int(num_clients * args.selection_ratio))
            selected_clients = select_clients_by_highest_loss(
                client_losses,
                num_to_select
            )
            if args.verbose:
                print(f"  Selected {len(selected_clients)}/{num_clients} clients with highest loss")
                print(f"  Selected client IDs: {selected_clients[:10]}{'...' if len(selected_clients) > 10 else ''}")

        # Server broadcasts to selected clients
        comm_tracker.record_client_download(len(selected_clients))

        # Local training on selected clients
        client_updates = []
        client_sizes = []

        for client_id in selected_clients:
            client = clients[client_id]

            # Update client model with global weights
            client.local_model.set_weights(global_weights)

            # Local training
            client.train_local(args.local_epochs)

            # Collect update
            client_updates.append(client.local_model.get_weights())
            client_sizes.append(len(client.x_train))

        # Selected clients upload to server
        comm_tracker.record_client_upload(len(selected_clients))

        # FedAvg aggregation
        aggregated_weights = fedavg_aggregate(client_updates, client_sizes)
        global_model.set_weights(aggregated_weights)

        # COMPUTE LOSS FOR ALL CLIENTS (for next round selection)
        new_client_losses = {}
        for client_id in range(num_clients):
            client = clients[client_id]

            # Update client with global model
            client.local_model.set_weights(global_model.get_weights())

            # Compute loss on client's local data
            y_train_labels = np.argmax(client.y_train, axis=1)
            loss, _ = client.local_model.evaluate(
                client.x_train,
                y_train_labels,
                verbose=0
            )
            new_client_losses[client_id] = loss

        client_losses = new_client_losses

        # Evaluation on test set
        x_test, y_test = test_data
        y_test_labels = np.argmax(y_test, axis=1)
        test_loss, test_accuracy = global_model.evaluate(
            x_test, y_test_labels, verbose=0
        )

        # Finalize communication cost for this round
        round_comm_cost = comm_tracker.finalize_round()

        round_time = time.time() - round_start

        accuracy_history.append(test_accuracy)
        loss_history.append(test_loss)
        round_times.append(round_time)
        communication_costs.append(round_comm_cost)
        client_losses_history.append(client_losses.copy())
        selected_clients_history.append(selected_clients.copy())

        if args.verbose or round_num % 5 == 0:
            avg_client_loss = np.mean(list(client_losses.values()))
            max_client_loss = max(client_losses.values())
            min_client_loss = min(client_losses.values())
            print(f"  Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
            print(
                f"  Client Losses - Avg: {avg_client_loss:.4f}, Max: {max_client_loss:.4f}, Min: {min_client_loss:.4f}")
            print(f"  Time: {round_time:.2f}s, Comm: {round_comm_cost:.2f} KB")

    # Print communication summary
    comm_tracker.print_summary()

    # Print final client loss statistics
    print_client_loss_summary(client_losses, selected_clients_history[-1])

    return {
        'accuracy_history': accuracy_history,
        'loss_history': loss_history,
        'round_times': round_times,
        'communication_costs': communication_costs,
        'communication_tracker': comm_tracker,
        'client_losses_history': client_losses_history,
        'selected_clients_history': selected_clients_history,
        'selection_ratio': args.selection_ratio
    }


def select_clients_by_highest_loss(client_losses, num_to_select):
    """
    Select clients with highest loss (worst performers)

    Args:
        client_losses: Dict mapping client_id to loss value
        num_to_select: Number of clients to select

    Returns:
        List of selected client IDs
    """
    # Sort clients by loss (descending - highest loss first)
    sorted_clients = sorted(
        client_losses.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Select top num_to_select clients with highest loss
    selected = [client_id for client_id, _ in sorted_clients[:num_to_select]]

    return selected


def print_client_loss_summary(client_losses, selected_clients):
    """
    Print summary of client losses

    Args:
        client_losses: Dict mapping client_id to loss
        selected_clients: List of selected client IDs
    """
    print("\n" + "=" * 60)
    print("CLIENT LOSS SUMMARY (Final Round)")
    print("=" * 60)

    losses = list(client_losses.values())

    print(f"Total Clients: {len(client_losses)}")
    print(f"Selected Clients: {len(selected_clients)}")
    print(f"\nLoss Statistics:")
    print(f"  Mean: {np.mean(losses):.4f}")
    print(f"  Std:  {np.std(losses):.4f}")
    print(f"  Min:  {np.min(losses):.4f}")
    print(f"  Max:  {np.max(losses):.4f}")
    print(f"  Median: {np.median(losses):.4f}")

    # Top 5 worst performers (highest loss)
    sorted_clients = sorted(
        client_losses.items(),
        key=lambda x: x[1],
        reverse=True
    )
    print(f"\nTop 5 Clients with HIGHEST Loss (selected for training):")
    for i, (client_id, loss) in enumerate(sorted_clients[:5], 1):
        selected_marker = "✓" if client_id in selected_clients else " "
        print(f"  {i}. Client {client_id}: {loss:.4f} [{selected_marker}]")

    # Top 5 best performers (lowest loss)
    print(f"\nTop 5 Clients with LOWEST Loss (not selected):")
    for i, (client_id, loss) in enumerate(reversed(sorted_clients[-5:]), 1):
        print(f"  {i}. Client {client_id}: {loss:.4f}")

    print("=" * 60)


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