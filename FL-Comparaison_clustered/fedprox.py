"""
FedProx Training
Adds proximal term to local objective to handle system heterogeneity
"""

import time
import numpy as np
import tensorflow as tf
from models import initialize_global_model
from communication_cost import CommunicationTracker
from config import VERBOSE


def train_fedprox(clients, test_data, args):
    """
    Train using FedProx algorithm

    Args:
        clients: List of FederatedClient objects
        test_data: (x_test, y_test) tuple
        args: Command line arguments

    Returns:
        results: Dictionary with training metrics
    """
    print(f"Training with FedProx (mu={args.mu})...")

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

        # Local training on all clients with proximal term
        client_updates = []
        client_sizes = []

        for client in clients:
            # Update client model with global weights
            client.local_model.set_weights(global_weights)

            # Local training with proximal term
            train_fedprox_client(
                client,
                global_weights,
                args.mu,
                args.local_epochs
            )

            # Collect update
            client_updates.append(client.local_model.get_weights())
            client_sizes.append(len(client.x_train))

        # All clients upload to server
        comm_tracker.record_client_upload(num_clients)

        # FedAvg aggregation (same as FedAvg)
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
        'communication_tracker': comm_tracker,
        'mu': args.mu
    }


def train_fedprox_client(client, global_weights, mu, local_epochs):
    """
    Train a client with FedProx proximal term

    Args:
        client: FederatedClient object
        global_weights: Global model weights
        mu: Proximal term coefficient
        local_epochs: Number of local epochs
    """
    model = client.local_model

    # Custom training loop with proximal term
    for epoch in range(local_epochs):
        for x_batch, y_batch in client.train_dataset:
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(x_batch, training=True)

                # Standard loss
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    tf.argmax(y_batch, axis=1),
                    predictions
                )
                loss = tf.reduce_mean(loss)

                # Add proximal term: (mu/2) * ||w - w_global||^2
                if mu > 0:
                    proximal_term = 0.0
                    current_weights = model.trainable_variables

                    for w_local, w_global in zip(current_weights, global_weights):
                        if len(w_global.shape) > 0:  # Skip scalar weights
                            proximal_term += tf.reduce_sum(tf.square(w_local - w_global))

                    proximal_term = (mu / 2.0) * proximal_term
                    loss = loss + proximal_term

            # Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def fedavg_aggregate(model_weights_list, data_sizes):
    """FedAvg aggregation weighted by data sizes"""
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