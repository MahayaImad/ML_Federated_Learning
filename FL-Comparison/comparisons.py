"""
Implémentation des comparaisons FL vs autres méthodes
"""
import time
import tensorflow as tf
import numpy as np
from models import create_comparison_model


def compare_fl_vs_central(data_splits, args):
    """Compare apprentissage fédéré vs centralisé"""
    print("🔄 Comparaison FL vs Centralisé...")

    results = {}

    # 1. Entraînement centralisé
    print("📊 Entraînement centralisé...")
    centralized_result = train_centralized(data_splits, args)
    results['Centralized'] = centralized_result

    # 2. Entraînement fédéré
    print("🌐 Entraînement fédéré...")
    federated_result = train_federated(data_splits, args)
    results['Federated'] = federated_result

    return results


def compare_fl_vs_distributed(data_splits, args):
    """Compare apprentissage fédéré vs distribué"""
    print("🔄 Comparaison FL vs Distribué...")

    results = {}

    # 1. Entraînement fédéré
    print("🌐 Entraînement fédéré...")
    federated_result = train_federated(data_splits, args)
    results['Federated'] = federated_result

    # 2. Entraînement distribué
    print("🔗 Entraînement distribué...")
    distributed_result = train_distributed(data_splits, args)
    results['Distributed'] = distributed_result

    return results


def compare_all_methods(data_splits, args):
    """Compare toutes les méthodes d'apprentissage"""
    print("🔄 Comparaison complète de toutes les méthodes...")

    results = {}

    # 1. Centralisé
    print("📊 Entraînement centralisé...")
    results['Centralized'] = train_centralized(data_splits, args)

    # 2. Fédéré standard (FedAvg)
    print("🌐 Entraînement fédéré (FedAvg)...")
    results['FL_FedAvg'] = train_federated(data_splits, args)

    # 3. Distribué
    print("🔗 Entraînement distribué...")
    results['Distributed'] = train_distributed(data_splits, args)

    # 4. Cyclic FL
    print("🔄 Entraînement fédéré cyclique pondéré...")
    results['FL_Cyclic'] = train_cyclic_fl(data_splits, args)

    # 5. Ensemble FL
    print("🎯 Entraînement fédéré ensemble...")
    results['FL_Ensemble'] = train_ensemble_fl(data_splits, args)

    return results


def train_centralized(data_splits, args):
    """Entraînement centralisé standard"""
    start_time = time.time()

    # Créer le modèle
    model = create_comparison_model(
        args.dataset, data_splits['input_shape'], data_splits['num_classes']
    )

    # Compiler
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entraîner
    history = model.fit(
        data_splits['centralized']['train'],
        epochs=args.epochs,
        validation_data=data_splits['centralized']['test'],
        verbose=1 if args.verbose else 0
    )

    # Évaluer
    test_loss, test_acc = model.evaluate(
        data_splits['centralized']['test'], verbose=0
    )

    training_time = time.time() - start_time

    return {
        'final_accuracy': test_acc,
        'training_time': training_time,
        'history': history.history,
        'communication_cost': 0,  # Pas de communication
        'convergence_round': len(history.history['accuracy'])
    }


def train_federated(data_splits, args):
    """Entraînement fédéré simple (FedAvg)"""
    start_time = time.time()

    # Modèle global
    global_model = create_comparison_model(
        args.dataset, data_splits['input_shape'], data_splits['num_classes']
    )

    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    client_datasets = data_splits['federated']['clients']
    test_data = data_splits['federated']['test']

    accuracy_history = []
    communication_cost = 0

    # Tours de communication
    for round_num in range(args.epochs):
        round_weights = []
        round_sizes = []

        # Entraînement local pour chaque client
        for client_data in client_datasets:
            # Copier le modèle global
            client_model = tf.keras.models.clone_model(global_model)
            client_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            client_model.set_weights(global_model.get_weights())

            # Entraîner localement (1 époque par tour)
            client_model.fit(client_data, epochs=1, verbose=0)

            # Collecter les poids et taille
            round_weights.append(client_model.get_weights())
            round_sizes.append(sum(1 for _ in client_data))

        # Agrégation FedAvg
        aggregated_weights = fedavg_aggregate(round_weights, round_sizes)
        global_model.set_weights(aggregated_weights)

        # Évaluer
        _, accuracy = global_model.evaluate(test_data, verbose=0)
        accuracy_history.append(accuracy)

        # Calculer coût de communication
        num_params = sum(np.prod(w.shape) for w in global_model.get_weights())
        communication_cost += len(client_datasets) * 2  # Upload + download

        if args.verbose:
            print(f"  Round {round_num + 1}: Accuracy = {accuracy:.4f}")

    training_time = time.time() - start_time

    return {
        'final_accuracy': accuracy_history[-1] if accuracy_history else 0.0,
        'training_time': training_time,
        'history': {'accuracy': accuracy_history},
        'communication_cost': communication_cost,
        'convergence_round': len(accuracy_history)
    }


def train_distributed(data_splits, args):
    """Entraînement distribué (parallèle sans coordination)"""
    start_time = time.time()

    node_datasets = data_splits['distributed']['nodes']
    test_data = data_splits['distributed']['test']

    node_models = []
    node_accuracies = []

    # Entraîner chaque nœud indépendamment
    for i, node_data in enumerate(node_datasets):
        if args.verbose:
            print(f"  Entraînement nœud {i + 1}/{len(node_datasets)}")

        # Créer modèle pour ce nœud
        node_model = create_comparison_model(
            args.dataset, data_splits['input_shape'], data_splits['num_classes']
        )

        node_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Entraîner
        node_model.fit(node_data, epochs=args.epochs, verbose=0)

        # Évaluer
        _, accuracy = node_model.evaluate(test_data, verbose=0)
        node_models.append(node_model)
        node_accuracies.append(accuracy)

    training_time = time.time() - start_time

    # Moyenne des performances
    avg_accuracy = np.mean(node_accuracies)
    std_accuracy = np.std(node_accuracies)

    return {
        'final_accuracy': avg_accuracy,
        'accuracy_std': std_accuracy,
        'training_time': training_time,
        'history': {'accuracy': node_accuracies},
        'communication_cost': 0,  # Pas de communication entre nœuds
        'convergence_round': args.epochs
    }


def train_cyclic_fl(data_splits, args):
    """
    CyclicFL: Deux-phases federated learning avec pre-training
    Phase P1: Cyclic Pre-Training (sequential client training)
    Phase P2: Standard Federated Training (parallel aggregation)
    """
    start_time = time.time()

    # Initialize global model
    global_model = create_comparison_model(
        args.dataset, data_splits['input_shape'], data_splits['num_classes']
    )

    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    client_datasets = data_splits['federated']['clients']
    test_data = data_splits['federated']['test']
    num_clients = len(client_datasets)

    accuracy_history = []
    communication_cost = 0

    # CyclicFL Parameters
    pretraining_rounds = max(5, args.epochs // 3)  # T rounds for P1 (1/3 of total epochs)
    federated_rounds = args.epochs - pretraining_rounds  # Remaining for P2
    clients_per_round = max(2, num_clients // 2)  # Subset size for selection
    local_steps_p1 = 2  # Fixed local steps in pre-training
    local_steps_p2 = 1  # Local steps in federated phase

    if args.verbose:
        print(f"  CyclicFL: P1={pretraining_rounds} rounds, P2={federated_rounds} rounds")

    # =================================================================
    # PHASE P1: CYCLIC PRE-TRAINING (Sequential Training)
    # =================================================================
    if args.verbose:
        print("  🔄 Phase P1: Cyclic Pre-Training...")

    for round_num in range(pretraining_rounds):
        # Step 2: Random client selection for this pre-training round
        selected_clients = np.random.choice(
            num_clients,
            size=min(clients_per_round, num_clients),
            replace=False
        )

        if args.verbose:
            print(f"    P1 Round {round_num + 1}: Selected clients {selected_clients}")

        # Step 3: Sequential Training (Cyclic Process)
        current_model = global_model

        for client_idx in selected_clients:
            client_data = client_datasets[client_idx]

            # Clone current model for this client
            client_model = tf.keras.models.clone_model(current_model)
            client_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            client_model.set_weights(current_model.get_weights())

            # Train for fixed number of local steps
            client_model.fit(client_data, epochs=local_steps_p1, verbose=0)

            # Update current model (sequential passing)
            current_model.set_weights(client_model.get_weights())

            # Communication cost (send + receive)
            communication_cost += 2

        # Step 4: Updated model becomes starting point for next round
        global_model.set_weights(current_model.get_weights())

        # Evaluate pre-training progress
        _, accuracy = global_model.evaluate(test_data, verbose=0)
        accuracy_history.append(accuracy)

        if args.verbose:
            print(f"    P1 Round {round_num + 1}: Accuracy = {accuracy:.4f}")

    # Step 5: Well-initialized model obtained
    pretrained_model = global_model
    if args.verbose:
        print(f"  ✅ P1 Complete: Pre-trained model accuracy = {accuracy_history[-1]:.4f}")

    # =================================================================
    # PHASE P2: STANDARD FEDERATED TRAINING (Parallel Aggregation)
    # =================================================================
    if args.verbose:
        print("  🌐 Phase P2: Standard Federated Training...")

    for round_num in range(federated_rounds):
        # Step 1: Client selection for federated round
        selected_clients = np.random.choice(
            num_clients,
            size=min(clients_per_round, num_clients),
            replace=False
        )

        # Step 2: Parallel training on selected clients
        round_weights = []
        round_sizes = []

        for client_idx in selected_clients:
            client_data = client_datasets[client_idx]

            # Clone global model
            client_model = tf.keras.models.clone_model(global_model)
            client_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            client_model.set_weights(global_model.get_weights())

            # Local training
            client_model.fit(client_data, epochs=local_steps_p2, verbose=0)

            # Collect weights and data size for aggregation
            data_size = sum(1 for _ in client_data)
            round_weights.append(client_model.get_weights())
            round_sizes.append(data_size)

            # Communication cost
            communication_cost += 2

        # Step 3: Aggregation (weighted averaging by data size)
        if round_weights:
            total_size = sum(round_sizes)
            normalized_weights = [size / total_size for size in round_sizes]

            # Weighted averaging
            new_weights = []
            for layer_idx in range(len(round_weights[0])):
                weighted_layer = sum(
                    weight * round_weights[client_idx][layer_idx]
                    for client_idx, weight in enumerate(normalized_weights)
                )
                new_weights.append(weighted_layer)

            global_model.set_weights(new_weights)

        # Evaluate federated training progress
        _, accuracy = global_model.evaluate(test_data, verbose=0)
        accuracy_history.append(accuracy)

        if args.verbose:
            print(f"    P2 Round {round_num + 1}: Accuracy = {accuracy:.4f}")

        # Early stopping for convergence
        # if len(accuracy_history) > pretraining_rounds + 3:
        #     recent_acc = accuracy_history[-(3):]
        #     if max(recent_acc) - min(recent_acc) < 0.001:
        #         if args.verbose:
        #             print(f"    Early stopping at P2 round {round_num + 1}")
        #         break

    training_time = time.time() - start_time

    return {
        'final_accuracy': accuracy_history[-1] if accuracy_history else 0.0,
        'training_time': training_time,
        'history': {'accuracy': accuracy_history},
        'communication_cost': communication_cost,
        #'convergence_round': len(accuracy_history),
        'pretraining_rounds': pretraining_rounds,
        'federated_rounds': len(accuracy_history) - pretraining_rounds
    }

def train_ensemble_fl(data_splits, args):
    """FL avec méthode ensemble (multiple models)"""
    start_time = time.time()

    client_datasets = data_splits['federated']['clients']
    test_data = data_splits['federated']['test']

    # Train multiple models on different client subsets
    ensemble_models = []
    ensemble_histories = []

    num_ensembles = min(3, len(client_datasets))  # Max 3 ensembles
    clients_per_ensemble = len(client_datasets) // num_ensembles

    for ensemble_id in range(num_ensembles):

        if args.verbose:
            print(f"  Entraînement ensemble {ensemble_id + 1}/{num_ensembles}")

        # Create model for this ensemble
        ensemble_model = create_comparison_model(
            args.dataset, data_splits['input_shape'], data_splits['num_classes']
        )

        ensemble_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Select clients for this ensemble
        start_idx = ensemble_id * clients_per_ensemble
        end_idx = start_idx + clients_per_ensemble if ensemble_id < num_ensembles - 1 else len(client_datasets)
        ensemble_clients = client_datasets[start_idx:end_idx]

        # FL training for this ensemble
        accuracy_history = []

        for round_num in range(args.epochs):
            round_weights = []
            round_sizes = []

            for client_data in ensemble_clients:
                client_model = tf.keras.models.clone_model(ensemble_model)
                client_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                client_model.set_weights(ensemble_model.get_weights())

                # Local training
                client_model.fit(client_data, epochs=1, verbose=0)

                data_size = sum(1 for _ in client_data)
                round_weights.append(client_model.get_weights())
                round_sizes.append(data_size)

            # Aggregate for this ensemble
            if round_weights:
                total_size = sum(round_sizes)
                new_weights = []

                for layer_idx in range(len(round_weights[0])):
                    weighted_layer = sum(
                        (size / total_size) * round_weights[client_idx][layer_idx]
                        for client_idx, size in enumerate(round_sizes)
                    )
                    new_weights.append(weighted_layer)

                ensemble_model.set_weights(new_weights)

            # Evaluate this ensemble
            _, accuracy = ensemble_model.evaluate(test_data, verbose=0)
            accuracy_history.append(accuracy)

        ensemble_models.append(ensemble_model)
        ensemble_histories.append(accuracy_history)

    # Final ensemble prediction (voting)
    test_predictions = []
    for model in ensemble_models:
        predictions = model.predict(test_data, verbose=0)
        test_predictions.append(predictions)

    # Average predictions
    ensemble_predictions = np.mean(test_predictions, axis=0)
    predicted_classes = np.argmax(ensemble_predictions, axis=1)

    # Get true labels
    true_labels = []
    for batch in test_data:
        true_labels.extend(batch[1].numpy())
    true_labels = np.array(true_labels)

    # Calculate ensemble accuracy
    final_accuracy = np.mean(predicted_classes == true_labels)

    training_time = time.time() - start_time
    communication_cost = sum(len(client_datasets[start_idx:end_idx]) for start_idx in
                             range(0, len(client_datasets), clients_per_ensemble)) * args.epochs * 2

    return {
        'final_accuracy': final_accuracy,
        'training_time': training_time,
        'history': {'accuracy': [max(histories) for histories in zip(*ensemble_histories)]},
        'communication_cost': communication_cost,
        'convergence_round': args.epochs,
        'ensemble_count': num_ensembles
    }


def train_federated_with_dropout(data_splits, args, dropout_rate=0.3):
    """FL avec simulation de dropout de clients"""
    start_time = time.time()

    # Même logique que train_federated mais avec dropout aléatoire
    global_model = create_comparison_model(
        args.dataset, data_splits['input_shape'], data_splits['num_classes']
    )

    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    client_datasets = data_splits['federated']['clients']
    test_data = data_splits['federated']['test']

    accuracy_history = []
    communication_cost = 0

    for round_num in range(args.epochs):
        # Sélection aléatoire des clients (simule dropout)
        available_clients = []
        for i, client_data in enumerate(client_datasets):
            if np.random.random() > dropout_rate:
                available_clients.append((i, client_data))

        if not available_clients:
            available_clients = [(0, client_datasets[0])]  # Au moins un client

        round_weights = []
        round_sizes = []

        # Entraîner les clients disponibles
        for client_id, client_data in available_clients:
            client_model = tf.keras.models.clone_model(global_model)
            client_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            client_model.set_weights(global_model.get_weights())

            client_model.fit(client_data, epochs=1, verbose=0)

            round_weights.append(client_model.get_weights())
            # round_sizes.append(sum(1 for _ in client_data))
            round_sizes.append(len(list(client_data.unbatch())))

        # Agrégation
        aggregated_weights = fedavg_aggregate(round_weights, round_sizes)
        global_model.set_weights(aggregated_weights)

        # Évaluer
        _, accuracy = global_model.evaluate(test_data, verbose=0)
        accuracy_history.append(accuracy)

        # Communication (seulement clients actifs)
        num_params = sum(np.prod(w.shape) for w in global_model.get_weights())
        communication_cost += num_params * len(available_clients) * 2

    training_time = time.time() - start_time

    return {
        'final_accuracy': accuracy_history[-1] if accuracy_history else 0.0,
        'training_time': training_time,
        'history': {'accuracy': accuracy_history},
        'communication_cost': communication_cost,
        'convergence_round': len(accuracy_history)
    }


def fedavg_aggregate(client_weights, client_sizes):
    """Agrégation FedAvg simple"""
    total_size = sum(client_sizes)

    # Calculer les poids normalisés
    weights = [size / total_size for size in client_sizes]

    # Agrégation pondérée
    aggregated = []
    for layer_idx in range(len(client_weights[0])):
        layer_sum = None
        for client_idx, client_w in enumerate(client_weights):
            layer = client_w[layer_idx] * weights[client_idx]
            if layer_sum is None:
                layer_sum = layer
            else:
                layer_sum += layer
        aggregated.append(layer_sum)

    return aggregated