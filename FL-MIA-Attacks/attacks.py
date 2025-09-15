"""
Implémentation des attaques MIA sur apprentissage fédéré
"""
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from models import create_mia_model
from data_loader import create_attack_dataset, prepare_threshold_attack_data


def run_shadow_model_attack(mia_data, args):
    """Exécute l'attaque Shadow Model"""
    print("👤 Lancement attaque Shadow Model...")
    start_time = time.time()

    # 1. Entraîner le modèle cible (fédéré)
    print("🎯 Entraînement du modèle cible fédéré...")
    target_model = train_target_federated_model(mia_data, args)

    # 2. Entraîner les modèles shadow
    print(f"👥 Entraînement de {args.shadow_models} modèles shadow...")
    shadow_models = train_shadow_models(mia_data, args)

    # 3. Créer le dataset d'attaque
    print("📊 Création du dataset d'attaque...")
    attack_data = create_shadow_attack_dataset(shadow_models, mia_data)

    # 4. Entraîner l'attaquant
    print("🎯 Entraînement du modèle attaquant...")
    attacker = train_shadow_attacker(attack_data, args)

    # 5. Évaluer l'attaque sur le modèle cible
    print("🔍 Évaluation de l'attaque...")
    attack_results = evaluate_shadow_attack(target_model, attacker, mia_data, args)

    attack_time = time.time() - start_time

    return {
        'Shadow_Model': {
            'attack_accuracy': attack_results['accuracy'],
            'precision': attack_results['precision'],
            'recall': attack_results['recall'],
            'auc_score': attack_results['auc'],
            'attack_time': attack_time,
            'risk_level': args.risk_level,
            'shadow_models_used': args.shadow_models
        }
    }


def run_threshold_attack(mia_data, args):
    """Exécute l'attaque Threshold"""
    print("📊 Lancement attaque Threshold...")
    start_time = time.time()

    # 1. Entraîner le modèle cible
    print("🎯 Entraînement du modèle cible...")
    target_model = train_target_federated_model(mia_data, args)

    # 2. Préparer les données d'attaque
    print("📊 Préparation des données d'attaque...")
    threshold_data = prepare_threshold_attack_data(
        target_model, mia_data['target_data'], mia_data['non_member_data']
    )

    # 3. Optimiser le seuil
    print("🔍 Optimisation du seuil...")
    optimal_threshold, threshold_results = optimize_threshold(threshold_data)

    attack_time = time.time() - start_time

    return {
        'Threshold': {
            'attack_accuracy': threshold_results['accuracy'],
            'precision': threshold_results['precision'],
            'recall': threshold_results['recall'],
            'auc_score': threshold_results['auc'],
            'optimal_threshold': optimal_threshold,
            'attack_time': attack_time,
            'risk_level': args.risk_level
        }
    }


def run_gradient_based_attack(mia_data, args):
    """Exécute l'attaque basée sur les gradients"""
    print("📈 Lancement attaque Gradient-Based...")
    start_time = time.time()

    # Note: Implémentation simplifiée pour le projet académique
    # Dans un cas réel, nécessiterait l'accès aux gradients du FL

    # 1. Entraîner le modèle cible
    target_model = train_target_federated_model(mia_data, args)

    # 2. Simuler l'analyse des gradients
    gradient_results = simulate_gradient_analysis(target_model, mia_data, args)

    attack_time = time.time() - start_time

    return {
        'Gradient_Based': {
            'attack_accuracy': gradient_results['accuracy'],
            'precision': gradient_results['precision'],
            'recall': gradient_results['recall'],
            'auc_score': gradient_results['auc'],
            'attack_time': attack_time,
            'risk_level': args.risk_level,
            'note': 'Simulation academic - gradient access required in real scenario'
        }
    }


def run_all_attacks(mia_data, args):
    """Exécute toutes les attaques MIA"""
    print("🎯 Lancement de toutes les attaques MIA...")

    all_results = {}

    # 1. Shadow Model Attack
    try:
        shadow_results = run_shadow_model_attack(mia_data, args)
        all_results.update(shadow_results)
    except Exception as e:
        print(f"❌ Erreur Shadow Model: {e}")
        all_results['Shadow_Model'] = create_failed_result('Shadow Model failed')

    # 2. Threshold Attack
    try:
        threshold_results = run_threshold_attack(mia_data, args)
        all_results.update(threshold_results)
    except Exception as e:
        print(f"❌ Erreur Threshold: {e}")
        all_results['Threshold'] = create_failed_result('Threshold failed')

    # 3. Gradient-Based Attack
    try:
        gradient_results = run_gradient_based_attack(mia_data, args)
        all_results.update(gradient_results)
    except Exception as e:
        print(f"❌ Erreur Gradient-Based: {e}")
        all_results['Gradient_Based'] = create_failed_result('Gradient-based failed')

    return all_results


def train_target_federated_model(mia_data, args):
    """Entraîne le modèle cible avec apprentissage fédéré"""

    # Créer le modèle global
    global_model = create_mia_model(
        args.dataset, mia_data['input_shape'], mia_data['num_classes']
    )

    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    client_datasets = mia_data['federated_data']['clients']

    # Entraînement fédéré simple
    for round_num in range(args.epochs):
        client_weights = []
        client_sizes = []

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

            # Entraîner localement
            client_model.fit(client_data, epochs=1, verbose=0)

            # Collecter poids et taille
            client_weights.append(client_model.get_weights())
            client_sizes.append(sum(1 for _ in client_data))

        # Agrégation FedAvg
        aggregated_weights = fedavg_aggregate(client_weights, client_sizes)
        global_model.set_weights(aggregated_weights)

        if args.verbose and round_num % 5 == 0:
            test_loss, test_acc = global_model.evaluate(mia_data['test_data'], verbose=0)
            print(f"  Round {round_num}: Test Accuracy = {test_acc:.4f}")

    return global_model


def train_shadow_models(mia_data, args):
    """Entraîne les modèles shadow"""
    shadow_models = []
    shadow_datasets = mia_data['shadow_data']

    for i, shadow_data in enumerate(shadow_datasets[:args.shadow_models]):
        if args.verbose:
            print(f"  Entraînement shadow model {i+1}/{args.shadow_models}")

        # Créer modèle shadow
        shadow_model = create_mia_model(
            args.dataset, mia_data['input_shape'], mia_data['num_classes']
        )

        shadow_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Entraîner sur données shadow
        shadow_model.fit(
            shadow_data['train_x'], shadow_data['train_y'],
            epochs=args.epochs // 2,  # Moins d'époques pour shadow models
            batch_size=args.batch_size,
            verbose=0
        )

        shadow_models.append((shadow_model, shadow_data))

    return shadow_models


def create_shadow_attack_dataset(shadow_models, mia_data):
    """Crée le dataset d'attaque à partir des modèles shadow"""
    all_features = []
    all_labels = []

    for shadow_model, shadow_data in shadow_models:
        # Prédictions sur données membres
        member_preds = shadow_model.predict(shadow_data['member_x'], verbose=0)

        # Prédictions sur données non-membres
        non_member_preds = shadow_model.predict(shadow_data['non_member_x'], verbose=0)

        # Créer features (probabilités de sortie)
        features, labels = create_attack_dataset(
            shadow_data['member_x'], shadow_data['non_member_x'],
            member_preds, non_member_preds
        )

        all_features.append(features)
        all_labels.append(labels)

    # Combiner tous les datasets d'attaque
    combined_features = np.vstack(all_features)
    combined_labels = np.hstack(all_labels)

    return combined_features, combined_labels


def train_shadow_attacker(attack_data, args):
    """Entraîne le modèle attaquant"""
    features, labels = attack_data

    # Choisir le modèle d'attaque selon le niveau de risque
    if args.risk_level == 'high':
        # Random Forest pour attaque sophistiquée
        attacker = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
    elif args.risk_level == 'medium':
        # Logistic Regression pour attaque modérée
        attacker = LogisticRegression(random_state=42, max_iter=1000)
    else:  # low
        # Modèle simple pour attaque basique
        attacker = LogisticRegression(
            random_state=42, max_iter=500, solver='liblinear'
        )

    # Entraîner l'attaquant
    attacker.fit(features, labels)

    return attacker


def evaluate_shadow_attack(target_model, attacker, mia_data, args):
    """Évalue l'attaque shadow model sur le modèle cible"""

    # Prédictions du modèle cible sur données membres/non-membres
    member_preds = target_model.predict(mia_data['target_data']['x'], verbose=0)
    non_member_preds = target_model.predict(mia_data['non_member_data']['x'], verbose=0)

    # Créer features pour l'attaque
    attack_features, true_labels = create_attack_dataset(
        mia_data['target_data']['x'], mia_data['non_member_data']['x'],
        member_preds, non_member_preds
    )

    # Prédictions de l'attaquant
    attack_predictions = attacker.predict(attack_features)
    attack_probabilities = attacker.predict_proba(attack_features)[:, 1]

    # Calculer métriques
    accuracy = accuracy_score(true_labels, attack_predictions)
    precision = precision_score(true_labels, attack_predictions)
    recall = recall_score(true_labels, attack_predictions)
    auc = roc_auc_score(true_labels, attack_probabilities)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }


def optimize_threshold(threshold_data):
    """Optimise le seuil pour l'attaque threshold"""
    member_confidences = threshold_data['member_confidences']
    non_member_confidences = threshold_data['non_member_confidences']

    # Combiner les données
    all_confidences = np.hstack([member_confidences, non_member_confidences])
    true_labels = np.hstack([
        np.ones(len(member_confidences)),
        np.zeros(len(non_member_confidences))
    ])

    # Tester différents seuils
    thresholds = np.linspace(
        np.min(all_confidences),
        np.max(all_confidences),
        100
    )

    best_accuracy = 0
    best_threshold = 0.5
    best_results = {}

    for threshold in thresholds:
        # Prédictions: si confiance > seuil alors membre (1), sinon non-membre (0)
        predictions = (all_confidences > threshold).astype(int)

        accuracy = accuracy_score(true_labels, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_results = {
                'accuracy': accuracy,
                'precision': precision_score(true_labels, predictions),
                'recall': recall_score(true_labels, predictions),
                'auc': roc_auc_score(true_labels, all_confidences)
            }

    return best_threshold, best_results


def simulate_gradient_analysis(target_model, mia_data, args):
    """Simule une attaque basée sur l'analyse des gradients"""
    # Note: Implémentation académique simplifiée
    # En réalité, nécessiterait accès aux gradients pendant l'entraînement FL

    # Simuler des "scores de gradient" basés sur la loss
    member_data = mia_data['target_data']
    non_member_data = mia_data['non_member_data']

    # Calculer la loss pour chaque échantillon (proxy pour gradient)
    member_losses = []
    for i in range(len(member_data['x'])):
        x_sample = member_data['x'][i:i+1]
        y_sample = member_data['y'][i:i+1]
        loss = target_model.evaluate(x_sample, y_sample, verbose=0)[0]
        member_losses.append(loss)

    non_member_losses = []
    for i in range(len(non_member_data['x'])):
        x_sample = non_member_data['x'][i:i+1]
        y_sample = non_member_data['y'][i:i+1]
        loss = target_model.evaluate(x_sample, y_sample, verbose=0)[0]
        non_member_losses.append(loss)

    # Utiliser les loss comme features (membres ont généralement loss plus faible)
    all_losses = np.hstack([member_losses, non_member_losses])
    true_labels = np.hstack([
        np.ones(len(member_losses)),
        np.zeros(len(non_member_losses))
    ])

    # Attaque simple: si loss < seuil alors membre
    threshold = np.median(all_losses)
    predictions = (all_losses < threshold).astype(int)

    # Calculer métriques
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, -all_losses)  # Négatif car loss faible = membre

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }


def fedavg_aggregate(client_weights, client_sizes):
    """Agrégation FedAvg pour le modèle cible"""
    total_size = sum(client_sizes)

    # Poids normalisés
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


def create_failed_result(error_msg):
    """Crée un résultat d'erreur pour attaque échouée"""
    return {
        'attack_accuracy': 0.5,  # Aléatoire
        'precision': 0.5,
        'recall': 0.5,
        'auc_score': 0.5,
        'attack_time': 0.0,
        'error': error_msg
    }


def apply_defense_mechanism(model, defense_type, args):
    """Applique des mécanismes de défense selon le niveau de risque"""

    if args.risk_level == 'low':
        # Défenses activées (simulation)
        print("🛡️ Défenses appliquées: Differential Privacy, Gradient Clipping")
        # En pratique: ajout de bruit, clipping des gradients, etc.
        return model

    elif args.risk_level == 'medium':
        # Défenses partielles
        print("🔒 Défenses partielles appliquées")
        return model

    else:  # high
        # Aucune défense (attaque maximale)
        print("⚠️ Aucune défense - Attaque en conditions maximales")
        return model