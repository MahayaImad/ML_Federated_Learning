"""
FL-Adaptive_Clustering: Apprentissage Fédéré avec Clustering Adaptatif
Méthode: Combinaison Direction (Cosinus) + Magnitude (Euclidienne)
Framework: TensorFlow/Keras
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import defaultdict
import copy

# Désactiver les avertissements TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class Config:
    # Paramètres du dataset
    num_clients = 20
    samples_per_client = 2500
    num_classes = 10

    # Paramètres d'entraînement
    local_epochs = 5
    batch_size = 50
    learning_rate = 0.01
    global_rounds = 50

    # Paramètres de clustering
    initial_clusters = 4
    min_cluster_size = 3
    cosine_weight = 0.6  # Poids pour la direction (cosinus)
    magnitude_weight = 0.4  # Poids pour la magnitude (euclidienne)

    # Paramètres FedProx
    mu_prox = 0.01  # Coefficient de régularisation proximale

    # Paramètres de robustesse
    gradient_clip_threshold = 5.0  # Clipping de magnitude
    min_cosine_similarity = 0.3  # Filtrage de direction

    # Non-IID
    alpha_dirichlet = 0.5  # Plus petit = plus non-IID

    # Seed pour reproductibilité
    seed = 42

# ============================================================================
# 2. MODÈLE CNN avec TensorFlow/Keras
# ============================================================================

def create_cnn_model(input_shape=(28,28,1), num_classes=10):
    """
    Crée un modèle CNN simple pour CIFAR-10
    """
    model = models.Sequential([
        # Première couche de convolution
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Deuxième couche de convolution
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Couches fully connected
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

# ============================================================================
# 3. DISTRIBUTION NON-IID AVEC DIRICHLET
# ============================================================================

def create_noniid_dirichlet(y_train, num_clients, alpha, num_classes):
    """
    Crée une distribution non-IID via Dirichlet
    alpha petit (0.1-0.5) = très non-IID
    alpha grand (10+) = presque IID
    """
    labels = y_train
    client_indices = [[] for _ in range(num_clients)]

    # Pour chaque classe, distribuer selon Dirichlet
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        # Proportions Dirichlet pour cette classe
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        # Distribuer les indices
        splits = np.split(idx_k, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split)

    # Mélanger les indices de chaque client
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])

    return client_indices

# ============================================================================
# 4. MÉTRIQUES HYBRIDES: DIRECTION + MAGNITUDE
# ============================================================================

class HybridMetrics:
    """
    Combine la direction (similarité cosinus) et la magnitude (distance euclidienne)
    """

    @staticmethod
    def cosine_similarity(grad1, grad2):
        """Similarité cosinus entre deux vecteurs"""
        dot_product = np.dot(grad1, grad2)
        norm1 = np.linalg.norm(grad1)
        norm2 = np.linalg.norm(grad2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def euclidean_distance_normalized(grad1, grad2):
        """
        Distance euclidienne sur gradients normalisés
        Mesure la différence de direction pure
        """
        norm1 = np.linalg.norm(grad1)
        norm2 = np.linalg.norm(grad2)

        if norm1 == 0 or norm2 == 0:
            return 1.0

        grad1_norm = grad1 / norm1
        grad2_norm = grad2 / norm2

        return np.linalg.norm(grad1_norm - grad2_norm)

    @staticmethod
    def euclidean_distance(grad1, grad2):
        """Distance euclidienne classique (magnitude)"""
        return np.linalg.norm(grad1 - grad2)

    @staticmethod
    def hybrid_distance(grad1, grad2, alpha=0.6, beta=0.4, normalize=True):
        """
        Distance hybride combinant direction et magnitude

        Args:
            grad1, grad2: Vecteurs de gradients
            alpha: Poids pour la composante direction (cosinus)
            beta: Poids pour la composante magnitude (euclidienne)
            normalize: Si True, normalise les distances dans [0,1]

        Returns:
            Distance combinée (plus petit = plus similaire)
        """
        # Composante direction (1 - cosinus pour avoir une distance)
        cos_sim = HybridMetrics.cosine_similarity(grad1, grad2)
        direction_dist = 1 - cos_sim  # [0, 2], 0 = même direction

        # Composante magnitude
        if normalize:
            magnitude_dist = HybridMetrics.euclidean_distance_normalized(grad1, grad2)
        else:
            magnitude_dist = HybridMetrics.euclidean_distance(grad1, grad2)
            max_norm = np.linalg.norm(grad1) + np.linalg.norm(grad2)
            if max_norm > 0:
                magnitude_dist = magnitude_dist / max_norm

        # Combinaison pondérée
        hybrid_dist = alpha * direction_dist + beta * magnitude_dist

        return hybrid_dist

    @staticmethod
    def compute_pairwise_distances(gradients, alpha=0.6, beta=0.4):
        """
        Calcule la matrice de distances hybrides entre tous les clients
        """
        n = len(gradients)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = HybridMetrics.hybrid_distance(
                    gradients[i], gradients[j], alpha, beta
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        return distance_matrix

# ============================================================================
# 5. AGRÉGATION ROBUSTE AVEC FILTRAGE
# ============================================================================

class RobustAggregation:
    """
    Agrégation robuste avec filtrage de direction et clipping de magnitude
    """

    @staticmethod
    def clip_gradient_norm(gradient, max_norm):
        """Clippe la norme du gradient à max_norm"""
        norm = np.linalg.norm(gradient)
        if norm > max_norm:
            return gradient * (max_norm / norm)
        return gradient

    @staticmethod
    def filter_by_direction(gradients, min_cosine_similarity=0.3):
        """
        Filtre les gradients dont la direction est trop différente de la médiane
        """
        if len(gradients) < 3:
            return list(range(len(gradients)))

        # Calculer le gradient médian (approximation robuste)
        median_grad = np.median(np.array(gradients), axis=0)

        valid_indices = []
        for i, grad in enumerate(gradients):
            cos_sim = HybridMetrics.cosine_similarity(grad, median_grad)
            if cos_sim >= min_cosine_similarity:
                valid_indices.append(i)

        # Si trop de gradients filtrés, garder au moins 50%
        if len(valid_indices) < len(gradients) // 2:
            similarities = [
                HybridMetrics.cosine_similarity(grad, median_grad)
                for grad in gradients
            ]
            top_k = max(len(gradients) // 2, 1)
            valid_indices = np.argsort(similarities)[-top_k:].tolist()

        return valid_indices

    @staticmethod
    def weighted_aggregation(updates, weights, clip_norm=None, filter_direction=False):
        """
        Agrégation pondérée avec options de robustesse
        """
        # Filtrage de direction si demandé
        valid_indices = list(range(len(updates)))
        if filter_direction:
            valid_indices = RobustAggregation.filter_by_direction(updates)
            updates = [updates[i] for i in valid_indices]
            weights = [weights[i] for i in valid_indices]

        # Clipping de magnitude si demandé
        if clip_norm is not None:
            updates = [
                RobustAggregation.clip_gradient_norm(u, clip_norm)
                for u in updates
            ]

        # Normalisation des poids
        total_weight = sum(weights)
        if total_weight == 0:
            return np.zeros_like(updates[0])

        normalized_weights = [w / total_weight for w in weights]

        # Agrégation pondérée
        aggregated = np.zeros_like(updates[0])
        for update, weight in zip(updates, normalized_weights):
            aggregated += weight * update

        return aggregated

# ============================================================================
# 6. UTILITAIRES POUR MODÈLES TENSORFLOW
# ============================================================================

class ModelUtils:
    """Utilitaires pour manipuler les modèles TensorFlow"""

    @staticmethod
    def get_weights_as_vector(model):
        """Aplatit tous les poids du modèle en un vecteur numpy"""
        weights = model.get_weights()
        return np.concatenate([w.flatten() for w in weights])

    @staticmethod
    def set_weights_from_vector(model, vector):
        """Restaure les poids du modèle depuis un vecteur"""
        weights = model.get_weights()
        start = 0
        new_weights = []

        for w in weights:
            size = np.prod(w.shape)
            new_w = vector[start:start+size].reshape(w.shape)
            new_weights.append(new_w)
            start += size

        model.set_weights(new_weights)

    @staticmethod
    def copy_model(model):
        """Crée une copie du modèle"""
        model_copy = keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        return model_copy

    @staticmethod
    def average_models(models, weights):
        """Moyenne pondérée de plusieurs modèles"""
        if not models:
            return None

        # Normaliser les poids
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Créer un nouveau modèle
        averaged_model = ModelUtils.copy_model(models[0])

        # Obtenir les poids de tous les modèles
        all_weights = [model.get_weights() for model in models]

        # Calculer la moyenne pondérée pour chaque couche
        averaged_weights = []
        for layer_idx in range(len(all_weights[0])):
            layer_weights = [
                w[layer_idx] * normalized_weights[i]
                for i, w in enumerate(all_weights)
            ]
            averaged_weights.append(np.sum(layer_weights, axis=0))

        averaged_model.set_weights(averaged_weights)

        # Compiler le modèle moyenné
        averaged_model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.01),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        return averaged_model

# ============================================================================
# 7. CLUSTERING ADAPTATIF AVEC MÉTRIQUE HYBRIDE
# ============================================================================

class AdaptiveClustering:
    """
    Clustering adaptatif utilisant la métrique hybride direction-magnitude
    """

    def __init__(self, config):
        self.config = config
        self.cluster_assignments = None
        self.cluster_models = {}

    def compute_gradient_updates(self, client_models, global_model):
        """Calcule les mises à jour (Delta w) pour chaque client"""
        gradients = []

        global_params = ModelUtils.get_weights_as_vector(global_model)

        for client_model in client_models:
            client_params = ModelUtils.get_weights_as_vector(client_model)
            delta_w = client_params - global_params
            gradients.append(delta_w)

        return gradients

    def perform_clustering(self, gradients, data_sizes):
        """
        Effectue le clustering avec la métrique hybride
        """
        # Calculer la matrice de distances hybrides
        distance_matrix = HybridMetrics.compute_pairwise_distances(
            gradients,
            alpha=self.config.cosine_weight,
            beta=self.config.magnitude_weight
        )

        # Clustering hiérarchique
        n_clusters = min(self.config.initial_clusters, len(gradients) // 2)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )

        labels = clustering.fit_predict(distance_matrix)

        # Vérifier la taille minimale des clusters
        labels = self._enforce_min_cluster_size(labels, data_sizes)

        self.cluster_assignments = labels
        return labels

    def _enforce_min_cluster_size(self, labels, data_sizes):
        """Assure que chaque cluster a au moins min_cluster_size clients"""
        unique_labels = np.unique(labels)

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < self.config.min_cluster_size:
                if len(unique_labels) > 1:
                    other_labels = [l for l in unique_labels if l != label]
                    best_label = max(
                        other_labels,
                        key=lambda l: len(np.where(labels == l)[0])
                    )
                    labels[cluster_indices] = best_label

        # Réindexer les labels
        unique_labels = np.unique(labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_mapping[l] for l in labels])

        return labels

    def aggregate_clusters(self, client_models, data_sizes, use_robust=True):
        """
        Agrège les modèles au sein de chaque cluster
        """
        if self.cluster_assignments is None:
            raise ValueError("Clustering non effectué")

        cluster_models = {}

        for cluster_id in np.unique(self.cluster_assignments):
            # Indices des clients dans ce cluster
            cluster_indices = np.where(self.cluster_assignments == cluster_id)[0]

            # Modèles et poids du cluster
            cluster_client_models = [client_models[i] for i in cluster_indices]
            cluster_weights = [data_sizes[i] for i in cluster_indices]

            # Agrégation robuste si demandé
            if use_robust and len(cluster_client_models) > 1:
                # Extraire les poids comme vecteurs
                weights_vectors = [
                    ModelUtils.get_weights_as_vector(model)
                    for model in cluster_client_models
                ]

                # Agrégation robuste
                aggregated_vector = RobustAggregation.weighted_aggregation(
                    weights_vectors,
                    cluster_weights,
                    clip_norm=self.config.gradient_clip_threshold,
                    filter_direction=True
                )

                # Créer le modèle agrégé
                aggregated_model = ModelUtils.copy_model(cluster_client_models[0])
                ModelUtils.set_weights_from_vector(aggregated_model, aggregated_vector)

                # Compiler le modèle agrégé
                aggregated_model.compile(
                    optimizer=keras.optimizers.SGD(learning_rate=self.config.learning_rate),
                    loss=keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy']
                )
            else:
                # Agrégation simple
                aggregated_model = ModelUtils.average_models(
                    cluster_client_models,
                    cluster_weights
                )

            cluster_models[cluster_id] = aggregated_model

        return cluster_models

# ============================================================================
# 8. ENTRAÎNEMENT LOCAL AVEC FEDPROX
# ============================================================================

class ProximalLoss(keras.losses.Loss):
    """Loss personnalisée avec terme proximal FedProx"""

    def __init__(self, global_weights, mu, base_loss=keras.losses.SparseCategoricalCrossentropy()):
        super().__init__()
        self.global_weights = global_weights
        self.mu = mu
        self.base_loss = base_loss

    def call(self, y_true, y_pred):
        # Loss de base (cross-entropy)
        base = self.base_loss(y_true, y_pred)

        # Terme proximal: mu/2 * ||w - w_global||^2
        # Note: sera ajouté lors de l'entraînement via une métrique personnalisée
        return base

def train_local_fedprox(model, x_train, y_train, config, global_model=None):
    """
    Entraînement local avec régularisation FedProx
    """
    # Compiler le modèle
    optimizer = keras.optimizers.SGD(
        learning_rate=config.learning_rate,
        momentum=0.9
    )

    if global_model is not None and config.mu_prox > 0:
        # Entraînement avec FedProx
        loss_fn = keras.losses.SparseCategoricalCrossentropy()
        global_weights = global_model.get_weights()

        # Entraînement personnalisé avec terme proximal
        for epoch in range(config.local_epochs):
            # Mélanger les données
            indices = np.random.permutation(len(x_train))
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batches
            num_batches = len(x_train) // config.batch_size
            for batch_idx in range(num_batches):
                start = batch_idx * config.batch_size
                end = start + config.batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = model(x_batch, training=True)
                    loss = loss_fn(y_batch, predictions)

                    # Ajouter le terme proximal FedProx
                    current_weights = model.get_weights()
                    proximal_term = 0.0
                    for cw, gw in zip(current_weights, global_weights):
                        proximal_term += tf.reduce_sum(tf.square(cw - gw))

                    loss += (config.mu_prox / 2) * proximal_term

                # Backward pass
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    else:
        # Entraînement standard sans FedProx
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        model.fit(
            x_train, y_train,
            batch_size=config.batch_size,
            epochs=config.local_epochs,
            verbose=0
        )

    return model

# ============================================================================
# 9. ANALYSE DE QUALITÉ DU CLUSTERING
# ============================================================================

def analyze_clustering_quality(gradients, cluster_labels, config):
    """
    Analyse la qualité du clustering avec métriques hybrides
    """
    # Matrice de distances hybrides
    distance_matrix = HybridMetrics.compute_pairwise_distances(
        gradients,
        alpha=config.cosine_weight,
        beta=config.magnitude_weight
    )

    # Score de silhouette
    if len(np.unique(cluster_labels)) > 1:
        silhouette = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
    else:
        silhouette = 0.0

    # Analyse par composante
    direction_scores = []
    magnitude_scores = []

    for cluster_id in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) < 2:
            continue

        cluster_gradients = [gradients[i] for i in cluster_indices]

        # Direction: similarité cosinus moyenne
        cos_sims = []
        for i in range(len(cluster_gradients)):
            for j in range(i + 1, len(cluster_gradients)):
                cos_sim = HybridMetrics.cosine_similarity(
                    cluster_gradients[i],
                    cluster_gradients[j]
                )
                cos_sims.append(cos_sim)

        if cos_sims:
            direction_scores.append(np.mean(cos_sims))

        # Magnitude: variance des normes
        norms = [np.linalg.norm(g) for g in cluster_gradients]
        if len(norms) > 1:
            magnitude_scores.append(np.std(norms) / (np.mean(norms) + 1e-8))

    return {
        'silhouette': silhouette,
        'avg_direction_cohesion': np.mean(direction_scores) if direction_scores else 0.0,
        'avg_magnitude_variance': np.mean(magnitude_scores) if magnitude_scores else 0.0
    }

# ============================================================================
# 10. FONCTION PRINCIPALE D'ENTRAÎNEMENT
# ============================================================================

def federated_learning_adaptive_clustering(config):
    """
    Apprentissage fédéré avec clustering adaptatif hybride - Version TensorFlow
    """
    print("=== FL-Adaptive_Clustering: Direction + Magnitude (TensorFlow) ===")
    print(f"Clients: {config.num_clients}")
    print(f"Poids Direction (cosinus): {config.cosine_weight}")
    print(f"Poids Magnitude (euclidienne): {config.magnitude_weight}")
    print(f"Régularisation FedProx: μ={config.mu_prox}")
    print(f"Gradient Clipping: {config.gradient_clip_threshold}")
    print(f"Dirichlet α: {config.alpha_dirichlet}")
    print()

    # Charger CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normaliser les données
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Aplatir les labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    print(f"Dataset chargé: {x_train.shape[0]} images d'entraînement")

    # Distribution non-IID
    client_indices = create_noniid_dirichlet(
        y_train,
        config.num_clients,
        config.alpha_dirichlet,
        config.num_classes
    )

    data_sizes = [len(indices) for indices in client_indices]
    print(f"Distribution créée: {config.num_clients} clients avec distribution non-IID (α={config.alpha_dirichlet})")

    # Modèle global initial
    global_model = create_cnn_model(num_classes=config.num_classes)
    global_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Clustering adaptatif
    adaptive_clustering = AdaptiveClustering(config)

    # Métriques
    history = {
        'test_acc': [],
        'test_loss': [],
        'cluster_count': [],
        'cluster_sizes': []
    }

    clustering_quality_history = []

    # Boucle d'entraînement global
    for round_num in range(config.global_rounds):
        print(f"\n--- Round {round_num + 1}/{config.global_rounds} ---")

        # Entraînement local
        client_models = []
        for client_id in range(config.num_clients):
            # Copier le modèle global
            local_model = ModelUtils.copy_model(global_model)

            # Obtenir les données du client
            client_idx = client_indices[client_id]
            x_client = x_train[client_idx]
            y_client = y_train[client_idx]

            # Entraînement avec FedProx
            local_model = train_local_fedprox(
                local_model,
                x_client,
                y_client,
                config,
                global_model=global_model
            )

            client_models.append(local_model)

        # Calculer les mises à jour (Delta w)
        gradients = adaptive_clustering.compute_gradient_updates(
            client_models,
            global_model
        )

        # Clustering avec métrique hybride
        cluster_labels = adaptive_clustering.perform_clustering(
            gradients,
            data_sizes
        )

        # Analyser la qualité du clustering
        quality_metrics = analyze_clustering_quality(gradients, cluster_labels, config)
        clustering_quality_history.append(quality_metrics)

        n_clusters = len(np.unique(cluster_labels))
        print(f"Clusters: {n_clusters} | Silhouette: {quality_metrics['silhouette']:.3f} | "
              f"Direction: {quality_metrics['avg_direction_cohesion']:.3f} | "
              f"Magnitude Var: {quality_metrics['avg_magnitude_variance']:.3f}")

        for cluster_id in range(n_clusters):
            cluster_size = np.sum(cluster_labels == cluster_id)
            print(f"  Cluster {cluster_id}: {cluster_size} clients")

        # Agrégation robuste par cluster
        cluster_models = adaptive_clustering.aggregate_clusters(
            client_models,
            data_sizes,
            use_robust=True
        )

        # Agrégation globale (moyenne des modèles de clusters)
        cluster_sizes = [
            sum([data_sizes[i] for i in range(config.num_clients)
                 if cluster_labels[i] == cid])
            for cid in cluster_models.keys()
        ]

        global_model = ModelUtils.average_models(
            list(cluster_models.values()),
            cluster_sizes
        )

        # Recompiler le modèle après agrégation
        global_model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=config.learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # Évaluation
        test_results = global_model.evaluate(x_test, y_test, verbose=0)
        test_loss, test_acc = test_results[0], test_results[1] * 100
        print(f"Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")

        # Enregistrer les métriques
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['cluster_count'].append(n_clusters)
        history['cluster_sizes'].append(cluster_sizes)

    return global_model, history, clustering_quality_history

# ============================================================================
# 11. ÉVALUATION
# ============================================================================

def evaluate_model(model, x_test, y_test):
    """Évalue la précision du modèle sur le test set"""
    results = model.evaluate(x_test, y_test, verbose=0)
    loss, accuracy = results[0], results[1] * 100
    return accuracy, loss

# ============================================================================
# 12. VISUALISATION
# ============================================================================

def plot_detailed_results(history, clustering_quality_history):
    """Visualisation détaillée avec analyse de qualité"""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Précision de test
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(history['test_acc'], marker='o', linewidth=2.5, color='#2E86AB', label='Test Accuracy')
    ax1.fill_between(range(len(history['test_acc'])), history['test_acc'], alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Evolution de la Précision sur CIFAR-10 (TensorFlow)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)

    # 2. Nombre de clusters
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(history['cluster_count'], marker='s', linewidth=2.5, color='#A23B72', label='Clusters')
    ax2.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Nombre', fontsize=11, fontweight='bold')
    ax2.set_title('Nombre de Clusters', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)

    # 3. Score de silhouette
    ax3 = fig.add_subplot(gs[1, 0])
    silhouette_scores = [q['silhouette'] for q in clustering_quality_history]
    ax3.plot(silhouette_scores, marker='d', linewidth=2, color='#F18F01', label='Silhouette')
    ax3.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax3.set_title('Qualité du Clustering\n(Silhouette)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.legend(fontsize=10)

    # 4. Cohésion de direction
    ax4 = fig.add_subplot(gs[1, 1])
    direction_scores = [q['avg_direction_cohesion'] for q in clustering_quality_history]
    ax4.plot(direction_scores, marker='o', linewidth=2, color='#06A77D', label='Direction')
    ax4.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cosinus Moyen', fontsize=11, fontweight='bold')
    ax4.set_title('Cohésion de Direction\n(Intra-cluster)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=10)

    # 5. Variance de magnitude
    ax5 = fig.add_subplot(gs[1, 2])
    magnitude_scores = [q['avg_magnitude_variance'] for q in clustering_quality_history]
    ax5.plot(magnitude_scores, marker='^', linewidth=2, color='#D62246', label='Magnitude')
    ax5.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Variance Relative', fontsize=11, fontweight='bold')
    ax5.set_title('Variance de Magnitude\n(Intra-cluster)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.legend(fontsize=10)

    # 6. Distribution des tailles de clusters (dernier round)
    ax6 = fig.add_subplot(gs[2, :])
    last_sizes = history['cluster_sizes'][-1]
    clusters_ids = range(len(last_sizes))
    colors_palette = plt.cm.Set3(np.linspace(0, 1, len(last_sizes)))

    bars = ax6.bar(clusters_ids, last_sizes, color=colors_palette, edgecolor='black', linewidth=1.5)
    ax6.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Nombre de Clients', fontsize=12, fontweight='bold')
    ax6.set_title('Distribution des Clients par Cluster (Dernier Round)', fontsize=14, fontweight='bold')
    ax6.set_xticks(clusters_ids)
    ax6.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Ajouter les valeurs sur les barres
    for bar, size in zip(bars, last_sizes):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(size)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('FL-Adaptive_Clustering: Analyse Complète (TensorFlow)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('fl_adaptive_clustering_tensorflow.png', dpi=300, bbox_inches='tight')
    print("\nGraphiques sauvegardés: fl_adaptive_clustering_tensorflow.png")
    plt.show()

def plot_convergence_comparison(history):
    """Compare la convergence: précision vs loss"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Précision
    axes[0].plot(history['test_acc'], marker='o', linewidth=2, color='#2E86AB')
    axes[0].set_xlabel('Round', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Convergence de la Précision', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history['test_loss'], marker='s', linewidth=2, color='#D62246')
    axes[1].set_xlabel('Round', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Convergence de la Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fl_convergence_tensorflow.png', dpi=300, bbox_inches='tight')
    print("Graphiques de convergence sauvegardés: fl_convergence_tensorflow.png")
    plt.show()

# ============================================================================
# 13. ÉTUDE D'ABLATION
# ============================================================================

def ablation_study(config):
    """
    Étude d'ablation: comparer différentes configurations de poids
    """
    print("\n" + "="*70)
    print("ÉTUDE D'ABLATION: Impact des Poids Direction vs Magnitude")
    print("="*70)

    weight_configs = [
        (1.0, 0.0, "Direction seule (Cosinus)"),
        (0.8, 0.2, "Direction dominante"),
        (0.6, 0.4, "Équilibrée (défaut)"),
        (0.4, 0.6, "Magnitude dominante"),
        (0.0, 1.0, "Magnitude seule (Euclidienne)"),
    ]

    results_summary = []

    for alpha, beta, description in weight_configs:
        print(f"\n--- Configuration: {description} (α={alpha}, β={beta}) ---")

        # Créer une config modifiée
        test_config = copy.deepcopy(config)
        test_config.cosine_weight = alpha
        test_config.magnitude_weight = beta
        test_config.global_rounds = 20  # Réduire pour l'ablation

        # Fixer la graine
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

        # Entraînement
        _, history, _ = federated_learning_adaptive_clustering(test_config)

        # Collecter les résultats
        final_acc = history['test_acc'][-1]
        max_acc = max(history['test_acc'])
        avg_clusters = np.mean(history['cluster_count'])

        results_summary.append({
            'config': description,
            'alpha': alpha,
            'beta': beta,
            'final_acc': final_acc,
            'max_acc': max_acc,
            'avg_clusters': avg_clusters
        })

        print(f"  Précision finale: {final_acc:.2f}%")
        print(f"  Précision max: {max_acc:.2f}%")
        print(f"  Clusters moyens: {avg_clusters:.1f}")

    # Visualiser les résultats de l'ablation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    configs = [r['config'] for r in results_summary]
    alphas = [r['alpha'] for r in results_summary]

    # 1. Précision finale
    axes[0].bar(range(len(results_summary)),
                [r['final_acc'] for r in results_summary],
                color=plt.cm.viridis(np.linspace(0.2, 0.8, len(results_summary))),
                edgecolor='black', linewidth=1.5)
    axes[0].set_xticks(range(len(results_summary)))
    axes[0].set_xticklabels([f"α={r['alpha']}\nβ={r['beta']}" for r in results_summary],
                             fontsize=9)
    axes[0].set_ylabel('Précision (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Précision Finale selon les Poids', fontsize=13, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)

    # 2. Précision maximale
    axes[1].bar(range(len(results_summary)),
                [r['max_acc'] for r in results_summary],
                color=plt.cm.plasma(np.linspace(0.2, 0.8, len(results_summary))),
                edgecolor='black', linewidth=1.5)
    axes[1].set_xticks(range(len(results_summary)))
    axes[1].set_xticklabels([f"α={r['alpha']}\nβ={r['beta']}" for r in results_summary],
                             fontsize=9)
    axes[1].set_ylabel('Précision (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Précision Maximale selon les Poids', fontsize=13, fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.3)

    # 3. Nombre moyen de clusters
    axes[2].bar(range(len(results_summary)),
                [r['avg_clusters'] for r in results_summary],
                color=plt.cm.coolwarm(np.linspace(0.2, 0.8, len(results_summary))),
                edgecolor='black', linewidth=1.5)
    axes[2].set_xticks(range(len(results_summary)))
    axes[2].set_xticklabels([f"α={r['alpha']}\nβ={r['beta']}" for r in results_summary],
                             fontsize=9)
    axes[2].set_ylabel('Nombre de Clusters', fontsize=12, fontweight='bold')
    axes[2].set_title('Clusters Moyens selon les Poids', fontsize=13, fontweight='bold')
    axes[2].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('fl_adaptive_ablation_tensorflow.png', dpi=300, bbox_inches='tight')
    print("\nGraphiques d'ablation sauvegardés: fl_adaptive_ablation_tensorflow.png")
    plt.show()

    return results_summary

# ============================================================================
# 14. COMPARAISON AVEC BASELINES
# ============================================================================

def compare_with_baselines(config):
    """
    Compare FL-Adaptive_Clustering avec des méthodes baseline
    """
    print("\n" + "="*70)
    print("COMPARAISON AVEC BASELINES")
    print("="*70)

    methods = [
        ("FedAvg", 1.0, 0.0, False, False),  # Direction only, no robust
        ("FedProx", 0.6, 0.4, False, False),  # Hybrid, no robust
        ("FL-Adaptive (Ours)", 0.6, 0.4, True, True),  # Hybrid + robust
    ]

    all_results = {}

    for method_name, alpha, beta, use_clustering, use_robust in methods:
        print(f"\n--- Méthode: {method_name} ---")

        # Configuration
        test_config = copy.deepcopy(config)
        test_config.cosine_weight = alpha
        test_config.magnitude_weight = beta
        test_config.global_rounds = 30

        # Désactiver le clustering pour FedAvg
        if not use_clustering:
            test_config.initial_clusters = 1

        # Fixer la graine
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

        # Entraînement
        _, history, quality_history = federated_learning_adaptive_clustering(test_config)

        all_results[method_name] = {
            'history': history,
            'quality': quality_history
        }

        print(f"  Précision finale: {history['test_acc'][-1]:.2f}%")
        print(f"  Précision max: {max(history['test_acc']):.2f}%")

    # Visualisation comparative
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['#E63946', '#457B9D', '#2A9D8F']

    # 1. Courbes de précision
    for idx, (method_name, results) in enumerate(all_results.items()):
        axes[0, 0].plot(results['history']['test_acc'],
                       marker='o' if idx == 2 else '',
                       linewidth=2.5 if idx == 2 else 2,
                       color=colors[idx],
                       label=method_name,
                       alpha=0.9)

    axes[0, 0].set_xlabel('Round', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Comparaison de Précision', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)

    # 2. Courbes de loss
    for idx, (method_name, results) in enumerate(all_results.items()):
        axes[0, 1].plot(results['history']['test_loss'],
                       linewidth=2.5 if idx == 2 else 2,
                       color=colors[idx],
                       label=method_name,
                       alpha=0.9)

    axes[0, 1].set_xlabel('Round', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Test Loss', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Comparaison de Loss', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)

    # 3. Nombre de clusters
    for idx, (method_name, results) in enumerate(all_results.items()):
        axes[1, 0].plot(results['history']['cluster_count'],
                       linewidth=2.5 if idx == 2 else 2,
                       color=colors[idx],
                       label=method_name,
                       alpha=0.9)

    axes[1, 0].set_xlabel('Round', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Nombre de Clusters', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Evolution du Clustering', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)

    # 4. Barres comparatives - précision finale
    final_accs = [results['history']['test_acc'][-1]
                  for results in all_results.values()]
    method_names = list(all_results.keys())

    bars = axes[1, 1].bar(range(len(method_names)), final_accs,
                          color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_xticks(range(len(method_names)))
    axes[1, 1].set_xticklabels(method_names, fontsize=10)
    axes[1, 1].set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Précision Finale - Comparaison', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, axis='y', alpha=0.3)

    # Ajouter les valeurs sur les barres
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('fl_baseline_comparison_tensorflow.png', dpi=300, bbox_inches='tight')
    print("\nGraphiques de comparaison sauvegardés: fl_baseline_comparison_tensorflow.png")
    plt.show()

    return all_results

# ============================================================================
# 15. EXÉCUTION PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("FL-ADAPTIVE_CLUSTERING: DIRECTION + MAGNITUDE (TENSORFLOW)")
    print("Apprentissage Fédéré avec Métrique Hybride et Agrégation Robuste")
    print("="*70)

    # Configuration
    config = Config()

    # Fixer les graines pour reproductibilité
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    # Menu interactif
    print("\nChoisissez le mode d'exécution:")
    print("1. Entraînement complet (50 rounds)")
    print("2. Étude d'ablation (tester différents poids)")
    print("3. Comparaison avec baselines (FedAvg, FedProx)")
    print("4. Tout exécuter (complet + ablation + comparaison)")

    choice = input("\nVotre choix (1-4): ").strip()

    if choice == "1" or choice == "4":
        print("\n" + "="*70)
        print("MODE 1: ENTRAÎNEMENT COMPLET")
        print("="*70)

        # Entraînement complet
        final_model, history, clustering_quality = federated_learning_adaptive_clustering(config)

        # Visualisations
        plot_detailed_results(history, clustering_quality)
        plot_convergence_comparison(history)

        # Résumé
        print("\n" + "="*70)
        print("RÉSUMÉ FINAL - ENTRAÎNEMENT COMPLET")
        print("="*70)
        print(f"Précision finale: {history['test_acc'][-1]:.2f}%")
        print(f"Précision maximale: {max(history['test_acc']):.2f}%")
        print(f"Précision moyenne (10 derniers rounds): {np.mean(history['test_acc'][-10:]):.2f}%")
        print(f"Loss finale: {history['test_loss'][-1]:.4f}")
        print(f"Nombre moyen de clusters: {np.mean(history['cluster_count']):.1f}")
        print(f"Silhouette final: {clustering_quality[-1]['silhouette']:.3f}")
        print(f"Cohésion de direction finale: {clustering_quality[-1]['avg_direction_cohesion']:.3f}")
        print("="*70)

    if choice == "2" or choice == "4":
        print("\n" + "="*70)
        print("MODE 2: ÉTUDE D'ABLATION")
        print("="*70)

        ablation_results = ablation_study(config)

        # Tableau récapitulatif
        print("\n" + "="*70)
        print("TABLEAU RÉCAPITULATIF - ÉTUDE D'ABLATION")
        print("="*70)
        print(f"{'Configuration':<30} {'α':>5} {'β':>5} {'Acc Final':>10} {'Acc Max':>10} {'Clusters':>10}")
        print("-"*70)
        for r in ablation_results:
            print(f"{r['config']:<30} {r['alpha']:>5.1f} {r['beta']:>5.1f} "
                  f"{r['final_acc']:>9.2f}% {r['max_acc']:>9.2f}% {r['avg_clusters']:>10.1f}")
        print("="*70)

    if choice == "3" or choice == "4":
        print("\n" + "="*70)
        print("MODE 3: COMPARAISON AVEC BASELINES")
        print("="*70)

        comparison_results = compare_with_baselines(config)

        # Tableau récapitulatif
        print("\n" + "="*70)
        print("TABLEAU RÉCAPITULATIF - COMPARAISON BASELINES")
        print("="*70)
        print(f"{'Méthode':<25} {'Acc Finale':>12} {'Acc Max':>12} {'Loss Finale':>12}")
        print("-"*70)
        for method_name, results in comparison_results.items():
            final_acc = results['history']['test_acc'][-1]
            max_acc = max(results['history']['test_acc'])
            final_loss = results['history']['test_loss'][-1]
            print(f"{method_name:<25} {final_acc:>11.2f}% {max_acc:>11.2f}% {final_loss:>12.4f}")
        print("="*70)

    print("\n✅ Expériences terminées avec succès!")
    print("📊 Fichiers générés:")
    print("   - fl_adaptive_clustering_tensorflow.png")
    print("   - fl_convergence_tensorflow.png")
    if choice in ["2", "4"]:
        print("   - fl_adaptive_ablation_tensorflow.png")
    if choice in ["3", "4"]:
        print("   - fl_baseline_comparison_tensorflow.png")