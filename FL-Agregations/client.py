"""
Simulation des clients fédérés
"""
import tensorflow as tf
import numpy as np
from models import copy_model
from config import LOCAL_EPOCHS, BATCH_SIZE

class FederatedClient:
    """Client pour l'apprentissage fédéré"""

    def __init__(self, client_id, data, aggregator):
        self.client_id = client_id
        self.x_train, self.y_train = data
        self.aggregator = aggregator
        self.local_model = None
        self.global_model_ref = None
        self.training_rounds_count = 0
        self.training_history = []

    def update_model(self, global_model):
        """Met à jour le modèle local avec le modèle global"""
        if self.local_model is None:
            # Première fois seulement
            self.local_model = copy_model(global_model)
        else:
            # Juste copier les poids pour ne pas gaspier de mémoire
            self.local_model.set_weights(global_model.get_weights())

        self.global_model_ref = global_model

    def train_local(self, epochs=LOCAL_EPOCHS):
        """Entraînement local selon le type d'agrégateur"""
        if self.local_model is None:
            raise ValueError("Modèle local non initialisé")

        # Entraînement spécialisé pour FedProx
        if hasattr(self.aggregator, 'train_client_with_proximal_term'):
            trained_model = self.aggregator.train_client_with_proximal_term(
                self.local_model, self.global_model_ref,
                self.x_train, self.y_train, epochs
            )
        else:
            # Entraînement standard
            history = self.local_model.fit(
                self.x_train, self.y_train,
                epochs=epochs,
                batch_size=BATCH_SIZE,
                verbose=0,
                validation_split=0.1
            )
            # Incrémenter le compteur
            self.training_rounds_count += 1

            MAX_HISTORY_SIZE = 3  # Garder seulement les 3 derniers
            if len(self.training_history) >= MAX_HISTORY_SIZE:
                self.training_history.pop(0)

            self.training_history.append(history.history)
            trained_model = self.local_model

        return trained_model

    def get_update(self, global_model):
        """Récupère la mise à jour pour le serveur"""
        return self.aggregator.prepare_client_update(
            self.client_id, self.local_model, global_model
        )

    def get_data_size(self):
        """Retourne la taille des données locales"""
        return len(self.x_train)

    def evaluate_local(self):
        """Évalue le modèle local"""
        if self.local_model is None:
            return 0.0
        loss, accuracy = self.local_model.evaluate(
            self.x_train, self.y_train, verbose=0
        )
        return accuracy

    def get_class_distribution(self):
        """Retourne la distribution des classes"""
        return np.sum(self.y_train, axis=0)

    def simulate_dropout(self, dropout_prob=0.1):
        """Simule la déconnexion d'un client"""
        return np.random.random() > dropout_prob

    def get_client_stats(self):
        """Retourne les statistiques du client"""
        return {
            'client_id': self.client_id,
            'data_size': self.get_data_size(),
            'class_distribution': self.get_class_distribution().tolist(),
            'local_accuracy': self.evaluate_local(),
            'training_rounds': self.training_rounds_count
        }

class RobustFederatedClient(FederatedClient):
    """Client robuste avec gestion d'erreurs"""

    def __init__(self, client_id, data, aggregator, max_retries=3):
        super().__init__(client_id, data, aggregator)
        self.max_retries = max_retries
        self.failed_rounds = 0

    def train_local(self, epochs=LOCAL_EPOCHS):
        """Entraînement avec gestion d'erreurs"""
        for attempt in range(self.max_retries):
            try:
                return super().train_local(epochs)
            except Exception as e:
                print(f"Client {self.client_id}: Erreur d'entraînement (tentative {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    self.failed_rounds += 1
                    return self.local_model

    def get_update(self, global_model):
        """Récupère la mise à jour pour le serveur"""
        # Vérifier si l'agrégateur a une méthode spécialisée
        if hasattr(self.aggregator, 'prepare_client_update'):
            return self.aggregator.prepare_client_update(
                self.client_id, self.local_model, global_model
            )
        else:
            # Fallback vers FedAvg standard
            from aggregations.base_aggregator import subtract_weights
            local_weights = self.local_model.get_weights()
            global_weights = global_model.get_weights()
            return subtract_weights(local_weights, global_weights)


class ResourceConstrainedClient(FederatedClient):
    """Client avec contraintes de ressources"""

    def __init__(self, client_id, data, aggregator, compute_budget=1.0, bandwidth_limit=1.0):
        super().__init__(client_id, data, aggregator)
        self.compute_budget = compute_budget  # 0.0 à 1.0
        self.bandwidth_limit = bandwidth_limit  # 0.0 à 1.0

    def train_local(self, epochs=LOCAL_EPOCHS):
        """Entraînement adapté aux ressources"""

        if self.local_model is None:
            raise ValueError("Modèle local non initialisé")

            # AJOUTER une validation des données
        if len(self.x_train) == 0:
            print(f"Client {self.client_id}: Aucune donnée d'entraînement")
            return self.local_model

        # Ajuster les époques selon le budget de calcul
        adjusted_epochs = max(1, int(epochs * self.compute_budget))

        # Ajuster la taille de batch selon les ressources
        adjusted_batch_size = max(8, int(BATCH_SIZE * self.compute_budget))

        if hasattr(self.aggregator, 'train_client_with_proximal_term'):
            return self.aggregator.train_client_with_proximal_term(
                self.local_model, self.global_model_ref,
                self.x_train, self.y_train, adjusted_epochs
            )
        else:
            history = self.local_model.fit(
                self.x_train, self.y_train,
                epochs=adjusted_epochs,
                batch_size=adjusted_batch_size,
                verbose=0
            )
            self.training_history.append(history.history)

        return self.local_model

    def should_participate(self):
        """Détermine si le client peut participer ce tour"""
        # Simulation de disponibilité basée sur les ressources
        availability = min(self.compute_budget, self.bandwidth_limit)
        return np.random.random() < availability