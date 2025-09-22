"""
Classes de base pour FL-MPC (copiées de FL_Agregations)
"""
import tensorflow as tf
import numpy as np
import random
import time
from datetime import datetime
import json
from models import copy_model
from config import LOCAL_EPOCHS, BATCH_SIZE, COMMUNICATION_ROUNDS, LEARNING_RATE


class BaseFederatedClient:
    """Client de base pour l'apprentissage fédéré"""

    def __init__(self, client_id, data):
        self.client_id = client_id
        self.x_train, self.y_train = data
        self.local_model = None
        self.global_model_ref = None
        self.training_rounds_count = 0

    def update_model(self, global_model):
        """Met à jour le modèle local avec le modèle global"""
        if self.local_model is None:
            self.local_model = copy_model(global_model)
        else:
            self.local_model.set_weights(global_model.get_weights())
        self.global_model_ref = global_model

    def train_local(self, epochs=LOCAL_EPOCHS):
        """Entraînement local standard"""
        if self.local_model is None:
            raise ValueError("Modèle local non initialisé")

        # Recompiler le modèle pour éviter les conflits
        self.local_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = self.local_model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        self.training_rounds_count += 1
        return self.local_model

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


class BaseFederatedServer:
    """Serveur de base pour l'apprentissage fédéré"""

    def __init__(self, aggregator, model_type="standard"):
        self.aggregator = aggregator
        self.global_model = None
        self.current_round = 0
        self.metrics = {
            'test_accuracy': [],
            'round_times': [],
            'communication_costs': []
        }

    def initialize_model(self, model_type="standard"):
        """Initialise le modèle global"""
        from models import create_cnn_model
        self.global_model = create_cnn_model(model_type)

    def select_clients(self, clients, selection_ratio=1.0):
        """Sélectionne les clients pour un tour"""
        num_selected = max(1, int(len(clients) * selection_ratio))
        return random.sample(clients, num_selected)

    def evaluate(self, test_data):
        """Évalue le modèle global"""
        if self.global_model is None:
            return 0.0
        
        x_test, y_test = test_data
        loss, accuracy = self.global_model.evaluate(x_test, y_test, verbose=0)
        return accuracy

    def train_round(self, clients, test_data, selection_ratio=1.0):
        """Round d'entraînement de base"""
        round_start_time = time.time()

        # Sélection des clients
        selected_clients = self.select_clients(clients, selection_ratio)

        # Phase d'entraînement local
        client_updates = []
        client_weights = []

        for client in selected_clients:
            client.update_model(self.global_model)
            client.train_local()

            # Calculer la mise à jour (différence)
            local_weights = client.local_model.get_weights()
            global_weights = self.global_model.get_weights()
            update = [local_w - global_w for local_w, global_w in 
                     zip(local_weights, global_weights)]
            
            client_updates.append(update)
            client_weights.append(client.get_data_size())

        # Agrégation (FedAvg simple par défaut)
        if hasattr(self.aggregator, 'aggregate'):
            new_weights = self.aggregator.aggregate(
                client_updates, client_weights, self.global_model
            )
        else:
            # FedAvg simple si pas d'agrégateur spécialisé
            new_weights = self._simple_fedavg(client_updates, client_weights)

        self.global_model.set_weights(new_weights)

        # Évaluation et métriques
        test_acc = self.evaluate(test_data)
        round_time = time.time() - round_start_time

        self.metrics['test_accuracy'].append(test_acc)
        self.metrics['round_times'].append(round_time)

        self.current_round += 1
        return test_acc

    def _simple_fedavg(self, client_updates, client_weights):
        """FedAvg simple"""
        total_weight = sum(client_weights)
        weighted_updates = []

        # Pour chaque couche
        for layer_idx in range(len(client_updates[0])):
            layer_sum = None
            for client_idx, update in enumerate(client_updates):
                weight = client_weights[client_idx] / total_weight
                weighted_layer = update[layer_idx] * weight
                
                if layer_sum is None:
                    layer_sum = weighted_layer
                else:
                    layer_sum += weighted_layer
            
            weighted_updates.append(layer_sum)

        # Appliquer les mises à jour
        global_weights = self.global_model.get_weights()
        new_weights = [global_w + update for global_w, update in 
                      zip(global_weights, weighted_updates)]
        
        return new_weights

    def train_federated(self, clients, test_data, rounds=COMMUNICATION_ROUNDS):
        """Entraînement fédéré complet"""
        if self.global_model is None:
            self.initialize_model()

        print(f"Début entraînement avec {len(clients)} clients pour {rounds} rounds")

        for round_num in range(rounds):
            test_acc = self.train_round(clients, test_data)
            
            if round_num % 5 == 0:
                print(f"Round {round_num}: Accuracy = {test_acc:.4f}")

        final_acc = self.evaluate(test_data)
        print(f"Accuracy finale: {final_acc:.4f}")

        return self.global_model, self.metrics


# Utilitaires pour FedAvg simple
class SimpleFedAvgAggregator:
    """Agrégateur FedAvg simple pour comparaison"""
    
    def __init__(self):
        self.name = "Simple-FedAvg"
    
    def prepare_client_update(self, client_id, local_model, global_model):
        """Prépare la mise à jour client"""
        local_weights = local_model.get_weights()
        global_weights = global_model.get_weights()
        return [local_w - global_w for local_w, global_w in 
                zip(local_weights, global_weights)]