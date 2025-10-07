"""
Classe de base pour tous les agrégateurs
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseAggregator(ABC):
    """Classe abstraite pour les agrégateurs fédérés"""

    def __init__(self, name: str = "BaseAggregator"):
        self.name = name
        self.round_number = 0
        self.history = {
            'communication_costs': [],
            'aggregation_times': []
        }

    @abstractmethod
    def aggregate(self, client_updates, client_weights, global_model):
        """
        Agrège les mises à jour des clients

        Args:
            client_updates: Liste des mises à jour des clients
            client_weights: Poids relatifs des clients (taille des données)
            global_model: Modèle global actuel

        Returns:
            new_global_weights: Nouveaux poids du modèle global
        """
        pass

    @abstractmethod
    def prepare_client_update(self, client_id, local_model, global_model):
        """
        Prépare la mise à jour d'un client

        Args:
            client_id: ID du client
            local_model: Modèle local entraîné
            global_model: Modèle global

        Returns:
            client_update: Mise à jour à envoyer au serveur
        """
        pass

    def update_round(self):
        """Met à jour le numéro de tour"""
        self.round_number += 1

    def reset(self):
        """Remet à zéro l'agrégateur"""
        self.round_number = 0
        self.history = {
            'communication_costs': [],
            'aggregation_times': []
        }


    def get_communication_cost(self, client_updates):
        """Calcule le coût de communication avec gestion robuste des formats"""
        total_params = 0

        try:
            for update in client_updates:
                if isinstance(update, dict):
                    # Format SCAFFOLD ou autres avec dictionnaire
                    for key, value in update.items():
                        if isinstance(value, list):
                            for item in value:
                                if hasattr(item, 'shape') and hasattr(item, 'size'):
                                    total_params += item.size
                elif isinstance(update, list):
                    # Format standard (liste de numpy arrays)
                    for layer_update in update:
                        if hasattr(layer_update, 'shape') and hasattr(layer_update, 'size'):
                            total_params += layer_update.size
                elif hasattr(update, 'shape') and hasattr(update, 'size'):
                    # Cas d'un seul array
                    total_params += update.size

        except Exception as e:
            print(f"Erreur calcul communication cost: {e}")
            return 0

        return total_params

    def get_stats(self):
        """Retourne les statistiques de l'agrégateur"""
        return {
            'name': self.name,
            'rounds_completed': self.round_number,
            'total_communication_cost': sum(self.history['communication_costs']),
            'avg_aggregation_time': np.mean(self.history['aggregation_times']) if self.history[
                'aggregation_times'] else 0
        }


def weighted_average(updates, weights):
    """
    Calcule la moyenne pondérée des mises à jour

    Args:
        updates: Liste des mises à jour
        weights: Poids pour chaque mise à jour

    Returns:
        averaged_update: Mise à jour moyennée
    """
    if not updates:
        return None

    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Calculer la moyenne pondérée
    averaged_update = []
    for layer_idx in range(len(updates[0])):
        layer_updates = [update[layer_idx] for update in updates]

        weighted_sum = np.zeros_like(layer_updates[0])
        for update, weight in zip(layer_updates, weights):
            weighted_sum += update * weight

        averaged_update.append(weighted_sum)

    return averaged_update

def subtract_weights(weights1, weights2):
    """Soustrait weights2 de weights1"""
    return [w1 - w2 for w1, w2 in zip(weights1, weights2)]


def add_weights(weights1, weights2):
    """Additionne weights1 et weights2"""
    return [w1 + w2 for w1, w2 in zip(weights1, weights2)]


def scale_weights(weights, factor):
    """Multiplie les poids par un facteur"""
    return [w * factor for w in weights]