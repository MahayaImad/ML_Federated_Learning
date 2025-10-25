"""
Implementation: UCB-based Client Selection with Model Divergence for Vanilla FL
Pas de clustering - Tous les clients dans un seul "pool" global
"""

import numpy as np
from collections import deque
from typing import List, Dict, Optional
import logging

# ============================================================================
# PARTIE 1: Adapter WeightedConsensusTracker pour Vanilla FL
# ============================================================================

class GlobalConsensusTracker:
    """
    Version simplifiée pour Vanilla FL (un seul pool global de clients)
    Suit la direction consensus de TOUS les clients
    """

    def __init__(self,
                 window_size: int = 5,
                 smooth_factor: float = 0.8,
                 min_quality: float = 0.1):
        """
        Args:
            window_size: Nombre de rounds récents à considérer
            smooth_factor: Facteur de lissage temporel (0-1)
            min_quality: Qualité minimale pour éviter division par zéro
        """
        self.window_size = window_size
        self.smooth_factor = smooth_factor
        self.min_quality = min_quality

        # Stockage des updates récentes PAR ROUND
        # Chaque round contient plusieurs clients
        self.recent_rounds = deque(maxlen=window_size)

        # Direction consensus actuelle
        self.consensus_direction = None
        self.consensus_magnitude = 0.0

        # Round actuel
        self.current_round_updates = []

        logging.info(f"GlobalConsensusTracker initialized: window={window_size}, smooth={smooth_factor}")

    def start_round(self):
        """Commence un nouveau round"""
        self.current_round_updates = []

    def add_client_update(self,
                         client_update: np.ndarray,
                         client_quality: float = 1.0):
        """
        Ajoute une mise à jour client au round actuel

        Args:
            client_update: Vecteur de mise à jour (W_client - W_global)
            client_quality: Qualité du client (reward moyen historique)
        """
        quality = np.clip(client_quality, self.min_quality, 10.0)

        self.current_round_updates.append({
            'update': client_update.copy(),
            'quality': quality
        })

    def end_round(self):
        """
        Termine le round et met à jour le consensus avec tous les clients du round
        """
        if len(self.current_round_updates) == 0:
            return

        # Agréger les updates du round actuel (moyenne pondérée)
        updates = [item['update'] for item in self.current_round_updates]
        qualities = [item['quality'] for item in self.current_round_updates]

        weighted_sum = np.zeros_like(updates[0])
        total_quality = 0.0

        for update, quality in zip(updates, qualities):
            weighted_sum += quality * update
            total_quality += quality

        round_consensus = weighted_sum / total_quality

        # Stocker consensus du round
        self.recent_rounds.append({
            'consensus': round_consensus,
            'num_clients': len(self.current_round_updates),
            'avg_quality': total_quality / len(self.current_round_updates)
        })

        # Recalculer consensus global
        self._recompute_global_consensus()

        # Reset round actuel
        self.current_round_updates = []

    def _recompute_global_consensus(self):
        """
        Recalcule le consensus global sur les derniers rounds
        """
        if len(self.recent_rounds) == 0:
            return

        # Moyenne pondérée des consensus de rounds
        weighted_sum = np.zeros_like(self.recent_rounds[0]['consensus'])
        total_weight = 0.0

        for round_data in self.recent_rounds:
            # Pondération par nombre de clients et qualité moyenne
            weight = round_data['num_clients'] * round_data['avg_quality']
            weighted_sum += weight * round_data['consensus']
            total_weight += weight

        new_consensus = weighted_sum / total_weight

        # Temporal smoothing
        if self.consensus_direction is None:
            self.consensus_direction = new_consensus
        else:
            self.consensus_direction = (
                self.smooth_factor * self.consensus_direction
                + (1 - self.smooth_factor) * new_consensus
            )

        self.consensus_magnitude = np.linalg.norm(self.consensus_direction)

    def get_consensus(self) -> Optional[np.ndarray]:
        """Retourne la direction consensus actuelle"""
        return self.consensus_direction

    def get_magnitude(self) -> float:
        """Retourne la magnitude du consensus"""
        return self.consensus_magnitude

    def reset(self):
        """Reset complet (généralement pas nécessaire en Vanilla FL)"""
        self.recent_rounds.clear()
        self.current_round_updates = []
        self.consensus_direction = None
        self.consensus_magnitude = 0.0
        logging.info("Global consensus tracker reset")


# ============================================================================
# PARTIE 2: DivergenceScorer (identique, réutilisable)
# ============================================================================

class DivergenceScorer:
    """
    Calcule le score de divergence avec fonction U-shape
    (Identique à la version clustering)
    """

    def __init__(self,
                 optimal_range: tuple = (0.3, 0.8),
                 high_alignment_score: float = 0.5,
                 optimal_score: float = 1.0,
                 low_alignment_score: float = 0.7,
                 negative_alignment_score: float = 0.2,
                 magnitude_weight: float = 0.3,
                 outlier_threshold: float = 5.0):

        self.optimal_min, self.optimal_max = optimal_range
        self.high_align_score = high_alignment_score
        self.optimal_score = optimal_score
        self.low_align_score = low_alignment_score
        self.negative_align_score = negative_alignment_score
        self.magnitude_weight = magnitude_weight
        self.outlier_threshold = outlier_threshold

    def compute_score(self,
                      client_update: np.ndarray,
                      consensus_direction: np.ndarray,
                      consensus_magnitude: float) -> Dict[str, float]:

        alignment = self._cosine_similarity(client_update, consensus_direction)
        base_score = self._u_shape_score(alignment)

        client_magnitude = np.linalg.norm(client_update)
        magnitude_ratio = client_magnitude / (consensus_magnitude + 1e-10)
        magnitude_factor = np.tanh(magnitude_ratio)

        is_outlier = magnitude_ratio > self.outlier_threshold
        if is_outlier:
            magnitude_factor *= 0.3

        final_score = base_score * (
            (1 - self.magnitude_weight)
            + self.magnitude_weight * magnitude_factor
        )

        return {
            'score': final_score,
            'alignment': alignment,
            'magnitude_ratio': magnitude_ratio,
            'magnitude_factor': magnitude_factor,
            'base_score': base_score,
            'is_outlier': is_outlier
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        dot_product = np.dot(vec1.flatten(), vec2.flatten())
        return dot_product / (norm1 * norm2)

    def _u_shape_score(self, alignment: float) -> float:
        if alignment > self.optimal_max:
            return self.high_align_score
        elif alignment >= self.optimal_min:
            return self.optimal_score
        elif alignment >= 0.0:
            return self.low_align_score
        else:
            return self.negative_align_score


# ============================================================================
# PARTIE 3: UCB Selector pour Vanilla FL
# ============================================================================

class VanillaUCBSelector:
    """
    Sélection UCB pour Vanilla FL (sans clustering)
    Tous les clients sont dans un seul pool global
    """

    def __init__(self,
                 num_clients: int,
                 ucb_c: float = 1.414,
                 freshness_lambda: float = 0.2,
                 reward_window: int = 10,
                 phase_thresholds: tuple = (0.2, 0.7)):
        """
        Args:
            num_clients: Nombre total de clients
            ucb_c: Paramètre exploration UCB
            freshness_lambda: Poids du bonus freshness
            reward_window: Taille fenêtre pour moyenne reward
            phase_thresholds: Seuils pour phases (début, fin)
        """
        self.num_clients = num_clients
        self.ucb_c = ucb_c
        self.freshness_lambda = freshness_lambda
        self.reward_window = reward_window
        self.phase_thresholds = phase_thresholds

        # Historique par client
        self.client_rewards = {i: deque(maxlen=reward_window)
                               for i in range(num_clients)}
        self.client_selection_count = {i: 0 for i in range(num_clients)}
        self.client_last_selected = {i: -100 for i in range(num_clients)}
        self.client_quality = {i: 1.0 for i in range(num_clients)}

        # Consensus tracker GLOBAL
        self.consensus_tracker = GlobalConsensusTracker()

        # Divergence scorer
        self.divergence_scorer = DivergenceScorer()

        # Distribution globale des labels (pour diversité)
        self.global_label_distribution = None

        logging.info(f"VanillaUCBSelector initialized for {num_clients} clients")

    def initialize_global_distribution(self, clients_data):
        """
        Initialise la distribution globale des labels

        Args:
            clients_data: Liste de (x_train, y_train) pour tous les clients
        """
        # Compter labels de tous les clients
        num_classes = clients_data[0][1].shape[1]  # One-hot encoded
        self.global_label_distribution = np.zeros(num_classes)

        for x_train, y_train in clients_data:
            labels = np.argmax(y_train, axis=1)
            unique, counts = np.unique(labels, return_counts=True)
            self.global_label_distribution[unique] += counts

        logging.info(f"Global label distribution initialized: {self.global_label_distribution}")

    def compute_reward(self,
                      client_id: int,
                      client_update: np.ndarray,
                      client_label_dist: np.ndarray,
                      round_t: int,
                      total_rounds: int) -> float:
        """
        Calcule le reward multi-critères pour un client

        Args:
            client_id: ID du client
            client_update: Mise à jour du client (W_client - W_global)
            client_label_dist: Distribution des labels du client
            round_t: Round actuel
            total_rounds: Total rounds prévus

        Returns:
            reward: Récompense composite
        """
        # [1] Déterminer phase d'entraînement
        phase = round_t / total_rounds
        weights = self._get_phase_weights(phase)

        # [2] Score de divergence
        consensus = self.consensus_tracker.get_consensus()
        consensus_mag = self.consensus_tracker.get_magnitude()

        if consensus is None or round_t < 3:
            # Cold start: pas encore de consensus
            divergence_score = 0.5  # Neutre
        else:
            div_result = self.divergence_scorer.compute_score(
                client_update, consensus, consensus_mag
            )
            divergence_score = div_result['score']

            if div_result['is_outlier']:
                logging.warning(
                    f"Client {client_id} detected as outlier: "
                    f"alignment={div_result['alignment']:.3f}, "
                    f"magnitude_ratio={div_result['magnitude_ratio']:.2f}"
                )

        # [3] Diversité JS (par rapport à distribution GLOBALE)
        diversity_js = self._compute_js_divergence(
            client_label_dist,
            self.global_label_distribution
        )

        # [4] Couverture classes rares
        coverage_rare = self._compute_coverage_rare(
            client_label_dist,
            self.global_label_distribution
        )

        # [5] Reward composite
        reward = (
            weights['divergence'] * divergence_score
            + weights['diversity'] * diversity_js
            + weights['coverage'] * coverage_rare
        )

        # [6] Enregistrer reward
        self.client_rewards[client_id].append(reward)

        return reward

    def _get_phase_weights(self, phase: float) -> Dict[str, float]:
        """Retourne poids adaptatifs selon phase"""
        phase_start, phase_end = self.phase_thresholds

        if phase < phase_start:
            # Début: Focus diversité et couverture
            return {
                'divergence': 0.30,
                'diversity': 0.50,
                'coverage': 0.20
            }
        elif phase < phase_end:
            # Milieu: Balance
            return {
                'divergence': 0.45,
                'diversity': 0.35,
                'coverage': 0.20
            }
        else:
            # Fin: Focus convergence
            return {
                'divergence': 0.60,
                'diversity': 0.25,
                'coverage': 0.15
            }

    def _compute_js_divergence(self, dist1: np.ndarray, dist2: np.ndarray) -> float:
        """Calcule Jensen-Shannon divergence"""
        from scipy.stats import entropy

        dist1 = np.array(dist1) + 1e-10
        dist2 = np.array(dist2) + 1e-10
        dist1 = dist1 / dist1.sum()
        dist2 = dist2 / dist2.sum()

        m = 0.5 * (dist1 + dist2)
        js_div = 0.5 * (entropy(dist1, m) + entropy(dist2, m))

        return np.sqrt(js_div)

    def _compute_coverage_rare(self,
                               client_dist: np.ndarray,
                               global_dist: np.ndarray,
                               rare_threshold: float = 0.05) -> float:
        """Calcule score de couverture des classes rares"""
        client_norm = client_dist / (client_dist.sum() + 1e-10)
        global_norm = global_dist / (global_dist.sum() + 1e-10)

        rare_classes = global_norm < rare_threshold

        if rare_classes.sum() == 0:
            return 0.0

        client_has_rare = client_norm[rare_classes] > 0
        coverage = client_has_rare.sum() / rare_classes.sum()

        return coverage

    def select_clients(self,
                      available_clients: List[int],
                      k: int,
                      round_t: int) -> List[int]:
        """
        Sélectionne k clients selon UCB

        Args:
            available_clients: Liste des clients disponibles
            k: Nombre de clients à sélectionner
            round_t: Round actuel

        Returns:
            selected: Liste des k clients sélectionnés
        """
        scores = {}

        for client_id in available_clients:
            scores[client_id] = self._compute_ucb_score(client_id, round_t)

        # Sélectionner top-k
        selected = sorted(scores.keys(), key=lambda c: -scores[c])[:k]

        # Mettre à jour compteurs
        for client_id in selected:
            self.client_selection_count[client_id] += 1
            self.client_last_selected[client_id] = round_t

        logging.info(f"Round {round_t}: Selected clients {selected}")
        logging.debug(f"Scores: {[(c, f'{scores[c]:.3f}') for c in selected]}")

        return selected

    def _compute_ucb_score(self, client_id: int, round_t: int) -> float:
        """Calcule le score UCB complet pour un client"""
        # [1] Exploitation: Q̄
        if len(self.client_rewards[client_id]) > 0:
            Q_bar = np.mean(list(self.client_rewards[client_id]))
        else:
            Q_bar = 0.5  # Prior neutre

        # [2] Exploration: bonus UCB
        N = self.client_selection_count[client_id]
        if N == 0:
            exploration_bonus = float('inf')  # Forcer essai au moins une fois
        else:
            exploration_bonus = self.ucb_c * np.sqrt(np.log(round_t + 1) / N)

        # [3] Freshness: bonus si pas sélectionné récemment
        staleness = round_t - self.client_last_selected[client_id]
        freshness_bonus = self.freshness_lambda * (staleness / (round_t + 1))

        # Score final
        score = Q_bar + exploration_bonus + freshness_bonus

        return score

    def update_after_training(self,
                             client_id: int,
                             client_update: np.ndarray,
                             reward: float):
        """
        Mise à jour après entraînement d'un client
        """
        # Mettre à jour qualité du client
        if len(self.client_rewards[client_id]) > 0:
            self.client_quality[client_id] = np.mean(
                list(self.client_rewards[client_id])
            )

        # Ajouter au consensus du round actuel
        self.consensus_tracker.add_client_update(
            client_update,
            self.client_quality[client_id]
        )

    def finalize_round(self):
        """
        Finalise le round actuel (calcule consensus du round)
        """
        self.consensus_tracker.end_round()

    def start_new_round(self):
        """
        Démarre un nouveau round
        """
        self.consensus_tracker.start_round()

    def get_statistics(self) -> Dict:
        """Retourne statistiques pour monitoring"""
        return {
            'selection_counts': dict(self.client_selection_count),
            'client_qualities': dict(self.client_quality),
            'consensus_magnitude': self.consensus_tracker.get_magnitude(),
            'num_rounds_in_consensus': len(self.consensus_tracker.recent_rounds)
        }


# ============================================================================
# PARTIE 4: Training Loop pour Vanilla FL avec UCB
# ============================================================================

def train_vanilla_fl_ucb(clients, test_data, args):
    """
    Entraînement Vanilla FL avec UCB + Model Divergence

    Args:
        clients: Liste des FederatedClient
        test_data: Données de test (x_test, y_test)
        args: Arguments de configuration
    """
    import time
    from models import initialize_global_model
    from utils import save_client_stats_csv
    from config import SAVE_CLIENTS_STATS, VERBOSE

    print(" Train of Vanilla FL with UCB Divergence...")

    global_model = initialize_global_model(args.dataset)

    # ✨ Initialiser UCB Selector
    ucb_selector = VanillaUCBSelector(
        num_clients=len(clients),
        ucb_c=2.0,  # Plus élevé pour Non-IID
        freshness_lambda=0.2,
        reward_window=10
    )

    # Initialiser distribution globale
    clients_data = [(c.x_train, c.y_train) for c in clients]
    ucb_selector.initialize_global_distribution(clients_data)

    # Résultats
    results = {
        'accuracy_history': [],
        'communication_costs': [],
        'round_times': []
    }

    # Training loop
    for round_num in range(args.rounds):
        start_time = time.time()

        if VERBOSE:
            print(f"  Round {round_num + 1}/{args.rounds}")

        # ✨ Démarrer nouveau round dans consensus tracker
        ucb_selector.start_new_round()

        # ✨ Sélection UCB des clients
        selection_ratio = getattr(args, 'client_selection_ratio', 0.3)
        k = max(1, int(len(clients) * selection_ratio))

        selected_ids = ucb_selector.select_clients(
            available_clients=list(range(len(clients))),
            k=k,
            round_t=round_num
        )

        if VERBOSE:
            print(f"    Selected {len(selected_ids)}/{len(clients)} clients: {selected_ids}")

        # Entraînement local
        client_updates = []
        client_sizes = []
        global_weights_before = global_model.get_weights()

        for client_id in selected_ids:
            client = clients[client_id]

            # ✨ Entraînement
            client.update_model(global_model)
            client.train_local(args.local_epochs)

            client_weights = client.local_model.get_weights()
            client_updates.append(client_weights)
            client_sizes.append(len(client.x_train))

            # ✨ Calculer update vector
            client_update_vector = flatten_weights(client_weights) - flatten_weights(global_weights_before)

            # ✨ Distribution labels du client
            client_label_dist = get_client_label_distribution(client, args.dataset)

            # ✨ Calculer reward
            reward = ucb_selector.compute_reward(
                client_id=client_id,
                client_update=client_update_vector,
                client_label_dist=client_label_dist,
                round_t=round_num,
                total_rounds=args.rounds
            )

            # ✨ Update consensus
            ucb_selector.update_after_training(
                client_id=client_id,
                client_update=client_update_vector,
                reward=reward
            )

            if VERBOSE:
                print(f"      Client {client_id}: reward={reward:.3f}")

        # ✨ Finaliser round (calculer consensus)
        ucb_selector.finalize_round()

        # Agrégation FedAvg GLOBALE
        global_weights = fedavg_aggregate(client_updates, client_sizes)
        global_model.set_weights(global_weights)

        # Évaluation
        x_test, y_test = test_data
        y_test = np.argmax(y_test, axis=1)
        _, accuracy = global_model.evaluate(x_test, y_test, verbose=0)

        # Métriques
        comm_cost = len(selected_ids) * 2  # Upload + download
        round_time = time.time() - start_time

        results['accuracy_history'].append(accuracy)
        results['communication_costs'].append(comm_cost)
        results['round_times'].append(round_time)

        if VERBOSE:
            print(f"    Accuracy: {accuracy:.4f}, Comm Cost: {comm_cost}, Time: {round_time:.2f}s")

        # ✨ Log statistiques UCB périodiquement
        if (round_num + 1) % 5 == 0:
            stats = ucb_selector.get_statistics()
            print(f"  UCB Statistics:")
            print(f"    Consensus magnitude: {stats['consensus_magnitude']:.3f}")
            sel_counts = stats['selection_counts']
            print(f"    Selection range: [{min(sel_counts.values())}, {max(sel_counts.values())}]")
            qualities = stats['client_qualities']
            print(f"    Quality range: [{min(qualities.values()):.3f}, {max(qualities.values()):.3f}]")

    # Sauvegarde des statistiques clients
    if SAVE_CLIENTS_STATS:
        save_client_stats_csv(clients, args)

    return results


# ============================================================================
# PARTIE 5: Helper Functions
# ============================================================================

def flatten_weights(weights_list):
    """Aplatit liste de poids en vecteur 1D"""
    return np.concatenate([w.flatten() for w in weights_list])


def get_client_label_distribution(client, dataset_name):
    """Récupère distribution des labels d'un client"""
    num_classes = get_num_classes(dataset_name)
    labels = np.argmax(client.y_train, axis=1)
    histogram = np.zeros(num_classes)
    unique, counts = np.unique(labels, return_counts=True)
    histogram[unique] = counts
    return histogram


def get_num_classes(dataset_name):
    """Retourne nombre de classes selon dataset"""
    if dataset_name == 'mnist':
        return 10
    elif dataset_name == 'cifar10':
        return 10
    elif dataset_name == 'cifar100':
        return 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def fedavg_aggregate(model_weights_list, data_sizes):
    """Agrégation FedAvg pondérée"""
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


# ============================================================================
# PARTIE 6: Exemple d'Utilisation
# ============================================================================

if __name__ == "__main__":
    """
    Exemple d'utilisation du VanillaUCBSelector
    """

    # Configuration logging
    logging.basicConfig(level=logging.INFO)

    # Paramètres
    num_clients = 30
    num_rounds = 20
    selection_ratio = 0.3  # 30% des clients

    # Initialiser selector
    selector = VanillaUCBSelector(
        num_clients=num_clients,
        ucb_c=2.0,
        freshness_lambda=0.2
    )

    # Simuler distribution globale
    num_classes = 10
    selector.global_label_distribution = np.random.randint(100, 1000, num_classes)

    # Simuler training
    for round_t in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"Round {round_t + 1}/{num_rounds}")
        print(f"{'='*60}")

        # Démarrer round
        selector.start_new_round()

        # Sélection
        k = max(1, int(num_clients * selection_ratio))
        selected = selector.select_clients(
            available_clients=list(range(num_clients)),
            k=k,
            round_t=round_t
        )

        print(f"Selected {len(selected)} clients: {selected}")

        # Simuler entraînement
        for client_id in selected:
            # Simuler update (aléatoire pour demo)
            client_update = np.random.randn(100)
            client_label_dist = np.random.randint(0, 100, num_classes)

            # Calculer reward
            reward = selector.compute_reward(
                client_id=client_id,
                client_update=client_update,
                client_label_dist=client_label_dist,
                round_t=round_t,
                total_rounds=num_rounds
            )

            # Update consensus
            selector.update_after_training(
                client_id=client_id,
                client_update=client_update,
                reward=reward
            )

        # Finaliser round
        selector.finalize_round()

        # Afficher stats
        if (round_t + 1) % 5 == 0:
            stats = selector.get_statistics()
            print(f"Consensus magnitude: {stats['consensus_magnitude']:.3f}")
            sel_counts = stats['selection_counts']
            print(f"Selection range: [{min(sel_counts.values())}, {max(sel_counts.values())}]")

    # Analyse finale
    final_stats = selector.get_statistics()
    print(f"\n{'='*60}")
    print("FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Selection counts: {final_stats['selection_counts']}")
    print(f"Min selections: {min(final_stats['selection_counts'].values())}")
    print(f"Max selections: {max(final_stats['selection_counts'].values())}")
    print(f"Mean selections: {np.mean(list(final_stats['selection_counts'].values())):.2f}")