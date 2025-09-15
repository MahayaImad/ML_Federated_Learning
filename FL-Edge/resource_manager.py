"""
Gestionnaire de ressources pour dispositifs edge
"""
import numpy as np
import random
from typing import List, Dict, Any, Optional


class ResourceManager:
    """Gestionnaire intelligent des ressources edge"""

    def __init__(self, privacy_level='medium'):
        self.privacy_level = privacy_level
        self.resource_policies = self._initialize_policies()
        self.device_profiles = {}
        self.resource_history = []

        print(f"⚡ Gestionnaire de ressources initialisé (confidentialité: {privacy_level})")

    def _initialize_policies(self):
        """Initialise les politiques de gestion des ressources"""
        return {
            'battery_management': {
                'critical_threshold': 15.0,  # %
                'conservative_threshold': 30.0,  # %
                'aggressive_threshold': 50.0  # %
            },
            'cpu_management': {
                'max_load_threshold': 0.85,
                'target_load_threshold': 0.70,
                'min_available_threshold': 0.15
            },
            'memory_management': {
                'max_usage_threshold': 0.90,
                'target_usage_threshold': 0.75,
                'min_available_gb': 0.5
            },
            'network_management': {
                'min_quality_threshold': 0.3,
                'preferred_quality_threshold': 0.7,
                'data_usage_limit_mb': 1000  # Par heure
            }
        }

    def select_available_devices(self, devices, battery_aware=True, simulate_failures=False):
        """Sélectionne les dispositifs disponibles selon les politiques"""
        available_devices = []

        for device in devices:
            # Vérifications de base
            if not device.is_online:
                continue

            # Simulation de pannes si activée
            if simulate_failures and self._simulate_device_failure(device):
                device.is_online = False
                continue

            # Vérifications des ressources
            if not self._check_resource_availability(device, battery_aware):
                continue

            # Vérifications de confidentialité
            if not self._check_privacy_constraints(device):
                continue

            available_devices.append(device)

        # Logging de la sélection
        self._log_selection_results(devices, available_devices)

        return available_devices

    def _check_resource_availability(self, device, battery_aware):
        """Vérifie la disponibilité des ressources d'un dispositif"""
        policies = self.resource_policies

        # Vérification batterie
        if battery_aware and device.specs['battery_capacity'] != float('inf'):
            if device.battery_level < policies['battery_management']['critical_threshold']:
                return False

            # Politique conservative pour batterie faible
            if (device.battery_level < policies['battery_management']['conservative_threshold'] and
                    not device.is_charging):
                if random.random() < 0.7:  # 70% chance de refus
                    return False

        # Vérification CPU
        if device.current_load > policies['cpu_management']['max_load_threshold']:
            return False

        # Vérification mémoire disponible
        memory_available = device.specs['memory_gb'] * (1 - device.current_load)
        if memory_available < policies['memory_management']['min_available_gb']:
            return False

        # Vérification qualité réseau
        if device.network_quality < policies['network_management']['min_quality_threshold']:
            return False

        return True

    def _check_privacy_constraints(self, device):
        """Vérifie les contraintes de confidentialité"""
        if self.privacy_level == 'low':
            return True

        # Contraintes moyennes
        if self.privacy_level == 'medium':
            # Exclure dispositifs médicaux sensibles parfois
            if device.device_type == 'medical_device' and random.random() < 0.1:
                return False

        # Contraintes élevées
        elif self.privacy_level == 'high':
            # Politiques strictes pour dispositifs sensibles
            sensitive_devices = ['medical_device', 'vehicle_ecu']
            if device.device_type in sensitive_devices:
                # Vérifications supplémentaires
                if not self._verify_secure_environment(device):
                    return False

        return True

    def _verify_secure_environment(self, device):
        """Vérifie l'environnement sécurisé pour dispositifs sensibles"""
        # Simulation de vérifications de sécurité
        security_checks = [
            device.network_quality > 0.7,  # Réseau de bonne qualité
            device.specs['reliability'] > 0.9,  # Dispositif fiable
            device.battery_level > 50 or device.is_charging,  # Alimentation stable
        ]

        # Toutes les vérifications doivent passer
        return all(security_checks)

    def _simulate_device_failure(self, device):
        """Simule les pannes de dispositifs"""
        # Taux de panne basé sur la fiabilité
        failure_rate = 1 - device.specs['reliability']

        # Facteurs aggravants
        if device.battery_level < 10:
            failure_rate *= 2

        if device.current_load > 0.9:
            failure_rate *= 1.5

        if device.network_quality < 0.3:
            failure_rate *= 1.3

        return random.random() < failure_rate

    def can_participate(self, device):
        """Détermine si un dispositif peut participer à ce round"""
        return self._check_resource_availability(device, battery_aware=True)

    def optimize_resource_allocation(self, available_devices, global_model_size):
        """Optimise l'allocation des ressources pour les dispositifs sélectionnés"""
        optimized_allocations = {}

        for device in available_devices:
            allocation = self._calculate_optimal_allocation(device, global_model_size)
            optimized_allocations[device.device_id] = allocation

        return optimized_allocations

    def _calculate_optimal_allocation(self, device, model_size):
        """Calcule l'allocation optimale pour un dispositif"""
        # Facteurs de performance
        cpu_factor = device.specs['cpu_power']
        memory_factor = min(1.0, device.specs['memory_gb'] / 2.0)
        battery_factor = device.battery_level / 100 if device.specs['battery_capacity'] != float('inf') else 1.0
        network_factor = device.network_quality

        # Score composite de performance
        performance_score = (cpu_factor * 0.4 + memory_factor * 0.3 +
                             battery_factor * 0.2 + network_factor * 0.1)

        # Allocation des ressources
        allocation = {
            'cpu_allocation': min(0.8, performance_score),
            'memory_allocation': min(device.specs['memory_gb'] * 0.7,
                                     model_size * 1.5),  # 1.5x la taille du modèle
            'training_epochs': 1 if performance_score > 0.7 else 1,
            'batch_size': self._calculate_optimal_batch_size(device),
            'communication_priority': self._calculate_priority(device),
            'energy_budget': self._calculate_energy_budget(device)
        }

        return allocation

    def _calculate_optimal_batch_size(self, device):
        """Calcule la taille de batch optimale"""
        base_batch_size = 32

        # Ajuster selon mémoire
        memory_factor = device.specs['memory_gb'] / 4.0  # Normaliser sur 4GB
        memory_batch_size = int(base_batch_size * memory_factor)

        # Ajuster selon CPU
        cpu_factor = device.specs['cpu_power']
        cpu_batch_size = int(memory_batch_size * cpu_factor)

        # Contraintes
        min_batch = 4
        max_batch = 128

        optimal_batch = max(min_batch, min(max_batch, cpu_batch_size))

        return optimal_batch

    def _calculate_priority(self, device):
        """Calcule la priorité de communication"""
        # Priorités par type
        type_priorities = {
            'medical_device': 1,
            'industrial_sensor': 2,
            'vehicle_ecu': 3,
            'smart_camera': 4,
            'raspberry_pi': 5,
            'smartphone_high': 6,
            'smartphone_low': 7,
            'tablet': 8
        }

        base_priority = type_priorities.get(device.device_type, 5)

        # Ajustements dynamiques
        if device.battery_level < 20 and not device.is_charging:
            base_priority -= 1  # Priorité plus élevée (valeur plus faible)

        if device.network_quality > 0.8:
            base_priority -= 1  # Bonne connexion = priorité

        return max(1, base_priority)

    def _calculate_energy_budget(self, device):
        """Calcule le budget énergétique alloué"""
        if device.specs['battery_capacity'] == float('inf'):
            return float('inf')  # Pas de limite pour dispositifs alimentés

        # Budget basé sur niveau de batterie et politiques
        battery_level = device.battery_level

        if device.is_charging:
            # En charge: budget généreux
            energy_budget = 100.0  # Joules
        elif battery_level > 50:
            # Batterie élevée: budget normal
            energy_budget = 50.0
        elif battery_level > 20:
            # Batterie moyenne: budget conservateur
            energy_budget = 20.0
        else:
            # Batterie faible: budget minimal
            energy_budget = 5.0

        return energy_budget

    def update_device_profile(self, device, training_metrics):
        """Met à jour le profil de performance d'un dispositif"""
        device_id = device.device_id

        if device_id not in self.device_profiles:
            self.device_profiles[device_id] = {
                'training_times': [],
                'energy_consumptions': [],
                'accuracy_contributions': [],
                'reliability_score': device.specs['reliability'],
                'performance_trend': 'stable'
            }

        profile = self.device_profiles[device_id]

        # Mettre à jour métriques
        if 'training_time' in training_metrics:
            profile['training_times'].append(training_metrics['training_time'])
            if len(profile['training_times']) > 10:
                profile['training_times'].pop(0)

        if 'energy_consumption' in training_metrics:
            profile['energy_consumptions'].append(training_metrics['energy_consumption'])
            if len(profile['energy_consumptions']) > 10:
                profile['energy_consumptions'].pop(0)

        # Analyser tendances
        self._analyze_performance_trends(profile)

    def _analyze_performance_trends(self, profile):
        """Analyse les tendances de performance d'un dispositif"""
        if len(profile['training_times']) < 3:
            return

        recent_times = profile['training_times'][-3:]
        avg_recent = np.mean(recent_times)

        if len(profile['training_times']) >= 6:
            older_times = profile['training_times'][-6:-3]
            avg_older = np.mean(older_times)

            if avg_recent < avg_older * 0.9:
                profile['performance_trend'] = 'improving'
            elif avg_recent > avg_older * 1.1:
                profile['performance_trend'] = 'degrading'
            else:
                profile['performance_trend'] = 'stable'

    def get_device_recommendations(self, device):
        """Fournit des recommandations pour optimiser un dispositif"""
        recommendations = []

        # Recommandations batterie
        if device.battery_level < 30 and not device.is_charging:
            recommendations.append({
                'type': 'battery',
                'priority': 'high',
                'message': 'Niveau de batterie faible - considérer la charge',
                'action': 'reduce_participation'
            })

        # Recommandations CPU
        if device.current_load > 0.8:
            recommendations.append({
                'type': 'cpu',
                'priority': 'medium',
                'message': 'Charge CPU élevée - réduire batch size',
                'action': 'reduce_batch_size'
            })

        # Recommandations réseau
        if device.network_quality < 0.5:
            recommendations.append({
                'type': 'network',
                'priority': 'medium',
                'message': 'Qualité réseau faible - activer compression',
                'action': 'enable_compression'
            })

        # Recommandations mobilité
        if device.is_mobile and device.velocity > 80:
            recommendations.append({
                'type': 'mobility',
                'priority': 'low',
                'message': 'Vitesse élevée - possible dégradation réseau',
                'action': 'monitor_network'
            })

        return recommendations

    def _log_selection_results(self, all_devices, available_devices):
        """Enregistre les résultats de sélection pour analyse"""
        selection_result = {
            'total_devices': len(all_devices),
            'available_devices': len(available_devices),
            'selection_rate': len(available_devices) / max(len(all_devices), 1),
            'excluded_reasons': self._analyze_exclusion_reasons(all_devices, available_devices)
        }

        self.resource_history.append(selection_result)

        # Garder seulement les 100 derniers résultats
        if len(self.resource_history) > 100:
            self.resource_history.pop(0)

    def _analyze_exclusion_reasons(self, all_devices, available_devices):
        """Analyse les raisons d'exclusion des dispositifs"""
        available_ids = {d.device_id for d in available_devices}
        excluded_devices = [d for d in all_devices if d.device_id not in available_ids]

        reasons = {
            'offline': 0,
            'battery_low': 0,
            'cpu_overload': 0,
            'network_poor': 0,
            'privacy_constraints': 0,
            'memory_insufficient': 0
        }

        for device in excluded_devices:
            if not device.is_online:
                reasons['offline'] += 1
            elif device.battery_level < self.resource_policies['battery_management']['critical_threshold']:
                reasons['battery_low'] += 1
            elif device.current_load > self.resource_policies['cpu_management']['max_load_threshold']:
                reasons['cpu_overload'] += 1
            elif device.network_quality < self.resource_policies['network_management']['min_quality_threshold']:
                reasons['network_poor'] += 1
            else:
                reasons['privacy_constraints'] += 1

        return reasons

    def get_resource_statistics(self):
        """Retourne les statistiques de gestion des ressources"""
        if not self.resource_history:
            return {}

        recent_history = self.resource_history[-20:]  # 20 derniers rounds

        avg_selection_rate = np.mean([h['selection_rate'] for h in recent_history])

        # Agrégation des raisons d'exclusion
        total_exclusions = {}
        for history in recent_history:
            for reason, count in history['excluded_reasons'].items():
                total_exclusions[reason] = total_exclusions.get(reason, 0) + count

        return {
            'average_selection_rate': avg_selection_rate,
            'total_exclusion_reasons': total_exclusions,
            'device_profiles_count': len(self.device_profiles),
            'privacy_level': self.privacy_level,
            'current_policies': self.resource_policies
        }

    def adapt_policies(self, performance_feedback):
        """Adapte les politiques selon les retours de performance"""

        # Adapter seuils de batterie
        if performance_feedback.get('low_participation', False):
            # Participation faible: assouplir critères batterie
            current_threshold = self.resource_policies['battery_management']['critical_threshold']
            new_threshold = max(10.0, current_threshold - 2.0)
            self.resource_policies['battery_management']['critical_threshold'] = new_threshold

        elif performance_feedback.get('high_energy_consumption', False):
            # Consommation élevée: durcir critères batterie
            current_threshold = self.resource_policies['battery_management']['critical_threshold']
            new_threshold = min(25.0, current_threshold + 2.0)
            self.resource_policies['battery_management']['critical_threshold'] = new_threshold

        # Adapter seuils CPU
        avg_cpu_load = performance_feedback.get('average_cpu_load', 0.5)
        if avg_cpu_load > 0.8:
            # Charge CPU élevée: réduire seuil max
            self.resource_policies['cpu_management']['max_load_threshold'] = min(
                0.80, self.resource_policies['cpu_management']['max_load_threshold'] - 0.05
            )
        elif avg_cpu_load < 0.4:
            # Charge CPU faible: augmenter seuil max
            self.resource_policies['cpu_management']['max_load_threshold'] = min(
                0.90, self.resource_policies['cpu_management']['max_load_threshold'] + 0.05
            )

        # Adapter seuils réseau
        avg_network_quality = performance_feedback.get('average_network_quality', 0.7)
        if avg_network_quality < 0.5:
            # Réseau de mauvaise qualité: assouplir seuil
            self.resource_policies['network_management']['min_quality_threshold'] = max(
                0.2, self.resource_policies['network_management']['min_quality_threshold'] - 0.05
            )


class EnergyManager:
    """Gestionnaire spécialisé pour l'efficacité énergétique"""

    def __init__(self):
        self.energy_strategies = {
            'aggressive': {'cpu_scaling': 0.6, 'network_optimization': True},
            'balanced': {'cpu_scaling': 0.8, 'network_optimization': True},
            'performance': {'cpu_scaling': 1.0, 'network_optimization': False}
        }

        self.device_energy_profiles = {}

    def select_energy_strategy(self, device):
        """Sélectionne la stratégie énergétique pour un dispositif"""

        if device.specs['battery_capacity'] == float('inf'):
            return 'performance'  # Dispositif alimenté

        battery_level = device.battery_level

        if battery_level < 20:
            return 'aggressive'
        elif battery_level < 50:
            return 'balanced'
        else:
            return 'performance'

    def estimate_training_energy(self, device, model_size, strategy):
        """Estime la consommation énergétique d'un entraînement"""

        strategy_params = self.energy_strategies[strategy]

        # Facteurs de base
        base_energy = model_size * 0.001  # 1mJ par paramètre
        cpu_energy = device.specs['cpu_power'] * 10  # 10J par unité CPU

        # Ajustements stratégiques
        cpu_scaling = strategy_params['cpu_scaling']
        adjusted_cpu_energy = cpu_energy * cpu_scaling

        # Énergie réseau
        if strategy_params['network_optimization']:
            network_energy = 5.0  # Énergie optimisée
        else:
            network_energy = 10.0  # Énergie standard

        total_energy = base_energy + adjusted_cpu_energy + network_energy

        return total_energy

    def optimize_device_for_energy(self, device, strategy):
        """Optimise un dispositif pour l'efficacité énergétique"""

        strategy_params = self.energy_strategies[strategy]

        optimizations = {
            'cpu_frequency_scaling': strategy_params['cpu_scaling'],
            'network_compression': strategy_params['network_optimization'],
            'reduced_precision': strategy == 'aggressive',
            'adaptive_batch_size': True,
            'early_stopping_threshold': 0.05 if strategy == 'aggressive' else 0.01
        }

        return optimizations

    def track_energy_consumption(self, device, actual_consumption):
        """Suit la consommation énergétique réelle"""

        device_id = device.device_id

        if device_id not in self.device_energy_profiles:
            self.device_energy_profiles[device_id] = {
                'consumption_history': [],
                'efficiency_trend': 'stable',
                'total_consumption': 0
            }

        profile = self.device_energy_profiles[device_id]
        profile['consumption_history'].append(actual_consumption)
        profile['total_consumption'] += actual_consumption

        # Garder seulement les 20 dernières mesures
        if len(profile['consumption_history']) > 20:
            profile['consumption_history'].pop(0)

        # Analyser tendance d'efficacité
        self._analyze_energy_efficiency(profile)

    def _analyze_energy_efficiency(self, profile):
        """Analyse l'efficacité énergétique d'un dispositif"""

        history = profile['consumption_history']
        if len(history) < 5:
            return

        recent_avg = np.mean(history[-3:])
        older_avg = np.mean(history[-6:-3]) if len(history) >= 6 else recent_avg

        if recent_avg < older_avg * 0.95:
            profile['efficiency_trend'] = 'improving'
        elif recent_avg > older_avg * 1.05:
            profile['efficiency_trend'] = 'degrading'
        else:
            profile['efficiency_trend'] = 'stable'


class PrivacyManager:
    """Gestionnaire de confidentialité pour edge FL"""

    def __init__(self, privacy_level='medium'):
        self.privacy_level = privacy_level
        self.privacy_techniques = self._initialize_privacy_techniques()
        self.privacy_budget = self._initialize_privacy_budget()

    def _initialize_privacy_techniques(self):
        """Initialise les techniques de confidentialité"""
        techniques = {
            'low': {
                'differential_privacy': False,
                'secure_aggregation': False,
                'local_noise': False,
                'gradient_clipping': False
            },
            'medium': {
                'differential_privacy': True,
                'secure_aggregation': False,
                'local_noise': True,
                'gradient_clipping': True,
                'epsilon': 10.0
            },
            'high': {
                'differential_privacy': True,
                'secure_aggregation': True,
                'local_noise': True,
                'gradient_clipping': True,
                'epsilon': 1.0,
                'homomorphic_encryption': True
            }
        }

        return techniques[self.privacy_level]

    def _initialize_privacy_budget(self):
        """Initialise le budget de confidentialité"""
        if self.privacy_level == 'low':
            return {'epsilon': float('inf'), 'delta': 0}
        elif self.privacy_level == 'medium':
            return {'epsilon': 10.0, 'delta': 1e-5}
        else:  # high
            return {'epsilon': 1.0, 'delta': 1e-6}

    def apply_privacy_protection(self, device_update, device):
        """Applique les protections de confidentialité"""

        protected_update = device_update
        privacy_cost = 0

        # Clipping des gradients
        if self.privacy_techniques.get('gradient_clipping', False):
            protected_update = self._clip_gradients(protected_update)
            privacy_cost += 0.1

        # Ajout de bruit local
        if self.privacy_techniques.get('local_noise', False):
            protected_update = self._add_local_noise(protected_update)
            privacy_cost += 0.2

        # Confidentialité différentielle
        if self.privacy_techniques.get('differential_privacy', False):
            protected_update = self._apply_differential_privacy(protected_update)
            privacy_cost += 0.5

        return protected_update, privacy_cost

    def _clip_gradients(self, update, clip_norm=1.0):
        """Applique le clipping des gradients"""
        clipped_update = []

        for layer_update in update:
            if hasattr(layer_update, 'flatten'):
                # Calculer norme L2
                norm = np.linalg.norm(layer_update)

                if norm > clip_norm:
                    # Clipper
                    clipped_layer = layer_update * (clip_norm / norm)
                    clipped_update.append(clipped_layer)
                else:
                    clipped_update.append(layer_update)
            else:
                clipped_update.append(layer_update)

        return clipped_update

    def _add_local_noise(self, update, noise_scale=0.1):
        """Ajoute du bruit local pour protection"""
        noisy_update = []

        for layer_update in update:
            if hasattr(layer_update, 'shape'):
                # Bruit gaussien
                noise = np.random.normal(0, noise_scale, layer_update.shape)
                noisy_layer = layer_update + noise
                noisy_update.append(noisy_layer)
            else:
                noisy_update.append(layer_update)

        return noisy_update

    def _apply_differential_privacy(self, update):
        """Applique la confidentialité différentielle"""
        epsilon = self.privacy_techniques.get('epsilon', 1.0)

        # Calcul du bruit nécessaire pour DP
        sensitivity = 1.0  # Sensibilité du mécanisme
        noise_scale = sensitivity / epsilon

        dp_update = []
        for layer_update in update:
            if hasattr(layer_update, 'shape'):
                # Mécanisme de Laplace pour DP
                noise = np.random.laplace(0, noise_scale, layer_update.shape)
                dp_layer = layer_update + noise
                dp_update.append(dp_layer)
            else:
                dp_update.append(layer_update)

        return dp_update

    def verify_privacy_compliance(self, device, data_sensitivity):
        """Vérifie la conformité aux exigences de confidentialité"""

        compliance_checks = {
            'device_type_allowed': self._check_device_type_privacy(device),
            'data_sensitivity_level': data_sensitivity,
            'privacy_budget_available': self._check_privacy_budget(),
            'secure_communication': self._check_secure_communication(device)
        }

        # Toutes les vérifications doivent passer pour niveau high
        if self.privacy_level == 'high':
            return all(compliance_checks.values())
        elif self.privacy_level == 'medium':
            # Au moins 75% des vérifications
            return sum(compliance_checks.values()) >= 0.75 * len(compliance_checks)
        else:
            # Niveau low: pas de contraintes strictes
            return True

    def _check_device_type_privacy(self, device):
        """Vérifie si le type de dispositif est autorisé"""

        if self.privacy_level == 'high':
            # Niveau élevé: seulement dispositifs sécurisés
            secure_devices = ['medical_device', 'industrial_sensor', 'raspberry_pi']
            return device.device_type in secure_devices

        elif self.privacy_level == 'medium':
            # Niveau moyen: exclure dispositifs personnels non sécurisés
            excluded_devices = ['smartphone_low']
            return device.device_type not in excluded_devices

        else:
            # Niveau faible: tous dispositifs autorisés
            return True

    def _check_privacy_budget(self):
        """Vérifie si le budget de confidentialité est disponible"""

        if self.privacy_level == 'low':
            return True

        # Vérifier si epsilon budget disponible
        return self.privacy_budget['epsilon'] > 0.1

    def _check_secure_communication(self, device):
        """Vérifie la sécurité de la communication"""

        # Vérifier qualité réseau pour communication sécurisée
        if device.network_quality < 0.5:
            return False

        # Vérifier technologies réseau sécurisées
        secure_networks = ['wifi', '5g', 'ethernet']
        device_networks = device.specs.get('network_capability', [])

        return any(net in secure_networks for net in device_networks)

    def consume_privacy_budget(self, epsilon_used):
        """Consomme le budget de confidentialité"""

        if self.privacy_level != 'low':
            self.privacy_budget['epsilon'] -= epsilon_used
            self.privacy_budget['epsilon'] = max(0, self.privacy_budget['epsilon'])

    def get_privacy_status(self):
        """Retourne l'état de la confidentialité"""

        return {
            'privacy_level': self.privacy_level,
            'techniques_enabled': self.privacy_techniques,
            'privacy_budget_remaining': self.privacy_budget,
            'compliance_level': 'strict' if self.privacy_level == 'high' else 'moderate' if self.privacy_level == 'medium' else 'basic'
        }