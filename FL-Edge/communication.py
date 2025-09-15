"""
Simulation de la communication réseau pour dispositifs edge
"""
import numpy as np
import random
import time
from typing import List, Tuple, Dict, Any


class NetworkSimulator:
    """Simulateur de réseau pour environnement edge"""

    def __init__(self, network_type='mixed', mobility_level='static'):
        self.network_type = network_type
        self.mobility_level = mobility_level

        # Caractéristiques des types de réseau
        self.network_specs = {
            'wifi': {
                'bandwidth_mbps': (20, 100),
                'latency_ms': (1, 10),
                'reliability': 0.95,
                'range_km': 0.1,
                'energy_cost_factor': 0.5
            },
            '4g': {
                'bandwidth_mbps': (5, 50),
                'latency_ms': (10, 50),
                'reliability': 0.90,
                'range_km': 10,
                'energy_cost_factor': 1.0
            },
            '5g': {
                'bandwidth_mbps': (50, 1000),
                'latency_ms': (1, 5),
                'reliability': 0.95,
                'range_km': 1,
                'energy_cost_factor': 1.2
            },
            'ethernet': {
                'bandwidth_mbps': (100, 1000),
                'latency_ms': (0.1, 2),
                'reliability': 0.99,
                'range_km': 0.001,  # Très courte portée
                'energy_cost_factor': 0.3
            }
        }

        # État du réseau
        self.current_conditions = self._initialize_network_conditions()
        self.communication_history = []

        print(f"📡 Simulateur réseau initialisé: {network_type} (mobilité: {mobility_level})")

    def _initialize_network_conditions(self):
        """Initialise les conditions réseau"""
        return {
            'global_congestion': 0.0,
            'interference_level': 0.0,
            'weather_impact': 1.0,
            'peak_hour_factor': 1.0
        }

    def simulate_communication(self, device_updates, compression_enabled=False):
        """
        Simule la communication des mises à jour FL

        Args:
            device_updates: Liste des mises à jour (device_id, update)
            compression_enabled: Si la compression est activée

        Returns:
            Tuple[float, float]: (coût communication, latence totale)
        """
        total_comm_cost = 0
        total_latency = 0

        for device_id, update in device_updates:
            # Calculer taille des données
            data_size_mb = self._calculate_update_size(update, compression_enabled)

            # Sélectionner type de réseau pour ce dispositif
            network_type = self._select_network_type(device_id)

            # Simuler transmission
            transmission_time, latency, energy_cost = self._simulate_transmission(
                data_size_mb, network_type, device_id
            )

            total_comm_cost += energy_cost
            total_latency += latency

            # Enregistrer dans l'historique
            self.communication_history.append({
                'device_id': device_id,
                'data_size_mb': data_size_mb,
                'network_type': network_type,
                'transmission_time': transmission_time,
                'latency': latency,
                'energy_cost': energy_cost,
                'timestamp': time.time()
            })

        # Garder seulement les 1000 dernières communications
        if len(self.communication_history) > 1000:
            self.communication_history = self.communication_history[-1000:]

        avg_latency = total_latency / len(device_updates) if device_updates else 0

        return total_comm_cost, avg_latency

    def _calculate_update_size(self, update, compression_enabled):
        """Calcule la taille de la mise à jour en MB"""
        # Estimer taille basée sur les paramètres du modèle
        total_params = 0

        if isinstance(update, list):
            for layer_update in update:
                if hasattr(layer_update, 'size'):
                    total_params += layer_update.size
                elif hasattr(layer_update, 'shape'):
                    total_params += np.prod(layer_update.shape)

        # 4 bytes par paramètre (float32)
        size_bytes = total_params * 4

        # Compression
        if compression_enabled:
            compression_ratio = random.uniform(0.1, 0.3)  # 10-30% de la taille originale
            size_bytes *= compression_ratio

        # Ajouter overhead protocole
        overhead = size_bytes * 0.1  # 10% d'overhead
        total_size_bytes = size_bytes + overhead

        # Convertir en MB
        size_mb = total_size_bytes / (1024 * 1024)

        return max(size_mb, 0.001)  # Minimum 1KB

    def _select_network_type(self, device_id):
        """Sélectionne le type de réseau pour un dispositif"""
        if self.network_type == 'mixed':
            # Distribution réaliste des types de réseau
            network_distribution = {
                'wifi': 0.4,
                '4g': 0.4,
                '5g': 0.15,
                'ethernet': 0.05
            }

            networks = list(network_distribution.keys())
            probabilities = list(network_distribution.values())

            return np.random.choice(networks, p=probabilities)
        else:
            return self.network_type

    def _simulate_transmission(self, data_size_mb, network_type, device_id):
        """Simule la transmission des données"""
        specs = self.network_specs[network_type]

        # Conditions actuelles du réseau
        conditions = self._get_current_network_conditions(network_type)

        # Bande passante effective
        base_bandwidth = random.uniform(*specs['bandwidth_mbps'])
        effective_bandwidth = base_bandwidth * conditions['quality_factor']

        # Temps de transmission
        transmission_time = (data_size_mb * 8) / effective_bandwidth  # Secondes

        # Latence
        base_latency = random.uniform(*specs['latency_ms'])
        congestion_latency = base_latency * (1 + conditions['congestion'])
        mobility_latency = self._calculate_mobility_latency(network_type)

        total_latency = congestion_latency + mobility_latency

        # Coût énergétique
        energy_cost = self._calculate_energy_cost(
            data_size_mb, transmission_time, specs['energy_cost_factor']
        )

        # Simuler échecs de transmission
        if self._should_transmission_fail(specs['reliability'], conditions):
            # Retransmission avec pénalité
            transmission_time *= 2
            total_latency *= 1.5
            energy_cost *= 1.3

        return transmission_time, total_latency, energy_cost

    def _get_current_network_conditions(self, network_type):
        """Obtient les conditions actuelles du réseau"""
        base_quality = 1.0

        # Impact de la congestion
        congestion_impact = 1 - (self.current_conditions['global_congestion'] * 0.5)

        # Impact des interférences
        interference_impact = 1 - (self.current_conditions['interference_level'] * 0.3)

        # Impact météorologique (plus fort pour 5G)
        weather_impact = self.current_conditions['weather_impact']
        if network_type == '5g':
            weather_impact = min(weather_impact, 0.8)  # 5G plus sensible

        # Facteur heure de pointe
        peak_impact = 1 / self.current_conditions['peak_hour_factor']

        quality_factor = (base_quality * congestion_impact *
                          interference_impact * weather_impact * peak_impact)

        return {
            'quality_factor': max(quality_factor, 0.1),  # Minimum 10%
            'congestion': self.current_conditions['global_congestion']
        }

    def _calculate_mobility_latency(self, network_type):
        """Calcule la latence additionnelle due à la mobilité"""
        mobility_factors = {
            'static': 0,
            'low': 2,
            'medium': 5,
            'high': 10
        }

        base_mobility_latency = mobility_factors[self.mobility_level]

        # 5G plus sensible aux handovers
        if network_type == '5g' and self.mobility_level in ['medium', 'high']:
            base_mobility_latency *= 1.5

        # WiFi problématique en mobilité
        if network_type == 'wifi' and self.mobility_level in ['medium', 'high']:
            base_mobility_latency *= 2

        return base_mobility_latency

    def _calculate_energy_cost(self, data_size_mb, transmission_time, energy_factor):
        """Calcule le coût énergétique de la transmission"""
        # Coût de base proportionnel à la taille et au temps
        base_cost = data_size_mb * 0.5 + transmission_time * 0.3

        # Facteur spécifique au type de réseau
        energy_cost = base_cost * energy_factor

        return energy_cost

    def _should_transmission_fail(self, base_reliability, conditions):
        """Détermine si la transmission échoue"""
        effective_reliability = base_reliability * conditions['quality_factor']
        return random.random() > effective_reliability

    def update_network_conditions(self, current_time, devices):
        """Met à jour les conditions réseau"""
        # Congestion basée sur l'heure
        hour_of_day = (current_time % 1440) / 60  # 0-24h

        # Pics de congestion aux heures de pointe
        if 8 <= hour_of_day <= 10 or 17 <= hour_of_day <= 19:
            self.current_conditions['peak_hour_factor'] = 1.5
            self.current_conditions['global_congestion'] = 0.7
        elif 12 <= hour_of_day <= 14:  # Pause déjeuner
            self.current_conditions['peak_hour_factor'] = 1.2
            self.current_conditions['global_congestion'] = 0.4
        else:
            self.current_conditions['peak_hour_factor'] = 1.0
            self.current_conditions['global_congestion'] = 0.2

        # Interférences basées sur la densité de dispositifs
        if devices:
            active_devices = [d for d in devices if d.is_online]
            device_density = len(active_devices) / max(len(devices), 1)
            self.current_conditions['interference_level'] = device_density * 0.3

        # Variations aléatoires
        self.current_conditions['global_congestion'] += random.uniform(-0.1, 0.1)
        self.current_conditions['global_congestion'] = np.clip(
            self.current_conditions['global_congestion'], 0.0, 1.0
        )

    def set_weather_impact(self, weather_condition):
        """Définit l'impact météorologique sur le réseau"""
        weather_impacts = {
            'clear': 1.0,
            'cloudy': 0.95,
            'rainy': 0.8,
            'stormy': 0.5
        }

        self.current_conditions['weather_impact'] = weather_impacts.get(weather_condition, 1.0)

    def get_network_statistics(self):
        """Retourne les statistiques réseau"""
        if not self.communication_history:
            return {}

        recent_comms = self.communication_history[-100:]  # 100 dernières communications

        avg_latency = np.mean([c['latency'] for c in recent_comms])
        avg_data_size = np.mean([c['data_size_mb'] for c in recent_comms])
        avg_energy_cost = np.mean([c['energy_cost'] for c in recent_comms])

        network_usage = {}
        for comm in recent_comms:
            net_type = comm['network_type']
            network_usage[net_type] = network_usage.get(net_type, 0) + 1

        return {
            'avg_latency_ms': avg_latency,
            'avg_data_size_mb': avg_data_size,
            'avg_energy_cost': avg_energy_cost,
            'network_type_usage': network_usage,
            'total_communications': len(recent_comms),
            'current_conditions': self.current_conditions.copy()
        }


class EdgeCommunicationProtocol:
    """Protocole de communication spécialisé pour edge"""

    def __init__(self, protocol_type='adaptive'):
        self.protocol_type = protocol_type
        self.adaptive_parameters = {
            'compression_threshold': 1.0,  # MB
            'retry_limit': 3,
            'timeout_ms': 5000,
            'batch_size_limit': 10
        }

        self.communication_stats = {
            'successful_transmissions': 0,
            'failed_transmissions': 0,
            'retransmissions': 0,
            'bytes_transmitted': 0
        }

    def prepare_transmission(self, device_updates, network_conditions):
        """Prépare la transmission selon les conditions"""
        if self.protocol_type == 'adaptive':
            return self._adaptive_preparation(device_updates, network_conditions)
        elif self.protocol_type == 'batch':
            return self._batch_preparation(device_updates)
        elif self.protocol_type == 'priority':
            return self._priority_preparation(device_updates)
        else:
            return device_updates  # Standard

    def _adaptive_preparation(self, device_updates, network_conditions):
        """Préparation adaptative selon conditions réseau"""
        prepared_updates = []

        # Adapter selon qualité réseau
        quality_factor = network_conditions.get('quality_factor', 1.0)

        for device_id, update in device_updates:
            # Compression dynamique
            if quality_factor < 0.5:  # Réseau de mauvaise qualité
                compressed_update = self._compress_update(update, ratio=0.1)
            elif quality_factor < 0.8:  # Réseau moyen
                compressed_update = self._compress_update(update, ratio=0.3)
            else:  # Bon réseau
                compressed_update = update

            prepared_updates.append((device_id, compressed_update))

        return prepared_updates

    def _batch_preparation(self, device_updates):
        """Préparation par lots pour optimiser bande passante"""
        if len(device_updates) <= self.adaptive_parameters['batch_size_limit']:
            return device_updates

        # Diviser en lots
        batches = []
        batch_size = self.adaptive_parameters['batch_size_limit']

        for i in range(0, len(device_updates), batch_size):
            batch = device_updates[i:i + batch_size]
            batches.append(batch)

        return batches

    def _priority_preparation(self, device_updates):
        """Préparation avec priorités selon type de dispositif"""
        # Priorités par type
        priorities = {
            'medical_device': 1,
            'industrial_sensor': 2,
            'vehicle_ecu': 3,
            'smart_camera': 4,
            'smartphone_high': 5,
            'smartphone_low': 6,
            'tablet': 7,
            'raspberry_pi': 8
        }

        # Trier par priorité (pas d'accès direct au type, simulation)
        sorted_updates = sorted(device_updates,
                                key=lambda x: priorities.get(f'type_{hash(x[0]) % 8}', 9))

        return sorted_updates

    def _compress_update(self, update, ratio=0.3):
        """Compresse une mise à jour modèle"""
        if not isinstance(update, list):
            return update

        compressed_update = []
        for layer_update in update:
            if hasattr(layer_update, 'flatten'):
                # Sparsification: garder seulement les valeurs importantes
                flat_update = layer_update.flatten()
                threshold = np.percentile(np.abs(flat_update), (1 - ratio) * 100)

                # Masque pour valeurs importantes
                mask = np.abs(flat_update) >= threshold
                sparse_update = flat_update * mask

                compressed_update.append(sparse_update.reshape(layer_update.shape))
            else:
                compressed_update.append(layer_update)

        return compressed_update

    def handle_transmission_failure(self, device_id, update, failure_reason):
        """Gère les échecs de transmission"""
        self.communication_stats['failed_transmissions'] += 1

        retry_strategies = {
            'network_congestion': 'delay_retry',
            'signal_weak': 'compress_retry',
            'timeout': 'fast_retry',
            'device_offline': 'skip'
        }

        strategy = retry_strategies.get(failure_reason, 'delay_retry')

        if strategy == 'delay_retry':
            # Attendre avant retry
            return {'action': 'retry', 'delay': 30}  # 30 secondes
        elif strategy == 'compress_retry':
            # Retry avec compression agressive
            compressed = self._compress_update(update, ratio=0.1)
            return {'action': 'retry', 'update': compressed, 'delay': 5}
        elif strategy == 'fast_retry':
            # Retry immédiat
            return {'action': 'retry', 'delay': 0}
        else:  # skip
            return {'action': 'skip'}

    def update_adaptive_parameters(self, network_stats):
        """Met à jour les paramètres adaptatifs selon performance"""
        avg_latency = network_stats.get('avg_latency_ms', 0)

        # Ajuster seuil de compression selon latence
        if avg_latency > 100:  # Latence élevée
            self.adaptive_parameters['compression_threshold'] = 0.5
        elif avg_latency > 50:  # Latence moyenne
            self.adaptive_parameters['compression_threshold'] = 1.0
        else:  # Faible latence
            self.adaptive_parameters['compression_threshold'] = 2.0

        # Ajuster timeout selon conditions
        congestion = network_stats.get('current_conditions', {}).get('global_congestion', 0)
        base_timeout = 5000
        self.adaptive_parameters['timeout_ms'] = base_timeout * (1 + congestion)


class SecureCommunication:
    """Module de communication sécurisée pour edge"""

    def __init__(self, security_level='medium'):
        self.security_level = security_level
        self.encryption_overhead = self._get_encryption_overhead()
        self.authentication_enabled = security_level in ['medium', 'high']
        self.integrity_check_enabled = security_level == 'high'

    def _get_encryption_overhead(self):
        """Calcule l'overhead d'encryption selon niveau"""
        overheads = {
            'low': 1.05,  # 5% d'overhead
            'medium': 1.15,  # 15% d'overhead
            'high': 1.25  # 25% d'overhead
        }
        return overheads[self.security_level]

    def encrypt_update(self, update, device_id):
        """Chiffre une mise à jour"""
        # Simulation du chiffrement
        encrypted_size = self._calculate_encrypted_size(update)
        encryption_time = self._calculate_encryption_time(update)

        return {
            'encrypted_update': update,  # Simulation
            'size_overhead': encrypted_size,
            'processing_time': encryption_time,
            'device_id': device_id
        }

    def _calculate_encrypted_size(self, update):
        """Calcule la taille après chiffrement"""
        if isinstance(update, list):
            total_size = sum(u.nbytes if hasattr(u, 'nbytes') else 1000 for u in update)
        else:
            total_size = 1000  # Taille par défaut

        return total_size * self.encryption_overhead

    def _calculate_encryption_time(self, update):
        """Calcule le temps de chiffrement"""
        if isinstance(update, list):
            total_params = sum(u.size if hasattr(u, 'size') else 1000 for u in update)
        else:
            total_params = 1000

        # Temps proportionnel au nombre de paramètres
        base_time = total_params / 1000000  # 1 seconde pour 1M paramètres

        security_factors = {'low': 1.0, 'medium': 1.5, 'high': 2.0}
        return base_time * security_factors[self.security_level]

    def verify_integrity(self, received_update, expected_hash):
        """Vérifie l'intégrité des données reçues"""
        if not self.integrity_check_enabled:
            return True

        # Simulation de vérification d'intégrité
        verification_success = random.random() > 0.01  # 99% de succès
        return verification_success


class QoSManager:
    """Gestionnaire de Qualité de Service pour communications edge"""

    def __init__(self):
        self.service_classes = {
            'critical': {'priority': 1, 'max_latency': 10, 'min_bandwidth': 10},
            'important': {'priority': 2, 'max_latency': 50, 'min_bandwidth': 5},
            'normal': {'priority': 3, 'max_latency': 200, 'min_bandwidth': 1},
            'background': {'priority': 4, 'max_latency': 1000, 'min_bandwidth': 0.5}
        }

        self.device_class_mapping = {
            'medical_device': 'critical',
            'industrial_sensor': 'important',
            'vehicle_ecu': 'important',
            'smart_camera': 'normal',
            'smartphone_high': 'normal',
            'smartphone_low': 'background',
            'tablet': 'normal',
            'raspberry_pi': 'background'
        }

    def classify_transmission(self, device_id, device_type, update_size):
        """Classifie une transmission selon QoS"""
        service_class = self.device_class_mapping.get(device_type, 'normal')
        qos_params = self.service_classes[service_class]

        return {
            'service_class': service_class,
            'priority': qos_params['priority'],
            'max_latency': qos_params['max_latency'],
            'min_bandwidth': qos_params['min_bandwidth'],
            'device_id': device_id
        }

    def schedule_transmissions(self, transmission_queue):
        """Ordonnance les transmissions selon QoS"""
        # Trier par priorité (1 = plus haute priorité)
        sorted_queue = sorted(transmission_queue,
                              key=lambda x: x.get('priority', 3))

        return sorted_queue

    def allocate_bandwidth(self, transmissions, total_bandwidth):
        """Alloue la bande passante selon QoS"""
        allocations = {}
        remaining_bandwidth = total_bandwidth

        # Allocation par priorité
        for transmission in sorted(transmissions, key=lambda x: x.get('priority', 3)):
            device_id = transmission['device_id']
            min_bandwidth = transmission.get('min_bandwidth', 1)

            allocated = min(min_bandwidth, remaining_bandwidth)
            allocations[device_id] = allocated
            remaining_bandwidth -= allocated

            if remaining_bandwidth <= 0:
                break

        # Distribuer la bande passante restante
        if remaining_bandwidth > 0 and allocations:
            bonus_per_device = remaining_bandwidth / len(allocations)
            for device_id in allocations:
                allocations[device_id] += bonus_per_device

        return allocations