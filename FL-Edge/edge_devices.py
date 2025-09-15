"""
Simulation des dispositifs edge pour apprentissage fédéré
"""
import numpy as np
import tensorflow as tf
import random
import time
from models import create_edge_model

# Spécifications des types de dispositifs edge
EDGE_DEVICE_SPECS = {
    'smartphone_low': {
        'cpu_power': 0.3,  # Puissance relative CPU (0-1)
        'memory_gb': 2,  # RAM en GB
        'storage_gb': 32,  # Stockage en GB
        'battery_capacity': 3000,  # mAh
        'network_capability': ['wifi', '4g'],
        'typical_usage': 'personal',
        'reliability': 0.7
    },
    'smartphone_high': {
        'cpu_power': 0.8,
        'memory_gb': 8,
        'storage_gb': 128,
        'battery_capacity': 4500,
        'network_capability': ['wifi', '4g', '5g'],
        'typical_usage': 'personal',
        'reliability': 0.9
    },
    'tablet': {
        'cpu_power': 0.6,
        'memory_gb': 4,
        'storage_gb': 64,
        'battery_capacity': 7000,
        'network_capability': ['wifi', '4g'],
        'typical_usage': 'personal',
        'reliability': 0.8
    },
    'raspberry_pi': {
        'cpu_power': 0.4,
        'memory_gb': 4,
        'storage_gb': 32,
        'battery_capacity': float('inf'),  # Alimenté
        'network_capability': ['wifi', 'ethernet'],
        'typical_usage': 'iot_gateway',
        'reliability': 0.95
    },
    'smart_camera': {
        'cpu_power': 0.2,
        'memory_gb': 1,
        'storage_gb': 16,
        'battery_capacity': 2000,
        'network_capability': ['wifi'],
        'typical_usage': 'surveillance',
        'reliability': 0.85
    },
    'industrial_sensor': {
        'cpu_power': 0.15,
        'memory_gb': 0.5,
        'storage_gb': 8,
        'battery_capacity': 1000,
        'network_capability': ['wifi', 'ethernet'],
        'typical_usage': 'monitoring',
        'reliability': 0.99
    },
    'vehicle_ecu': {
        'cpu_power': 0.5,
        'memory_gb': 2,
        'storage_gb': 32,
        'battery_capacity': float('inf'),  # Alimenté par véhicule
        'network_capability': ['4g', '5g', 'dsrc'],
        'typical_usage': 'automotive',
        'reliability': 0.92
    },
    'medical_device': {
        'cpu_power': 0.3,
        'memory_gb': 1,
        'storage_gb': 16,
        'battery_capacity': 1500,
        'network_capability': ['wifi', 'bluetooth'],
        'typical_usage': 'healthcare',
        'reliability': 0.98
    }
}


class EdgeDevice:
    """Classe représentant un dispositif edge"""

    def __init__(self, device_id, device_type, location=None):
        self.device_id = device_id
        self.device_type = device_type
        self.specs = EDGE_DEVICE_SPECS[device_type].copy()

        # État dynamique
        self.location = location or self._generate_random_location()
        self.battery_level = random.uniform(60, 100)  # %
        self.is_charging = random.choice([True, False])
        self.network_quality = random.uniform(0.3, 1.0)
        self.current_load = random.uniform(0.1, 0.4)  # Charge CPU actuelle

        # Historique et données
        self.training_data = None
        self.local_model = None
        self.training_history = []
        self.energy_history = []

        # États de mobilité
        self.velocity = 0.0  # km/h
        self.is_mobile = device_type in ['smartphone_low', 'smartphone_high', 'tablet', 'vehicle_ecu']

        # État de disponibilité
        self.is_online = True
        self.last_communication = time.time()

    def _generate_random_location(self):
        """Génère une localisation aléatoire"""
        return {
            'lat': random.uniform(-90, 90),
            'lon': random.uniform(-180, 180),
            'zone': random.choice(['urban', 'suburban', 'rural'])
        }

    def set_training_data(self, x_data, y_data):
        """Définit les données d'entraînement local"""
        self.training_data = (x_data, y_data)

    def estimate_training_time(self, epochs=1):
        """Estime le temps d'entraînement basé sur les specs"""
        if self.training_data is None:
            return 0

        data_size = len(self.training_data[0])

        # Facteurs influençant le temps
        cpu_factor = 1 / max(self.specs['cpu_power'], 0.1)
        memory_factor = max(1, data_size / (self.specs['memory_gb'] * 1000))
        load_factor = 1 + self.current_load

        # Temps de base (secondes) pour l'entraînement
        base_time = epochs * data_size * 0.001  # 1ms par échantillon de base
        estimated_time = base_time * cpu_factor * memory_factor * load_factor

        return max(1.0, estimated_time)

    def estimate_energy_consumption(self, training_time):
        """Estime la consommation énergétique"""
        if self.specs['battery_capacity'] == float('inf'):
            return 0  # Dispositif alimenté

        # Consommation de base + consommation training
        base_power = 2.0  # Watts de base
        training_power = self.specs['cpu_power'] * 5.0  # Watts pendant training
        communication_power = 1.5  # Watts pour communication

        # Énergie en Joules
        training_energy = (base_power + training_power) * training_time
        comm_energy = communication_power * 2  # 2 secondes de communication

        total_energy = training_energy + comm_energy
        return total_energy

    def can_participate(self, battery_threshold=20.0, min_network_quality=0.3):
        """Vérifie si le dispositif peut participer à ce round"""
        # Vérifications de base
        if not self.is_online:
            return False

        if self.battery_level < battery_threshold and not self.is_charging:
            return False

        if self.network_quality < min_network_quality:
            return False

        # Vérification surcharge CPU
        if self.current_load > 0.8:
            return False

        # Vérification mémoire disponible
        required_memory = 0.5  # GB pour FL
        if self.specs['memory_gb'] * (1 - self.current_load) < required_memory:
            return False

        return True

    def train_with_constraints(self, global_model, args):
        """Entraîne le modèle local avec contraintes edge"""
        if self.training_data is None:
            return None, 0, 0

        try:
            # Créer modèle local adapté aux contraintes
            self.local_model = self._create_constrained_model(global_model, args)

            # Estimer temps et énergie avant entraînement
            estimated_time = self.estimate_training_time(epochs=1)
            estimated_energy = self.estimate_energy_consumption(estimated_time)

            # Vérifier si assez de batterie
            if not self._check_energy_budget(estimated_energy):
                return None, 0, estimated_time

            start_time = time.time()

            # Entraînement local avec contraintes
            x_train, y_train = self.training_data

            # Adapter taille de batch selon mémoire
            batch_size = self._adapt_batch_size(args.batch_size if hasattr(args, 'batch_size') else 32)

            # Entraînement
            history = self.local_model.fit(
                x_train, y_train,
                epochs=1,
                batch_size=batch_size,
                verbose=0,
                validation_split=0.1 if len(x_train) > 100 else 0
            )

            actual_time = time.time() - start_time
            actual_energy = self.estimate_energy_consumption(actual_time)

            # Calculer mise à jour (delta poids)
            update = self._compute_model_update(global_model)

            # Compression si demandée
            if hasattr(args, 'compression') and args.compression:
                update = self._compress_update(update)

            # Enregistrer historique
            self.training_history.append({
                'round': len(self.training_history),
                'time': actual_time,
                'energy': actual_energy,
                'accuracy': history.history.get('accuracy', [0])[-1]
            })
            self.energy_history.append(actual_energy)

            return update, actual_energy, actual_time

        except Exception as e:
            print(f"Erreur entraînement dispositif {self.device_id}: {e}")
            return None, 0, 0

    def _create_constrained_model(self, global_model, args):
        """Crée un modèle adapté aux contraintes du dispositif"""
        # Copier le modèle global
        local_model = tf.keras.models.clone_model(global_model)
        local_model.set_weights(global_model.get_weights())

        # Adapter l'optimiseur selon les ressources
        if self.specs['cpu_power'] < 0.3:
            # Dispositifs faibles: optimiseur simple
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        else:
            # Dispositifs puissants: Adam
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        local_model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return local_model

    def _adapt_batch_size(self, requested_batch_size):
        """Adapte la taille de batch selon la mémoire disponible"""
        available_memory = self.specs['memory_gb'] * (1 - self.current_load)

        if available_memory < 1:
            return min(8, requested_batch_size)
        elif available_memory < 2:
            return min(16, requested_batch_size)
        else:
            return requested_batch_size

    def _check_energy_budget(self, required_energy):
        """Vérifie si assez d'énergie pour l'entraînement"""
        if self.specs['battery_capacity'] == float('inf'):
            return True  # Dispositif alimenté

        if self.is_charging:
            return True  # En charge

        # Convertir énergie en pourcentage de batterie
        # Approximation: 1J ≈ 0.01% de batterie pour dispositif moyen
        energy_cost_percent = required_energy * 0.01

        return self.battery_level > energy_cost_percent + 10  # Garder 10% de marge

    def _compute_model_update(self, global_model):
        """Calcule la mise à jour (différence entre modèle local et global)"""
        local_weights = self.local_model.get_weights()
        global_weights = global_model.get_weights()

        update = []
        for local_w, global_w in zip(local_weights, global_weights):
            update.append(local_w - global_w)

        return update

    def _compress_update(self, update, compression_ratio=0.1):
        """Compresse la mise à jour pour réduire communication"""
        compressed_update = []

        for layer_update in update:
            # Sparsification: garder seulement les plus grandes valeurs
            flat_update = layer_update.flatten()
            threshold = np.percentile(np.abs(flat_update), (1 - compression_ratio) * 100)

            # Masque pour garder les valeurs importantes
            mask = np.abs(flat_update) >= threshold
            sparse_update = flat_update * mask

            compressed_update.append(sparse_update.reshape(layer_update.shape))

        return compressed_update

    def update_battery(self, energy_consumed):
        """Met à jour le niveau de batterie"""
        if self.specs['battery_capacity'] == float('inf'):
            return  # Dispositif alimenté

        if self.is_charging:
            # En charge: gain d'énergie
            charge_rate = 5.0  # % par minute
            self.battery_level = min(100, self.battery_level + charge_rate * 0.1)
        else:
            # Consommation d'énergie
            energy_cost_percent = energy_consumed * 0.01
            self.battery_level = max(0, self.battery_level - energy_cost_percent)

    def update_mobility(self, time_delta=1.0):
        """Met à jour la position si dispositif mobile"""
        if not self.is_mobile or self.velocity == 0:
            return

        # Déplacement simple en ligne droite
        distance_km = self.velocity * (time_delta / 3600)  # time_delta en secondes

        # Mise à jour approximative position (simplifiée)
        lat_change = distance_km / 111  # 1 degré ≈ 111 km
        lon_change = distance_km / (111 * np.cos(np.radians(self.location['lat'])))

        self.location['lat'] += random.uniform(-lat_change, lat_change)
        self.location['lon'] += random.uniform(-lon_change, lon_change)

        # Garder dans les limites
        self.location['lat'] = np.clip(self.location['lat'], -90, 90)
        self.location['lon'] = np.clip(self.location['lon'], -180, 180)

    def simulate_failure(self, failure_rate=0.01):
        """Simule des pannes aléatoires"""
        if random.random() < failure_rate:
            self.is_online = False
            return True
        return False

    def recover_from_failure(self, recovery_rate=0.1):
        """Simule la récupération après panne"""
        if not self.is_online and random.random() < recovery_rate:
            self.is_online = True
            return True
        return False

    def get_status(self):
        """Retourne l'état actuel du dispositif"""
        return {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'battery_level': self.battery_level,
            'is_charging': self.is_charging,
            'network_quality': self.network_quality,
            'current_load': self.current_load,
            'is_online': self.is_online,
            'location': self.location,
            'velocity': self.velocity,
            'can_participate': self.can_participate()
        }


def create_edge_devices(num_devices, device_types, scenario):
    """
    Crée une flotte de dispositifs edge selon le scénario

    Args:
        num_devices: nombre de dispositifs à créer
        device_types: types de dispositifs ('mixed' ou liste spécifique)
        scenario: scénario d'usage

    Returns:
        List[EdgeDevice]: liste des dispositifs créés
    """

    devices = []

    # Définir la distribution des types selon le scénario
    type_distributions = get_scenario_device_distribution(scenario, device_types)

    for i in range(num_devices):
        # Sélectionner type de dispositif
        device_type = select_device_type(type_distributions)

        # Créer localisation selon scénario
        location = generate_scenario_location(scenario, i, num_devices)

        # Créer dispositif
        device = EdgeDevice(
            device_id=f"edge_device_{i:03d}",
            device_type=device_type,
            location=location
        )

        # Personnaliser selon scénario
        customize_device_for_scenario(device, scenario)

        devices.append(device)

    print(f"✅ Créé {len(devices)} dispositifs edge:")
    device_count = {}
    for device in devices:
        device_count[device.device_type] = device_count.get(device.device_type, 0) + 1

    for device_type, count in device_count.items():
        print(f"   📱 {device_type}: {count}")

    return devices


def get_scenario_device_distribution(scenario, device_types):
    """Définit la distribution des types de dispositifs par scénario"""

    if device_types != 'mixed':
        # Types spécifiés par l'utilisateur
        if isinstance(device_types, str):
            specified_types = device_types.split(',')
        else:
            specified_types = device_types
        return {dtype.strip(): 1.0 / len(specified_types) for dtype in specified_types}

    # Distributions prédéfinies par scénario
    distributions = {
        'iot': {
            'industrial_sensor': 0.4,
            'raspberry_pi': 0.3,
            'smart_camera': 0.2,
            'smartphone_low': 0.1
        },
        'mobile': {
            'smartphone_high': 0.4,
            'smartphone_low': 0.3,
            'tablet': 0.3
        },
        'vehicular': {
            'vehicle_ecu': 0.6,
            'smartphone_high': 0.3,
            'tablet': 0.1
        },
        'healthcare': {
            'medical_device': 0.5,
            'tablet': 0.3,
            'smartphone_high': 0.2
        },
        'smart_city': {
            'smart_camera': 0.3,
            'industrial_sensor': 0.3,
            'raspberry_pi': 0.2,
            'smartphone_low': 0.2
        },
        'industrial': {
            'industrial_sensor': 0.5,
            'raspberry_pi': 0.3,
            'vehicle_ecu': 0.2
        }
    }

    return distributions.get(scenario, {
        'smartphone_low': 0.3,
        'smartphone_high': 0.2,
        'tablet': 0.2,
        'raspberry_pi': 0.15,
        'smart_camera': 0.1,
        'industrial_sensor': 0.05
    })


def select_device_type(type_distribution):
    """Sélectionne un type de dispositif selon la distribution"""
    types = list(type_distribution.keys())
    probabilities = list(type_distribution.values())

    return np.random.choice(types, p=probabilities)


def generate_scenario_location(scenario, device_index, total_devices):
    """Génère une localisation selon le scénario"""

    if scenario == 'iot':
        # IoT: dispositifs dans zones industrielles/domestiques
        zones = ['industrial_zone', 'residential', 'office_building']
        zone = zones[device_index % len(zones)]

        return {
            'lat': random.uniform(45.0, 45.1),  # Zone concentrée
            'lon': random.uniform(2.0, 2.1),
            'zone': zone,
            'building_floor': random.randint(1, 10) if zone == 'office_building' else 1
        }

    elif scenario == 'mobile':
        # Mobile: utilisateurs dispersés en ville
        return {
            'lat': random.uniform(48.8, 48.9),  # Paris approximatif
            'lon': random.uniform(2.3, 2.4),
            'zone': 'urban',
            'mobility_pattern': random.choice(['commuter', 'tourist', 'resident'])
        }

    elif scenario == 'vehicular':
        # Véhiculaire: le long des routes
        highways = [
            {'lat_range': (48.85, 48.87), 'lon_range': (2.30, 2.35)},  # A1
            {'lat_range': (48.82, 48.84), 'lon_range': (2.25, 2.30)},  # A4
        ]
        highway = highways[device_index % len(highways)]

        return {
            'lat': random.uniform(*highway['lat_range']),
            'lon': random.uniform(*highway['lon_range']),
            'zone': 'highway',
            'road_segment': f"segment_{device_index % 10}"
        }

    elif scenario == 'healthcare':
        # Santé: hôpitaux, cliniques, domiciles
        healthcare_zones = ['hospital', 'clinic', 'home', 'ambulance']
        zone = healthcare_zones[device_index % len(healthcare_zones)]

        return {
            'lat': random.uniform(48.85, 48.87),
            'lon': random.uniform(2.33, 2.37),
            'zone': zone,
            'ward': f"ward_{random.randint(1, 20)}" if zone == 'hospital' else None
        }

    elif scenario == 'smart_city':
        # Ville intelligente: capteurs urbains distribués
        city_zones = ['downtown', 'residential', 'industrial', 'park', 'transport_hub']
        zone = city_zones[device_index % len(city_zones)]

        return {
            'lat': random.uniform(48.84, 48.88),
            'lon': random.uniform(2.32, 2.38),
            'zone': zone,
            'district': f"district_{device_index // (total_devices // 5)}"
        }

    elif scenario == 'industrial':
        # Industriel: usines et sites de production
        return {
            'lat': random.uniform(48.90, 48.95),  # Zone industrielle
            'lon': random.uniform(2.40, 2.50),
            'zone': 'industrial',
            'factory_line': f"line_{device_index % 5}",
            'machine_id': f"machine_{device_index}"
        }

    else:
        # Défaut: localisation aléatoire
        return {
            'lat': random.uniform(-90, 90),
            'lon': random.uniform(-180, 180),
            'zone': 'unknown'
        }


def customize_device_for_scenario(device, scenario):
    """Personnalise le dispositif selon le scénario"""

    if scenario == 'iot':
        # IoT: faible mobilité, longue durée de vie
        device.velocity = 0
        device.is_mobile = False
        if device.device_type == 'industrial_sensor':
            device.battery_level = 100
            device.is_charging = True

    elif scenario == 'mobile':
        # Mobile: forte mobilité, usage personnel
        device.velocity = random.uniform(0, 50)  # 0-50 km/h
        device.is_mobile = True
        device.current_load = random.uniform(0.2, 0.6)  # Usage actif

    elif scenario == 'vehicular':
        # Véhiculaire: très mobile, connectivité variable
        device.velocity = random.uniform(30, 130)  # 30-130 km/h
        device.is_mobile = True
        device.network_quality = random.uniform(0.4, 0.9)  # Réseau variable
        if device.device_type == 'vehicle_ecu':
            device.is_charging = True  # Alimenté par véhicule

    elif scenario == 'healthcare':
        # Santé: haute fiabilité, sécurité critique
        device.specs['reliability'] = min(0.99, device.specs['reliability'] + 0.1)
        if device.device_type == 'medical_device':
            device.battery_level = random.uniform(80, 100)  # Batterie surveillée

    elif scenario == 'smart_city':
        # Ville intelligente: déploiement fixe, alimentation variable
        device.velocity = 0
        device.is_mobile = False
        if device.device_type in ['smart_camera', 'industrial_sensor']:
            device.is_charging = random.choice([True, False])  # Parfois sur batterie

    elif scenario == 'industrial':
        # Industriel: environnement contrôlé, haute disponibilité
        device.velocity = random.uniform(0, 5)  # Mouvement limité (robots)
        device.network_quality = random.uniform(0.8, 1.0)  # Réseau stable
        device.specs['reliability'] = min(0.99, device.specs['reliability'] + 0.05)
        device.is_charging = True  # Souvent alimenté


def simulate_device_fleet_evolution(devices, time_minutes=60):
    """Simule l'évolution d'une flotte de dispositifs dans le temps"""

    for minute in range(time_minutes):
        for device in devices:
            # Mise à jour batterie
            if not device.is_charging and device.specs['battery_capacity'] != float('inf'):
                drain_rate = 0.1 + device.current_load * 0.2  # %/minute
                device.battery_level = max(0, device.battery_level - drain_rate)
            elif device.is_charging:
                charge_rate = 2.0  # %/minute
                device.battery_level = min(100, device.battery_level + charge_rate)

            # Mise à jour mobilité
            if device.is_mobile:
                device.update_mobility(time_delta=60)  # 60 secondes

            # Variation de la charge CPU
            device.current_load += random.uniform(-0.05, 0.05)
            device.current_load = np.clip(device.current_load, 0.05, 0.95)

            # Variation qualité réseau
            device.network_quality += random.uniform(-0.1, 0.1)
            device.network_quality = np.clip(device.network_quality, 0.1, 1.0)

            # Pannes et récupérations
            if device.is_online:
                device.simulate_failure(failure_rate=0.001)  # 0.1% par minute
            else:
                device.recover_from_failure(recovery_rate=0.1)  # 10% récupération

    return devices


def get_fleet_statistics(devices):
    """Calcule les statistiques de la flotte de dispositifs"""

    total_devices = len(devices)
    online_devices = sum(1 for d in devices if d.is_online)
    available_devices = sum(1 for d in devices if d.can_participate())

    avg_battery = np.mean([d.battery_level for d in devices
                           if d.specs['battery_capacity'] != float('inf')])

    avg_network_quality = np.mean([d.network_quality for d in devices])

    device_types = {}
    for device in devices:
        device_types[device.device_type] = device_types.get(device.device_type, 0) + 1

    return {
        'total_devices': total_devices,
        'online_devices': online_devices,
        'available_devices': available_devices,
        'availability_rate': available_devices / total_devices,
        'avg_battery_level': avg_battery,
        'avg_network_quality': avg_network_quality,
        'device_type_distribution': device_types,
        'mobile_devices': sum(1 for d in devices if d.is_mobile),
        'charging_devices': sum(1 for d in devices if d.is_charging)
    }
    