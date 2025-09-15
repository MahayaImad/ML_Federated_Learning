"""
Chargement et distribution des données pour edge FL
"""
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import random


def load_edge_dataset(dataset_name, devices, scenario):
    """
    Charge et distribue les données selon le scénario edge

    Args:
        dataset_name: 'mnist', 'cifar', 'synthetic'
        devices: liste des dispositifs edge
        scenario: scénario d'usage edge

    Returns:
        dict: données distribuées par dispositif
    """

    if dataset_name == 'mnist':
        return load_mnist_edge(devices, scenario)
    elif dataset_name == 'cifar':
        return load_cifar_edge(devices, scenario)
    elif dataset_name == 'synthetic':
        return load_synthetic_edge(devices, scenario)
    else:
        raise ValueError(f"Dataset non supporté: {dataset_name}")


def load_mnist_edge(devices, scenario):
    """Charge MNIST pour environnement edge"""
    print("📥 Chargement MNIST pour edge...")

    # Charger données
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Ajouter dimension canal
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Distribution selon scénario
    device_data = distribute_data_by_scenario(
        x_train, y_train, devices, scenario, 'mnist'
    )

    return {
        'device_data': device_data,
        'test_data': (x_test, y_test),
        'input_shape': x_train.shape[1:],
        'num_classes': 10
    }


def load_cifar_edge(devices, scenario):
    """Charge CIFAR-10 pour environnement edge"""
    print("📥 Chargement CIFAR-10 pour edge...")

    # Charger données
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Aplatir labels
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Distribution selon scénario
    device_data = distribute_data_by_scenario(
        x_train, y_train, devices, scenario, 'cifar'
    )

    return {
        'device_data': device_data,
        'test_data': (x_test, y_test),
        'input_shape': x_train.shape[1:],
        'num_classes': 10
    }


def load_synthetic_edge(devices, scenario):
    """Génère des données synthétiques pour edge"""
    print("🔄 Génération données synthétiques pour edge...")

    # Paramètres selon scénario
    num_features = 20
    num_classes = 5
    total_samples = 10000

    # Générer données synthétiques
    np.random.seed(42)
    x_data = np.random.randn(total_samples, num_features)

    # Créer classes avec séparabilité variable selon scénario
    if scenario in ['iot', 'industrial']:
        # Données plus séparables (capteurs avec patterns clairs)
        class_centers = np.random.randn(num_classes, num_features) * 3
    else:
        # Données moins séparables (usage général)
        class_centers = np.random.randn(num_classes, num_features) * 1.5

    # Assigner classes
    y_data = np.zeros(total_samples)
    for i, sample in enumerate(x_data):
        distances = [np.linalg.norm(sample - center) for center in class_centers]
        y_data[i] = np.argmin(distances)

    # Split train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    # Distribution aux dispositifs
    device_data = distribute_data_by_scenario(
        x_train, y_train, devices, scenario, 'synthetic'
    )

    return {
        'device_data': device_data,
        'test_data': (x_test, y_test),
        'input_shape': (num_features,),
        'num_classes': num_classes
    }


def distribute_data_by_scenario(x_data, y_data, devices, scenario, dataset_type):
    """Distribue les données selon le scénario edge"""

    device_data = {}

    if scenario == 'iot':
        device_data = distribute_iot_data(x_data, y_data, devices)
    elif scenario == 'mobile':
        device_data = distribute_mobile_data(x_data, y_data, devices)
    elif scenario == 'vehicular':
        device_data = distribute_vehicular_data(x_data, y_data, devices)
    elif scenario == 'healthcare':
        device_data = distribute_healthcare_data(x_data, y_data, devices)
    elif scenario == 'smart_city':
        device_data = distribute_smart_city_data(x_data, y_data, devices)
    elif scenario == 'industrial':
        device_data = distribute_industrial_data(x_data, y_data, devices)
    else:
        # Distribution par défaut
        device_data = distribute_default_data(x_data, y_data, devices)

    # Assigner données aux dispositifs
    for device in devices:
        if device.device_id in device_data:
            x_device, y_device = device_data[device.device_id]
            device.set_training_data(x_device, y_device)

    return device_data


def distribute_iot_data(x_data, y_data, devices):
    """Distribution pour IoT: données par zone/capteur"""
    device_data = {}

    # Grouper par zones IoT
    zones = {}
    for device in devices:
        zone = device.location.get('zone', 'default')
        if zone not in zones:
            zones[zone] = []
        zones[zone].append(device.device_id)

    # Distribuer données par zone avec spécialisation
    data_per_zone = len(x_data) // max(len(zones), 1)

    for zone_idx, (zone_name, device_ids) in enumerate(zones.items()):
        # Données spécifiques à cette zone
        start_idx = zone_idx * data_per_zone
        end_idx = min(start_idx + data_per_zone, len(x_data))

        zone_x = x_data[start_idx:end_idx]
        zone_y = y_data[start_idx:end_idx]

        # Distribuer dans la zone (très non-IID par dispositif)
        data_per_device = len(zone_x) // len(device_ids)

        for device_idx, device_id in enumerate(device_ids):
            dev_start = device_idx * data_per_device
            dev_end = min(dev_start + data_per_device, len(zone_x))

            if dev_start < len(zone_x):
                device_data[device_id] = (
                    zone_x[dev_start:dev_end],
                    zone_y[dev_start:dev_end]
                )

    return device_data


def distribute_mobile_data(x_data, y_data, devices):
    """Distribution pour mobile: données personnalisées"""
    device_data = {}

    # Chaque utilisateur mobile a ses propres patterns
    num_classes = len(np.unique(y_data))
    data_per_device = len(x_data) // len(devices)

    for i, device in enumerate(devices):
        device_id = device.device_id

        # Sélection personnalisée (certaines classes favorisées)
        user_preferences = np.random.dirichlet([1] * num_classes)

        # Sélectionner échantillons selon préférences
        device_indices = []
        target_samples = min(data_per_device, len(x_data) - len(device_indices))

        for class_id in range(num_classes):
            class_mask = y_data == class_id
            class_indices = np.where(class_mask)[0]

            if len(class_indices) > 0:
                num_class_samples = int(target_samples * user_preferences[class_id])
                selected = np.random.choice(
                    class_indices,
                    min(num_class_samples, len(class_indices)),
                    replace=False
                )
                device_indices.extend(selected)

        if device_indices:
            device_data[device_id] = (
                x_data[device_indices],
                y_data[device_indices]
            )

    return device_data


def distribute_vehicular_data(x_data, y_data, devices):
    """Distribution pour véhiculaire: données par route/zone"""
    device_data = {}

    # Grouper véhicules par segment de route
    road_segments = {}
    for device in devices:
        segment = device.location.get('road_segment', 'highway_1')
        if segment not in road_segments:
            road_segments[segment] = []
        road_segments[segment].append(device.device_id)

    # Données partagées par segment (plus IID qu'IoT)
    for segment_name, device_ids in road_segments.items():
        # Chaque segment a accès à un sous-ensemble de données
        segment_size = len(x_data) // len(road_segments)
        segment_indices = np.random.choice(
            len(x_data), segment_size, replace=False
        )

        segment_x = x_data[segment_indices]
        segment_y = y_data[segment_indices]

        # Distribution plus équitable dans le segment
        for device_id in device_ids:
            # Échantillonnage aléatoire dans le segment
            device_size = len(segment_x) // len(device_ids)
            if device_size > 0:
                device_indices = np.random.choice(
                    len(segment_x), device_size, replace=False
                )
                device_data[device_id] = (
                    segment_x[device_indices],
                    segment_y[device_indices]
                )

    return device_data


def distribute_healthcare_data(x_data, y_data, devices):
    """Distribution pour santé: données très privées et spécialisées"""
    device_data = {}

    # Classification par type de dispositif médical
    medical_specialties = {
        'medical_device': 'cardiology',
        'tablet': 'general',
        'smartphone_high': 'monitoring'
    }

    # Créer spécialisation des données par domaine médical
    specialty_data = {}
    num_specialties = len(set(medical_specialties.values()))
    data_per_specialty = len(x_data) // num_specialties

    for spec_idx, specialty in enumerate(set(medical_specialties.values())):
        start_idx = spec_idx * data_per_specialty
        end_idx = min(start_idx + data_per_specialty, len(x_data))
        specialty_data[specialty] = (
            x_data[start_idx:end_idx],
            y_data[start_idx:end_idx]
        )

    # Distribuer selon spécialité
    for device in devices:
        device_id = device.device_id
        device_type = device.device_type

        specialty = medical_specialties.get(device_type, 'general')

        if specialty in specialty_data:
            spec_x, spec_y = specialty_data[specialty]

            # Petits datasets pour confidentialité
            max_samples = min(len(spec_x) // 3, 1000)
            if max_samples > 0:
                sample_indices = np.random.choice(
                    len(spec_x), max_samples, replace=False
                )
                device_data[device_id] = (
                    spec_x[sample_indices],
                    spec_y[sample_indices]
                )

    return device_data


def distribute_smart_city_data(x_data, y_data, devices):
    """Distribution pour ville intelligente: données par zone urbaine"""
    device_data = {}

    # Zones urbaines avec caractéristiques différentes
    city_zones = {
        'downtown': {'data_volume': 'high', 'diversity': 'high'},
        'residential': {'data_volume': 'medium', 'diversity': 'medium'},
        'industrial': {'data_volume': 'medium', 'diversity': 'low'},
        'park': {'data_volume': 'low', 'diversity': 'low'}
    }

    # Grouper dispositifs par zone
    zone_devices = {}
    for device in devices:
        zone = device.location.get('zone', 'residential')
        if zone not in zone_devices:
            zone_devices[zone] = []
        zone_devices[zone].append(device.device_id)

    # Allouer données selon caractéristiques de zone
    total_allocated = 0

    for zone_name, device_ids in zone_devices.items():
        zone_props = city_zones.get(zone_name, {'data_volume': 'medium', 'diversity': 'medium'})

        # Volume de données selon la zone
        if zone_props['data_volume'] == 'high':
            zone_data_ratio = 0.4
        elif zone_props['data_volume'] == 'medium':
            zone_data_ratio = 0.3
        else:  # low
            zone_data_ratio = 0.2

        zone_data_size = int(len(x_data) * zone_data_ratio / len(zone_devices))
        start_idx = total_allocated
        end_idx = min(start_idx + zone_data_size, len(x_data))

        if start_idx < len(x_data):
            zone_x = x_data[start_idx:end_idx]
            zone_y = y_data[start_idx:end_idx]
            total_allocated = end_idx

            # Diversité selon la zone
            if zone_props['diversity'] == 'low':
                # Filtrer pour réduire diversité
                dominant_classes = np.random.choice(
                    np.unique(zone_y),
                    size=min(3, len(np.unique(zone_y))),
                    replace=False
                )
                mask = np.isin(zone_y, dominant_classes)
                zone_x = zone_x[mask]
                zone_y = zone_y[mask]

            # Distribuer aux dispositifs de la zone
            if len(zone_x) > 0:
                data_per_device = len(zone_x) // len(device_ids)

                for device_idx, device_id in enumerate(device_ids):
                    dev_start = device_idx * data_per_device
                    dev_end = min(dev_start + data_per_device, len(zone_x))

                    if dev_start < len(zone_x):
                        device_data[device_id] = (
                            zone_x[dev_start:dev_end],
                            zone_y[dev_start:dev_end]
                        )

    return device_data


def distribute_industrial_data(x_data, y_data, devices):
    """Distribution pour industrie: données par ligne de production"""
    device_data = {}

    # Grouper par ligne de production
    production_lines = {}
    for device in devices:
        line = device.location.get('factory_line', 'line_1')
        if line not in production_lines:
            production_lines[line] = []
        production_lines[line].append(device.device_id)

    # Chaque ligne a des patterns de données spécifiques
    for line_idx, (line_name, device_ids) in enumerate(production_lines.items()):
        # Données spécifiques à cette ligne (très spécialisées)
        line_classes = np.random.choice(
            np.unique(y_data),
            size=min(2, len(np.unique(y_data))),
            replace=False
        )

        # Filtrer données pour cette ligne
        line_mask = np.isin(y_data, line_classes)
        line_x = x_data[line_mask]
        line_y = y_data[line_mask]

        if len(line_x) > 0:
            # Distribution équitable dans la ligne
            data_per_device = len(line_x) // len(device_ids)

            for device_idx, device_id in enumerate(device_ids):
                dev_start = device_idx * data_per_device
                dev_end = min(dev_start + data_per_device, len(line_x))

                if dev_start < len(line_x):
                    device_data[device_id] = (
                        line_x[dev_