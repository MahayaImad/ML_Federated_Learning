"""
Modèles optimisés pour dispositifs edge
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def create_edge_model(dataset_name, input_shape, num_classes, device_constraints=None):
    """
    Crée un modèle optimisé pour dispositifs edge

    Args:
        dataset_name: 'mnist', 'cifar', 'synthetic'
        input_shape: forme des données d'entrée
        num_classes: nombre de classes
        device_constraints: contraintes du dispositif (optionnel)

    Returns:
        model: modèle Keras optimisé pour edge
    """

    if device_constraints is None:
        device_constraints = {'cpu_power': 0.5, 'memory_gb': 2}

    if dataset_name == 'mnist':
        return create_mnist_edge_model(input_shape, num_classes, device_constraints)
    elif dataset_name == 'cifar':
        return create_cifar_edge_model(input_shape, num_classes, device_constraints)
    elif dataset_name == 'synthetic':
        return create_synthetic_edge_model(input_shape, num_classes, device_constraints)
    else:
        raise ValueError(f"Dataset non supporté: {dataset_name}")


def create_mnist_edge_model(input_shape, num_classes, device_constraints):
    """Modèle CNN léger pour MNIST sur edge"""

    cpu_power = device_constraints.get('cpu_power', 0.5)
    memory_gb = device_constraints.get('memory_gb', 2)

    # Adapter architecture selon contraintes
    if cpu_power < 0.3 or memory_gb < 1:
        # Dispositifs très contraints: modèle minimal
        return create_minimal_mnist_model(input_shape, num_classes)
    elif cpu_power < 0.6 or memory_gb < 2:
        # Dispositifs moyennement contraints: modèle léger
        return create_lightweight_mnist_model(input_shape, num_classes)
    else:
        # Dispositifs performants: modèle standard edge
        return create_standard_mnist_edge_model(input_shape, num_classes)


def create_minimal_mnist_model(input_shape, num_classes):
    """Modèle minimal pour dispositifs très contraints"""
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ], name='MNIST_Minimal_Edge')

    return model


def create_lightweight_mnist_model(input_shape, num_classes):
    """Modèle léger pour dispositifs moyennement contraints"""
    model = tf.keras.Sequential([
        layers.Conv2D(8, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='MNIST_Lightweight_Edge')

    return model


def create_standard_mnist_edge_model(input_shape, num_classes):
    """Modèle standard edge pour MNIST"""
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ], name='MNIST_Standard_Edge')

    return model


def create_cifar_edge_model(input_shape, num_classes, device_constraints):
    """Modèle CNN léger pour CIFAR-10 sur edge"""

    cpu_power = device_constraints.get('cpu_power', 0.5)
    memory_gb = device_constraints.get('memory_gb', 2)

    if cpu_power < 0.3 or memory_gb < 1:
        return create_minimal_cifar_model(input_shape, num_classes)
    elif cpu_power < 0.6 or memory_gb < 3:
        return create_lightweight_cifar_model(input_shape, num_classes)
    else:
        return create_standard_cifar_edge_model(input_shape, num_classes)


def create_minimal_cifar_model(input_shape, num_classes):
    """Modèle minimal pour CIFAR-10"""
    model = tf.keras.Sequential([
        layers.Conv2D(8, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((4, 4)),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='CIFAR_Minimal_Edge')

    return model


def create_lightweight_cifar_model(input_shape, num_classes):
    """Modèle léger pour CIFAR-10"""
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ], name='CIFAR_Lightweight_Edge')

    return model


def create_standard_cifar_edge_model(input_shape, num_classes):
    """Modèle standard edge pour CIFAR-10"""
    model = tf.keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='CIFAR_Standard_Edge')

    return model


def create_synthetic_edge_model(input_shape, num_classes, device_constraints):
    """Modèle pour données synthétiques edge"""

    # Modèle simple pour données synthétiques
    model = tf.keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ], name='Synthetic_Edge')

    return model


def create_mobilenet_edge(input_shape, num_classes, alpha=0.25):
    """MobileNet adapté pour edge (version très légère)"""

    def depthwise_conv_block(x, filters, stride=1):
        """Bloc convolution dépthwise"""
        x = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters, (1, 1), strides=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        return x

    inputs = layers.Input(shape=input_shape)

    # Première convolution
    x = layers.Conv2D(int(32 * alpha), (3, 3), strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Blocs depthwise
    x = depthwise_conv_block(x, int(64 * alpha))
    x = depthwise_conv_block(x, int(128 * alpha), stride=2)
    x = depthwise_conv_block(x, int(128 * alpha))
    x = depthwise_conv_block(x, int(256 * alpha), stride=2)
    x = depthwise_conv_block(x, int(256 * alpha))

    # Classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs, name=f'MobileNet_Edge_α{alpha}')

    return model


def create_quantized_model(base_model):
    """Crée une version quantifiée d'un modèle pour edge"""

    # Simulation de quantification (TensorFlow Lite quantization en pratique)
    quantized_model = tf.keras.models.clone_model(base_model)
    quantized_model.set_weights(base_model.get_weights())

    # Marquer comme quantifié
    quantized_model._name = base_model.name + '_Quantized'

    return quantized_model


def create_pruned_model(base_model, pruning_ratio=0.5):
    """Crée une version élaguée d'un modèle"""

    pruned_model = tf.keras.models.clone_model(base_model)

    # Simulation d'élagage (magnitude-based pruning en pratique)
    weights = base_model.get_weights()
    pruned_weights = []

    for weight_matrix in weights:
        if len(weight_matrix.shape) > 1:  # Seulement les couches avec poids
            # Élagage par magnitude
            threshold = np.percentile(np.abs(weight_matrix), pruning_ratio * 100)
            mask = np.abs(weight_matrix) > threshold
            pruned_weight = weight_matrix * mask
            pruned_weights.append(pruned_weight)
        else:
            # Garder les biais inchangés
            pruned_weights.append(weight_matrix)

    pruned_model.set_weights(pruned_weights)
    pruned_model._name = base_model.name + f'_Pruned_{int(pruning_ratio * 100)}%'

    return pruned_model


def create_federated_averaging_model(client_models, client_weights=None):
    """Crée un modèle moyenné pour agrégation fédérée"""

    if not client_models:
        raise ValueError("Liste de modèles clients vide")

    # Poids uniformes si non spécifiés
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)

    # Normaliser les poids
    total_weight = sum(client_weights)
    normalized_weights = [w / total_weight for w in client_weights]

    # Créer modèle agrégé
    aggregated_model = tf.keras.models.clone_model(client_models[0])

    # Moyenner les poids
    aggregated_weights = []
    for layer_idx in range(len(client_models[0].get_weights())):
        layer_weights = []

        for model_idx, model in enumerate(client_models):
            model_weights = model.get_weights()
            weighted_layer = model_weights[layer_idx] * normalized_weights[model_idx]
            layer_weights.append(weighted_layer)

        # Sommer les poids pondérés
        aggregated_layer = sum(layer_weights)
        aggregated_weights.append(aggregated_layer)

    aggregated_model.set_weights(aggregated_weights)
    aggregated_model._name = 'FedAvg_Aggregated'

    return aggregated_model


def create_adaptive_model(base_model, device_profile):
    """Crée un modèle adapté au profil d'un dispositif"""

    cpu_power = device_profile.get('cpu_power', 0.5)
    memory_gb = device_profile.get('memory_gb', 2)
    battery_level = device_profile.get('battery_level', 50)

    # Décider des adaptations selon le profil
    adaptations = []

    if cpu_power < 0.3:
        adaptations.append('quantization')

    if memory_gb < 1.5:
        adaptations.append('pruning')

    if battery_level < 30:
        adaptations.append('early_stopping')

    # Appliquer adaptations
    adapted_model = base_model

    if 'quantization' in adaptations:
        adapted_model = create_quantized_model(adapted_model)

    if 'pruning' in adaptations:
        pruning_ratio = 0.7 if cpu_power < 0.2 else 0.5
        adapted_model = create_pruned_model(adapted_model, pruning_ratio)

    # Marquer les adaptations
    adaptation_suffix = '_'.join(adaptations) if adaptations else 'NoAdapt'
    adapted_model._name = f"{base_model.name}_Adaptive_{adaptation_suffix}"

    return adapted_model, adaptations


def get_model_complexity_metrics(model):
    """Calcule les métriques de complexité d'un modèle"""

    # Nombre de paramètres
    total_params = model.count_params()

    # Taille en mémoire (approximation)
    param_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes par paramètre float32

    # Complexité computationnelle (FLOPS approximés)
    flops = estimate_model_flops(model)

    # Taille du modèle sauvegardé (approximation)
    model_size_mb = param_size_mb * 1.2  # +20% pour métadonnées

    return {
        'total_parameters': total_params,
        'parameter_size_mb': param_size_mb,
        'estimated_flops': flops,
        'model_size_mb': model_size_mb,
        'memory_overhead_mb': param_size_mb * 2,  # Poids + gradients
        'complexity_score': calculate_complexity_score(total_params, flops)
    }


def estimate_model_flops(model):
    """Estime le nombre de FLOPS d'un modèle"""

    total_flops = 0

    for layer in model.layers:
        layer_flops = 0

        if isinstance(layer, layers.Dense):
            # Dense: input_dim * output_dim
            input_dim = layer.input_shape[-1] if layer.input_shape else 1
            output_dim = layer.units
            layer_flops = input_dim * output_dim


def estimate_model_flops(model):
    """Estime le nombre de FLOPS d'un modèle"""

    total_flops = 0

    for layer in model.layers:
        layer_flops = 0

        if isinstance(layer, layers.Dense):
            # Dense: input_dim * output_dim
            input_dim = layer.input_shape[-1] if layer.input_shape else 1
            output_dim = layer.units
            layer_flops = input_dim * output_dim

        elif isinstance(layer, layers.Conv2D):
            # Conv2D: output_h * output_w * kernel_h * kernel_w * input_channels * output_channels
            if hasattr(layer, 'kernel_size') and hasattr(layer, 'filters'):
                kernel_h, kernel_w = layer.kernel_size
                output_channels = layer.filters
                # Approximation pour output dimensions
                input_h, input_w = 32, 32  # Approximation
                layer_flops = input_h * input_w * kernel_h * kernel_w * output_channels

        elif isinstance(layer, layers.DepthwiseConv2D):
            # DepthwiseConv2D: moins de FLOPS que Conv2D standard
            if hasattr(layer, 'kernel_size'):
                kernel_h, kernel_w = layer.kernel_size
                input_h, input_w = 32, 32  # Approximation
                layer_flops = input_h * input_w * kernel_h * kernel_w

        total_flops += layer_flops

    return total_flops


def calculate_complexity_score(num_params, flops):
    """Calcule un score de complexité normalisé"""

    # Normaliser sur des modèles de référence
    # ResNet50: ~25M params, ~4G FLOPS
    ref_params = 25_000_000
    ref_flops = 4_000_000_000

    param_score = num_params / ref_params
    flops_score = flops / ref_flops

    # Score combiné (0-1, plus bas = moins complexe)
    complexity_score = (param_score * 0.4 + flops_score * 0.6)

    return min(complexity_score, 2.0)  # Cap à 2.0


def create_edge_optimized_compiler():
    """Crée un compilateur optimisé pour edge"""

    def compile_for_edge(model, device_constraints):
        """Compile un modèle pour dispositif edge"""

        cpu_power = device_constraints.get('cpu_power', 0.5)
        memory_gb = device_constraints.get('memory_gb', 2)

        # Choisir optimiseur selon contraintes
        if cpu_power < 0.3:
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        elif cpu_power < 0.6:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Compiler avec optimisations edge
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False  # Mode graph pour performance
        )

        return model

    return compile_for_edge


def create_transfer_learning_edge_model(base_model_name, input_shape, num_classes, device_constraints):
    """Crée un modèle edge basé sur transfer learning"""

    # Modèles de base légers disponibles
    if base_model_name == 'mobilenet_v2':
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            alpha=0.35,  # Version très légère
            include_top=False,
            weights='imagenet'
        )
    else:
        # Fallback vers modèle custom
        return create_edge_model('cifar', input_shape, num_classes, device_constraints)

    # Geler les couches de base
    base_model.trainable = False

    # Ajouter couches de classification
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ], name=f'TransferLearning_{base_model_name}_Edge')

    return model


class EdgeModelManager:
    """Gestionnaire de modèles pour environnement edge"""

    def __init__(self):
        self.model_registry = {}
        self.model_profiles = {}

    def register_model(self, model_id, model, device_constraints):
        """Enregistre un modèle avec ses contraintes"""

        self.model_registry[model_id] = {
            'model': model,
            'device_constraints': device_constraints,
            'complexity_metrics': get_model_complexity_metrics(model),
            'creation_time': tf.timestamp(),
            'usage_count': 0
        }

        # Créer profil de performance
        self.model_profiles[model_id] = {
            'training_times': [],
            'inference_times': [],
            'accuracy_scores': [],
            'energy_consumptions': []
        }

    def get_best_model_for_device(self, device_profile, task_requirements):
        """Sélectionne le meilleur modèle pour un dispositif"""

        best_model_id = None
        best_score = float('-inf')

        for model_id, model_info in self.model_registry.items():
            score = self._evaluate_model_device_compatibility(
                model_info, device_profile, task_requirements
            )

            if score > best_score:
                best_score = score
                best_model_id = model_id

        if best_model_id:
            self.model_registry[best_model_id]['usage_count'] += 1
            return self.model_registry[best_model_id]['model'], best_score

        return None, 0

    def _evaluate_model_device_compatibility(self, model_info, device_profile, task_requirements):
        """Évalue la compatibilité modèle-dispositif"""

        complexity = model_info['complexity_metrics']
        device_cpu = device_profile.get('cpu_power', 0.5)
        device_memory = device_profile.get('memory_gb', 2)
        device_battery = device_profile.get('battery_level', 50)

        # Score de compatibilité (0-1)
        compatibility_score = 0

        # Facteur CPU
        cpu_requirement = complexity['complexity_score'] * 0.5
        if device_cpu >= cpu_requirement:
            compatibility_score += 0.3
        else:
            compatibility_score += 0.3 * (device_cpu / cpu_requirement)

        # Facteur mémoire
        memory_requirement = complexity['memory_overhead_mb'] / 1024  # GB
        if device_memory >= memory_requirement:
            compatibility_score += 0.3
        else:
            compatibility_score += 0.3 * (device_memory / memory_requirement)

        # Facteur énergétique
        if device_battery > 50:
            compatibility_score += 0.2
        elif device_battery > 20:
            compatibility_score += 0.1

        # Facteur performance historique
        model_id = None
        for mid, minfo in self.model_registry.items():
            if minfo == model_info:
                model_id = mid
                break

        if model_id and model_id in self.model_profiles:
            profile = self.model_profiles[model_id]
            if profile['accuracy_scores']:
                avg_accuracy = np.mean(profile['accuracy_scores'])
                compatibility_score += 0.2 * avg_accuracy

        return compatibility_score

    def update_model_performance(self, model_id, performance_metrics):
        """Met à jour les métriques de performance d'un modèle"""

        if model_id not in self.model_profiles:
            return

        profile = self.model_profiles[model_id]

        # Mettre à jour historique (garder 20 dernières mesures)
        for metric_name, values_list in profile.items():
            if metric_name in performance_metrics:
                values_list.append(performance_metrics[metric_name])
                if len(values_list) > 20:
                    values_list.pop(0)

    def get_model_recommendations(self, device_profile):
        """Recommande des modèles pour un dispositif"""

        recommendations = []

        for model_id, model_info in self.model_registry.items():
            compatibility = self._evaluate_model_device_compatibility(
                model_info, device_profile, {}
            )

            if compatibility > 0.5:  # Seuil de recommandation
                recommendations.append({
                    'model_id': model_id,
                    'compatibility_score': compatibility,
                    'complexity_metrics': model_info['complexity_metrics'],
                    'usage_count': model_info['usage_count']
                })

        # Trier par score de compatibilité
        recommendations.sort(key=lambda x: x['compatibility_score'], reverse=True)

        return recommendations[:5]  # Top 5 recommandations


class ModelCompressionSuite:
    """Suite d'outils de compression de modèles pour edge"""

    @staticmethod
    def apply_knowledge_distillation(teacher_model, student_model, temperature=3.0):
        """Applique la distillation de connaissances"""

        def distillation_loss(y_true, y_pred, teacher_pred, temperature, alpha=0.7):
            """Loss de distillation combinée"""

            # Loss standard
            student_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

            # Loss de distillation
            teacher_soft = tf.nn.softmax(teacher_pred / temperature)
            student_soft = tf.nn.softmax(y_pred / temperature)
            distillation_loss = tf.keras.losses.categorical_crossentropy(
                teacher_soft, student_soft
            ) * (temperature ** 2)

            # Loss combinée
            total_loss = alpha * distillation_loss + (1 - alpha) * student_loss

            return total_loss

        # Retourner fonction de loss personnalisée
        return lambda y_true, y_pred: distillation_loss(
            y_true, y_pred, teacher_model.predict(y_pred), temperature
        )

    @staticmethod
    def apply_structured_pruning(model, pruning_schedule):
        """Applique l'élagage structuré"""

        # Simulation d'élagage structuré
        pruned_model = tf.keras.models.clone_model(model)

        # Supprimer des neurones/filtres entiers selon le schedule
        for layer_name, pruning_ratio in pruning_schedule.items():
            # En pratique, utiliserait TensorFlow Model Optimization
            pass

        pruned_model._name = model.name + '_StructurallyPruned'
        return pruned_model

    @staticmethod
    def apply_low_rank_approximation(model, rank_ratio=0.5):
        """Applique l'approximation de rang faible"""

        # Simulation d'approximation SVD pour couches denses
        approximated_model = tf.keras.models.clone_model(model)

        weights = model.get_weights()
        new_weights = []

        for weight_matrix in weights:
            if len(weight_matrix.shape) == 2:  # Matrice de poids dense
                # SVD approximation
                U, s, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
                rank = int(len(s) * rank_ratio)

                # Reconstruction avec rang réduit
                approx_matrix = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
                new_weights.append(approx_matrix)
            else:
                new_weights.append(weight_matrix)

        approximated_model.set_weights(new_weights)
        approximated_model._name = model.name + f'_LowRank_{int(rank_ratio * 100)}%'

        return approximated_model