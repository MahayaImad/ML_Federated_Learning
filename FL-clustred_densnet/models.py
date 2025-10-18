import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from data_preparation import number_classes, data_shape


def create_model(dataset_name, input_shape):
    """
    Create a standard model for comparison

    Args:
        dataset_name: 'mnist', 'cifar10' or 'cifar100'
        input_shape: (28, 28, 1) or (32, 32, 3)

    Returns:
        model: Keras model
    """
    num_classes = number_classes(dataset_name)

    if dataset_name == 'mnist':
        return create_modernCNN(input_shape, num_classes)
    elif dataset_name == 'cifar10':
        # Use DenseNet with transfer learning for CIFAR-10
        return create_densenet_transfer(input_shape, num_classes)
    elif dataset_name == 'cifar100':
        return create_resnet18(input_shape, num_classes)
    else:
        raise ValueError(f" Non supported Dataset: {dataset_name}")


def create_modernCNN(input_shape, num_classes):
    """
    Modern CNN architecture for MNIST
    Architecture Overview:
    - Conv 32 filters 5x5 (ReLU)
    - MaxPool 2x2
    - Conv 64 filters 5x5 (ReLU)
    - MaxPool 2x2
    - Flatten
    - Dense 512 (ReLU)
    - Dropout 0.3
    - Dense num_classes (Softmax)
    """
    model = tf.keras.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='Modern_CNN')
    return model


def create_densenet_transfer(input_shape, num_classes):
    """
    DenseNet121 with transfer learning for CIFAR-10
    - Frozen base layers (pretrained on ImageNet)
    - Trainable top layers (custom classifier)

    Only trainable weights will be communicated in FL

    Args:
        input_shape: (32, 32, 3) for CIFAR-10
        num_classes: 10 for CIFAR-10

    Returns:
        model: Keras model with frozen base and trainable top
    """
    # Load pretrained DenseNet121 without top layers
    base_model = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )

    # Freeze the base model layers
    base_model.trainable = False

    # Build custom top layers (trainable)
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # Force inference mode

    # Custom classifier head
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(128, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs, outputs, name='DenseNet121_Transfer')

    return model


def resnet_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """
    Résidual Bloc of ResNet

    Args:
        x: input tensor
        conv_shortcut: use a convolution pour for the shortcut
        name: name of bloc
    """
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(name=name + '_0_bn')(shortcut)
    else:
        shortcut = x if stride == 1 else layers.MaxPooling2D(1, strides=stride)(x)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', name=name + '_1_conv')(x)
    x = layers.BatchNormalization(name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(name=name + '_2_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)

    return x


def create_resnet18(input_shape, num_classes):
    """
    ResNet-18 architecture for CIFAR-10/CIFAR-100

    Architecture:
    - Conv 64 filters 3x3
    - 2x Residual Blocks (64 filters)
    - 2x Residual Blocks (128 filters, stride 2)
    - 2x Residual Blocks (256 filters, stride 2)
    - 2x Residual Blocks (512 filters, stride 2)
    - Global Average Pooling
    - Dense num_classes
    """
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, 3, strides=1, padding='same', name='conv1_conv')(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)

    # Stage 1: 2 blocks, 64 filters
    x = resnet_block(x, 64, name='conv2_block1')
    x = resnet_block(x, 64, name='conv2_block2')

    # Stage 2: 2 blocks, 128 filters
    x = resnet_block(x, 128, stride=2, conv_shortcut=True, name='conv3_block1')
    x = resnet_block(x, 128, name='conv3_block2')

    # Stage 3: 2 blocks, 256 filters
    x = resnet_block(x, 256, stride=2, conv_shortcut=True, name='conv4_block1')
    x = resnet_block(x, 256, name='conv4_block2')

    # Stage 4: 2 blocks, 512 filters
    x = resnet_block(x, 512, stride=2, conv_shortcut=True, name='conv5_block1')
    x = resnet_block(x, 512, name='conv5_block2')

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs, outputs, name='ResNet18')

    return model


def get_trainable_weights(model):
    """
    Extract only trainable weights from model

    Args:
        model: Keras model

    Returns:
        trainable_weights: List of trainable weight arrays
        trainable_indices: Indices of trainable weights in full weight list
    """
    trainable_weights = []
    trainable_indices = []

    all_weights = model.get_weights()
    weight_idx = 0

    for layer in model.layers:
        layer_weights = layer.get_weights()
        num_weights = len(layer_weights)

        if layer.trainable and num_weights > 0:
            # This layer is trainable, collect its weights
            for i in range(num_weights):
                trainable_weights.append(all_weights[weight_idx + i])
                trainable_indices.append(weight_idx + i)

        weight_idx += num_weights

    return trainable_weights, trainable_indices


def set_trainable_weights(model, trainable_weights, trainable_indices):
    """
    Set only trainable weights in model, keeping frozen weights unchanged

    Args:
        model: Keras model
        trainable_weights: List of trainable weight arrays
        trainable_indices: Indices of trainable layers
    """
    all_weights = model.get_weights()

    # Update only trainable weights
    for i, idx in enumerate(trainable_indices):
        if i < len(trainable_weights):
            all_weights[idx] = trainable_weights[i]

    model.set_weights(all_weights)


def copy_model(model, dataset_name):
    """
    Copy model with same architecture and weights

    Args:
        model: Source model
        dataset_name: Dataset name

    Returns:
        model_copy: Copied model
    """
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Create new model with same architecture
        model_copy = tf.keras.models.clone_model(model)

        # Compile the model with appropriate loss
        model_copy.compile(
            optimizer=get_optimizer(dataset_name),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Copy weights
        weights = model.get_weights()
        model_copy.set_weights(weights)

        # Ensure trainability matches
        for layer_orig, layer_copy in zip(model.layers, model_copy.layers):
            layer_copy.trainable = layer_orig.trainable

        return model_copy


def initialize_global_model(dataset_name):
    """
    Initialize global model

    Args:
        dataset_name: 'mnist', 'cifar10', or 'cifar100'

    Returns:
        model: Compiled Keras model
    """
    input_shape = data_shape(dataset_name)

    # Create model
    model = create_model(dataset_name, input_shape)

    # Compile with appropriate loss function
    # Data comes as one-hot encoded, so use categorical_crossentropy
    model.compile(
        optimizer=get_optimizer(dataset_name),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def initialize_edge_models(edge_servers, dataset_name, global_model):
    """
    Initialize edge server models

    Args:
        edge_servers: List of edge servers
        dataset_name: Dataset name
        global_model: Global model to copy from
    """
    input_shape = data_shape(dataset_name)

    for edge in edge_servers:
        if edge.local_model is None:
            edge.local_model = create_model(dataset_name, input_shape)
            edge.local_model.compile(
                optimizer=get_optimizer(dataset_name),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        edge.local_model.set_weights(global_model.get_weights())

        # Ensure trainability matches
        for layer_global, layer_edge in zip(global_model.layers, edge.local_model.layers):
            layer_edge.trainable = layer_global.trainable


def get_optimizer(dataset_name: str):
    """
    Get optimizer based on dataset

    Args:
        dataset_name: Dataset name

    Returns:
        optimizer: Keras optimizer
    """
    if dataset_name in ['mnist']:
        return tf.keras.optimizers.Adam(learning_rate=1e-3)
    elif dataset_name in ['cifar10']:
        # For transfer learning, use lower learning rate
        return tf.keras.optimizers.Adam(learning_rate=1e-4)
    elif dataset_name in ['cifar100']:
        return tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    else:
        return tf.keras.optimizers.Adam(learning_rate=1e-3)