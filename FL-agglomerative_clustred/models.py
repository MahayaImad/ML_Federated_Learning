"""
Modèles pour comparaisons FL vs autres méthodes
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from data_preparation import number_classes, data_shape

def create_model(dataset_name, input_shape):
    """
    Create a standard model for comparison

    Args:
        dataset_name: 'mnist', 'cifar10' or 'cifar100'
        input_shape: (28, 28, 1)
        num_classes: 10

    Returns:
        model: Keras model
    """
    num_classes = number_classes()

    if dataset_name == 'mnist':
        return create_lenet5(input_shape, num_classes)
    elif dataset_name == 'cifar10' or dataset_name == 'cifar100':
        return create_resnet18(input_shape, num_classes)
    else:
        raise ValueError(f" Non supported Dataset: {dataset_name}")


def create_lenet5(input_shape, num_classes):
    """
    LeNet-5 architecture for MNIST
    Original Architecture :
    - Conv 6 filters 5x5
    - AvgPool 2x2
    - Conv 16 filters 5x5
    - AvgPool 2x2
    - Flatten
    - Dense 120
    - Dense 84
    - Dense num_classes
    """
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        # C1: Convolutional Layer
        layers.Conv2D(6, kernel_size=(5, 5), strides=1, activation='tanh', padding='same', name='C1'),
        # S2: Subsampling Layer (Average Pooling)
        layers.AveragePooling2D(pool_size=(2, 2), strides=2, name='S2'),
        # C3: Convolutional Layer
        layers.Conv2D(16, kernel_size=(5, 5), strides=1, activation='tanh', padding='valid', name='C3'),
        # S4: Subsampling Layer
        layers.AveragePooling2D(pool_size=(2, 2), strides=2, name='S4'),
        # Flatten
        layers.Flatten(),
        # C5: Fully Connected Layer
        layers.Dense(120, activation='tanh', name='C5'),
        # F6: Fully Connected Layer
        layers.Dense(84, activation='tanh', name='F6'),
        # Output Layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='LeNet5')
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


def copy_model(model, learning_rate=0.001):

    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Créer un nouveau modèle avec la même architecture
        model_copy = tf.keras.models.clone_model(model)

        # Compiler le modèle
        model_copy.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Copier les poids en s'assurant qu'ils sont sur le bon device
        weights = model.get_weights()
        model_copy.set_weights(weights)

        return model_copy


def initialize_global_model(dataset_name, learning_rate=0.001):

    input_shape = data_shape(dataset_name)

    # Create model
    model = create_model(dataset_name, input_shape)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def initialize_edge_models(edge_servers, dataset_name, global_model, learning_rate=0.001):

    input_shape = data_shape(dataset_name)

    for edge in edge_servers:
        if edge.local_model is None:
            edge.local_model = create_model(dataset_name, input_shape)
            edge.local_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        edge.local_model.set_weights(global_model.get_weights())
