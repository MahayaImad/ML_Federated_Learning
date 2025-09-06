"""
Architectures RNN optimisées pour datasets benchmark
"""

import tensorflow as tf
from tensorflow.keras import layers


def create_rnn_model(model_type, sequence_length, feature_dim, hidden_units=50, dropout=0.2):
    """
    Crée un modèle RNN optimisé pour datasets benchmark

    Args:
        model_type: 'lstm', 'gru', ou 'rnn'
        sequence_length: longueur des séquences temporelles
        feature_dim: nombre de features par timestep
        hidden_units: nombre d'unités cachées (couche 1)
        dropout: taux de dropout pour régularisation

    Returns:
        model: modèle Keras optimisé pour time series benchmark
    """

    if model_type == 'lstm':
        return create_lstm_benchmark(sequence_length, feature_dim, hidden_units, dropout)
    elif model_type == 'gru':
        return create_gru_benchmark(sequence_length, feature_dim, hidden_units, dropout)
    elif model_type == 'rnn':
        return create_simple_rnn_benchmark(sequence_length, feature_dim, hidden_units, dropout)

    raise ValueError(f"Type de modèle non supporté: {model_type}")


def create_lstm_benchmark(sequence_length, feature_dim, hidden_units, dropout):
    """
    LSTM optimisé pour datasets financiers/énergétiques
    Architecture: LSTM → Dropout → LSTM → Dropout → Dense → Output
    """
    model = tf.keras.Sequential([
        # Première couche LSTM avec return_sequences=True
        layers.LSTM(
            hidden_units,
            return_sequences=True,
            input_shape=(sequence_length, feature_dim),
            recurrent_dropout=dropout * 0.5  # Dropout récurrent léger
        ),
        layers.Dropout(dropout),

        # Deuxième couche LSTM (plus petite)
        layers.LSTM(
            hidden_units // 2,
            return_sequences=False,
            recurrent_dropout=dropout * 0.5
        ),
        layers.Dropout(dropout),

        # Couches denses pour la régression
        layers.Dense(25, activation='relu'),
        layers.Dropout(dropout * 0.5),  # Dropout plus léger pour les couches denses
        layers.Dense(1, activation='linear')  # Sortie linéaire pour régression

    ], name=f'LSTM_Benchmark_{hidden_units}')

    return model


def create_gru_benchmark(sequence_length, feature_dim, hidden_units, dropout):
    """
    GRU optimisé - Plus rapide que LSTM, souvent comparable
    Architecture: GRU → Dropout → GRU → Dropout → Dense → Output
    """
    model = tf.keras.Sequential([
        layers.GRU(
            hidden_units,
            return_sequences=True,
            input_shape=(sequence_length, feature_dim),
            recurrent_dropout=dropout * 0.5
        ),
        layers.Dropout(dropout),

        layers.GRU(
            hidden_units // 2,
            return_sequences=False,
            recurrent_dropout=dropout * 0.5
        ),
        layers.Dropout(dropout),

        layers.Dense(25, activation='relu'),
        layers.Dropout(dropout * 0.5),
        layers.Dense(1, activation='linear')

    ], name=f'GRU_Benchmark_{hidden_units}')

    return model


def create_simple_rnn_benchmark(sequence_length, feature_dim, hidden_units, dropout):
    """
    SimpleRNN - Baseline pour comparaison
    Plus simple mais peut souffrir du vanishing gradient
    """
    model = tf.keras.Sequential([
        layers.SimpleRNN(
            hidden_units,
            return_sequences=True,
            input_shape=(sequence_length, feature_dim)
        ),
        layers.Dropout(dropout),

        layers.SimpleRNN(
            hidden_units // 2,
            return_sequences=False
        ),
        layers.Dropout(dropout),

        layers.Dense(25, activation='relu'),
        layers.Dropout(dropout * 0.5),
        layers.Dense(1, activation='linear')

    ], name=f'SimpleRNN_Benchmark_{hidden_units}')

    return model


def create_bidirectional_lstm_benchmark(sequence_length, feature_dim, hidden_units, dropout):
    """
    LSTM Bidirectionnel - Pour datasets avec dépendances futures
    Plus coûteux mais parfois plus performant
    """
    model = tf.keras.Sequential([
        layers.Bidirectional(
            layers.LSTM(hidden_units, return_sequences=True),
            input_shape=(sequence_length, feature_dim)
        ),
        layers.Dropout(dropout),

        layers.Bidirectional(
            layers.LSTM(hidden_units // 2, return_sequences=False)
        ),
        layers.Dropout(dropout),

        layers.Dense(25, activation='relu'),
        layers.Dropout(dropout * 0.5),
        layers.Dense(1, activation='linear')

    ], name=f'BiLSTM_Benchmark_{hidden_units}')

    return model