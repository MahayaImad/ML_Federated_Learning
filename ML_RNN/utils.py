"""
Fonctions utilitaires pour RNN avec datasets benchmark
"""

import os
import json
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def check_dependencies(dataset_name):
    """Vérifie les dépendances selon le dataset"""
    missing = []

    if dataset_name in ['stock']:
        try:
            import yfinance
        except ImportError:
            missing.append('yfinance')

    if dataset_name in ['crypto']:
        try:
            import requests
        except ImportError:
            missing.append('requests')

    if dataset_name in ['energy', 'weather', 'housing']:
        try:
            import sklearn
            from sklearn.datasets import fetch_openml
        except ImportError:
            missing.append('scikit-learn')

    # Dépendances communes
    try:
        import pandas
    except ImportError:
        missing.append('pandas')

    return missing


def setup_gpu(gpu_id):
    """Configure l'utilisation du GPU"""
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpu_id == -1:
        print("🖥️  Utilisation du CPU uniquement")
        tf.config.set_visible_devices([], 'GPU')
    elif gpus and gpu_id < len(gpus):
        try:
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            print(f"🎮 Utilisation du GPU: {gpus[gpu_id].name}")
        except RuntimeError as e:
            print(f"❌ Erreur GPU: {e}")
            print("🖥️  Utilisation du CPU par défaut")
    else:
        print(f"⚠️  GPU {gpu_id} non trouvé, utilisation du CPU")


def print_training_info(args):
    """Affiche les informations de configuration"""
    dataset_info = {
        'stock': 'Actions Apple (AAPL) - Yahoo Finance',
        'crypto': 'Bitcoin (BTC) - CoinGecko API',
        'energy': 'UCI Electric Power Consumption',
        'weather': 'Daily Temperature - OpenML/NOAA',
        'housing': 'Boston Housing Time Series'
    }

    print("=" * 60)
    print("🔧 CONFIGURATION D'ENTRAÎNEMENT RNN")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset.upper()}")
    print(f"    Source: {dataset_info.get(args.dataset, 'Unknown')}")
    print(f"🏗️  Modèle: {args.model.upper()}")
    print(f"🔄 Époques: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.lr}")
    print(f"📏 Sequence length: {args.sequence_length}")
    print(f"🧠 Hidden units: {args.hidden_units}")
    print(f"🎭 Dropout: {args.dropout}")
    print(f"🎲 Seed: {args.seed}")
    print("=" * 60)


def save_results(model, history, args, test_loss, test_mae, scaler):
    """Sauvegarde selon la structure FL-MIA avec benchmark info"""

    # Timestamp
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    base_name = f"{args.dataset}_{args.model}_{args.epochs}ep"

    # 1. Sauvegarder le modèle
    model_path = os.path.join('models', f"{base_name}.keras")
    model.save(model_path)
    print(f"💾 Modèle sauvegardé: {model_path}")

    # 2. Sauvegarder les results
    log_path = os.path.join('results', f"{timestamp}_{base_name}.txt")
    save_training_results(history, args, test_loss, test_mae, log_path)

    # 3. Sauvegarder visualisations
    plot_path = os.path.join('visualizations', f"{base_name}_{timestamp}.png")
    plot_training_curves(history, args, base_name, plot_path, model)

    # 4. Sauvegarder le scaler (crucial pour les prédictions)
    import pickle
    scaler_path = os.path.join('models', f"{base_name}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"🔧 Scaler sauvegardé: {scaler_path}")


def save_training_results(history, args, test_loss, test_mae, log_path):
    """Sauvegarde les results au format FL-MIA avec info benchmark"""

    dataset_sources = {
        'stock': 'Yahoo Finance AAPL 5Y',
        'crypto': 'CoinGecko Bitcoin 5Y',
        'energy': 'UCI Household Power Consumption',
        'weather': 'OpenML Daily Temperature',
        'housing': 'Boston Housing Time Series'
    }

    try:
        with open(log_path, 'w') as log_file:
            source = dataset_sources.get(args.dataset, 'Unknown')
            log_file.write(f"{args.dataset.upper()}, {args.model.upper()}, Time Series Benchmark, source: {source}\n")
            log_file.write(f"Batch Size = {args.batch_size}\n")
            log_file.write(f"Sequence Length = {args.sequence_length}\n")
            log_file.write(f"Train Loss = {history.history['loss']}\n")
            log_file.write(f"Val Loss = {history.history.get('val_loss', [])}\n")
            log_file.write(f"Train MAE = {history.history.get('mae', [])}\n")
            log_file.write(f"Val MAE = {history.history.get('val_mae', [])}\n")
            log_file.write(f"Final Test Loss (MSE) = {test_loss}\n")
            log_file.write(f"Final Test MAE = {test_mae}\n")
            log_file.write(f"Training Parameters:\n")
            log_file.write(f"  - Epochs Completed: {len(history.history['loss'])}\n")
            log_file.write(f"  - Max Epochs: {args.epochs}\n")
            log_file.write(f"  - Learning Rate: {args.lr}\n")
            log_file.write(f"  - Sequence Length: {args.sequence_length}\n")
            log_file.write(f"  - Hidden Units: {args.hidden_units}\n")
            log_file.write(f"  - Dropout Rate: {args.dropout}\n")
            log_file.write(f"  - Batch Size: {args.batch_size}\n")
            log_file.write(f"  - Random Seed: {args.seed}\n")
            log_file.write(f"Dataset Information:\n")
            log_file.write(f"  - Benchmark Source: {source}\n")
            log_file.write(f"  - Data Split: 70% train, 15% val, 15% test\n")
            log_file.write(f"  - Normalization: MinMaxScaler\n")

        print(f"📄 results sauvegardés: {log_path}")
    except IOError:
        print("❌ Erreur lors de la sauvegarde des results")


def plot_training_curves(history, args, base_name, plot_path, model):
    """Crée les visualisations avec info benchmark"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    epochs = range(1, len(history.history['loss']) + 1)

    # 1. Loss curves
    ax1.plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    if 'val_loss' in history.history:
        ax1.plot(epochs, history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title(f'Model Loss - {args.dataset.upper()} {args.model.upper()}', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale pour mieux voir l'évolution

    # 2. MAE curves
    if 'mae' in history.history:
        ax2.plot(epochs, history.history['mae'], 'g-', linewidth=2, label='Training MAE')
    if 'val_mae' in history.history:
        ax2.plot(epochs, history.history['val_mae'], 'orange', linewidth=2, label='Validation MAE')
    ax2.set_title('Mean Absolute Error Evolution', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Dataset & Training info
    dataset_info = {
        'stock': 'Apple Stock (AAPL)\nYahoo Finance - 5 years',
        'crypto': 'Bitcoin (BTC-USD)\nCoinGecko API - 5 years',
        'energy': 'Household Power Consumption\nUCI ML Repository',
        'weather': 'Daily Temperature\nOpenML/NOAA Data',
        'housing': 'Boston Housing Prices\nTime Series Simulation'
    }

    ax3.text(0.05, 0.95, f'📊 Dataset: {dataset_info.get(args.dataset, args.dataset)}',
             fontsize=12, transform=ax3.transAxes, va='top', fontweight='bold')
    ax3.text(0.05, 0.80, f'🏗️  Model: {args.model.upper()}', fontsize=11, transform=ax3.transAxes, va='top')
    ax3.text(0.05, 0.70, f'📏 Sequence Length: {args.sequence_length}', fontsize=11, transform=ax3.transAxes, va='top')
    ax3.text(0.05, 0.60, f'🧠 Hidden Units: {args.hidden_units}', fontsize=11, transform=ax3.transAxes, va='top')
    ax3.text(0.05, 0.50, f'🎭 Dropout: {args.dropout}', fontsize=11, transform=ax3.transAxes, va='top')
    ax3.text(0.05, 0.40, f'📈 Learning Rate: {args.lr}', fontsize=11, transform=ax3.transAxes, va='top')
    ax3.text(0.05, 0.30, f'📦 Batch Size: {args.batch_size}', fontsize=11, transform=ax3.transAxes, va='top')
    ax3.text(0.05, 0.20, f'🔄 Epochs: {len(epochs)}/{args.epochs}', fontsize=11, transform=ax3.transAxes, va='top')
    ax3.text(0.05, 0.10, f'🎲 Seed: {args.seed}', fontsize=11, transform=ax3.transAxes, va='top')
    ax3.set_title('Training Configuration', fontweight='bold', fontsize=14)
    ax3.axis('off')

    # 4. Model architecture & performance
    architectures = {
        'lstm': 'LSTM(n) → Dropout → LSTM(n/2)\n→ Dropout → Dense(25) → Dense(1)',
        'gru': 'GRU(n) → Dropout → GRU(n/2)\n→ Dropout → Dense(25) → Dense(1)',
        'rnn': 'SimpleRNN(n) → Dropout → SimpleRNN(n/2)\n→ Dropout → Dense(25) → Dense(1)'
    }

    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history.get('val_loss', [0])[-1] if history.history.get('val_loss') else 0

    ax4.text(0.05, 0.90, 'Model Architecture:', fontsize=14, fontweight='bold', transform=ax4.transAxes, va='top')
    ax4.text(0.05, 0.75, architectures.get(args.model, 'RNN Architecture'),
             fontsize=10, transform=ax4.transAxes, va='top', family='monospace')
    ax4.text(0.05, 0.50, f'Parameters: ~{model.count_params():,}',
             fontsize=12, transform=ax4.transAxes, va='top')
    ax4.text(0.05, 0.35, f'Final Train Loss: {final_train_loss:.6f}',
             fontsize=11, transform=ax4.transAxes, va='top')
    ax4.text(0.05, 0.25, f'Final Val Loss: {final_val_loss:.6f}',
             fontsize=11, transform=ax4.transAxes, va='top')

    # Early stopping info si applicable
    if len(epochs) < args.epochs:
        ax4.text(0.05, 0.10, f'⏹️  Early Stopping at epoch {len(epochs)}',
                 fontsize=10, transform=ax4.transAxes, va='top', color='red')

    ax4.set_title('Model Info & Performance', fontweight='bold', fontsize=14)
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📊 Visualisation sauvegardée: {plot_path}")


def plot_prediction_sample(model, test_data, scaler, save_path):
    """Crée un graphique des prédictions vs vraies valeurs (optionnel)"""

    # Prendre un échantillon de test
    for batch_x, batch_y in test_data.take(1):
        predictions = model.predict(batch_x)

        # Dénormaliser si possible (approximation)
        true_values = batch_y.numpy()[:50]  # Premier batch, 50 échantillons
        pred_values = predictions[:50, 0]

        plt.figure(figsize=(15, 6))

        x_axis = range(len(true_values))
        plt.plot(x_axis, true_values, 'b-', label='Vraies valeurs', linewidth=2)
        plt.plot(x_axis, pred_values, 'r--', label='Prédictions', linewidth=2)

        plt.title('Échantillon de Prédictions vs Valeurs Réelles', fontsize=14, fontweight='bold')
        plt.xlabel('Échantillons de test')
        plt.ylabel('Valeur normalisée')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        break