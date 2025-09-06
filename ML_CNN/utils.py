"""
Fonctions utilitaires
"""

import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt


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
    print("=" * 50)
    print("🔧 CONFIGURATION D'ENTRAÎNEMENT")
    print("=" * 50)
    print(f"📊 Dataset: {args.dataset.upper()}")
    print(f"🏗️  Modèle: {args.model.upper()}")
    print(f"🔄 Époques: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.lr}")
    print(f"⚡ Momentum: {args.momentum}")
    print(f"🎲 Seed: {args.seed}")
    print("=" * 50)


def save_results(model, history, args, test_loss, test_acc):
    """Sauvegarde selon la structure FL-MIA: results/, models/, visualizations/"""

    # Créer les dossiers comme dans FL-MIA
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Timestamp comme dans FL-MIA
    import datetime
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    # Nom de base pour les fichiers
    base_name = f"{args.dataset}_{args.model}_{args.epochs}ep"

    # 1. Sauvegarder le modèle dans models/
    model_path = os.path.join('models', f"{base_name}.keras")
    model.save(model_path)
    print(f"💾 Modèle sauvegardé: {model_path}")

    # 2. Sauvegarder les results dans results/ (format FL-MIA)
    log_path = os.path.join('results', f"{timestamp}_{base_name}.txt")
    save_training_results(history, args, test_loss, test_acc, log_path)

    # 3. Sauvegarder les visualizations dans visualizations/
    plot_path = os.path.join('visualizations', f"{base_name}_{timestamp}.png")
    plot_training_curves(history, args, base_name, plot_path)


def save_training_results(history, args, test_loss, test_acc, log_path):
    """Sauvegarde les results"""
    try:
        with open(log_path, 'w') as log_file:
            # Format identique à FL-MIA
            log_file.write(
                f"{args.dataset.upper()}, {args.model.upper()}, Centralized Training, batch_size: {args.batch_size}\n")
            log_file.write(f"Train Loss = {history.history['loss']}\n")
            log_file.write(f"Val Loss = {history.history.get('val_loss', [])}\n")
            log_file.write(f"Val Accuracy = {history.history.get('val_accuracy', [])}\n")
            log_file.write(f"Final Test Loss = {test_loss}\n")
            log_file.write(f"Final Test Accuracy = {test_acc}\n")
            log_file.write(f"Training Parameters:\n")
            log_file.write(f"  - Epochs: {args.epochs}\n")
            log_file.write(f"  - Learning Rate: {args.lr}\n")
            log_file.write(f"  - Momentum: {args.momentum}\n")
            log_file.write(f"  - Batch Size: {args.batch_size}\n")
            log_file.write(f"  - Seed: {args.seed}\n")

        print(f"📄 résultas sauvegardés: {log_path}")
    except IOError:
        print("❌ Erreur lors de la sauvegarde des résultats")


def plot_training_curves(history, args, base_name, plot_path):
    """Crée les visualisations comme dans FL-MIA"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Couleurs et style FL-MIA
    epochs = range(1, len(history.history['loss']) + 1)

    # Graphique de perte
    ax1.plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    if 'val_loss' in history.history and history.history['val_loss']:
        ax1.plot(epochs, history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')

    ax1.set_title(f'Model Loss - {args.dataset.upper()} {args.model.upper()}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Graphique de précision
    ax2.plot(epochs, history.history['accuracy'], 'g-', linewidth=2, label='Training Accuracy')
    if 'val_accuracy' in history.history and history.history['val_accuracy']:
        ax2.plot(epochs, history.history['val_accuracy'], 'orange', linewidth=2, label='Validation Accuracy')

    ax2.set_title(f'Model Accuracy - {args.dataset.upper()} {args.model.upper()}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    # Sauvegarder avec haute qualité
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📊 Visualisation sauvegardée: {plot_path}")
