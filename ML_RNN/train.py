"""
Script principal pour l'entraînement de modèles RNN sur séries temporelles benchmark
Usage: python train.py --dataset stock --model lstm --epochs 50

Exemples rapides:
  python train.py --dataset stock --model lstm
  python train.py --dataset crypto --model gru --epochs 100
  python train.py --dataset energy --model lstm --sequence_length 60
"""

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import create_rnn_model
from data_loader import load_timeseries_dataset
from utils  import setup_gpu, save_results, print_training_info, check_dependencies


def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Entraînement de modèles RNN sur séries temporelles benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

     Actions Apple via Yahoo Finance:
     python train.py --dataset stock --model lstm --epochs 50

     Bitcoin via CoinGecko:
     python train.py --dataset crypto --model gru --epochs 100

     Consommation énergétique UCI:
     python train.py --dataset energy --model lstm --sequence_length 72

     Températures journalières:
     python train.py --dataset weather --model gru --epochs 75

     Prix immobilier Boston:
     python train.py --dataset housing --model lstm --epochs 60
        """
    )

    # Arguments obligatoires
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['stock', 'crypto', 'energy', 'weather', 'housing'],
                        help=' Dataset benchmark (stock=AAPL, crypto=BTC, energy=UCI, weather=DailyTemp, housing=Boston)')

    parser.add_argument('--model', type=str, required=True,
                        choices=['lstm', 'gru', 'rnn'],
                        help='  Type de RNN (lstm, gru, rnn)')

    # Arguments optionnels
    parser.add_argument('--epochs', type=int, default=50,
                        help='   Nombre d\'époques (défaut: 50)')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='   Taille des lots (défaut: 32)')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='   Taux d\'apprentissage (défaut: 0.001)')

    parser.add_argument('--sequence_length', type=int, default=50,
                        help='   Longueur des séquences (défaut: 50)')

    parser.add_argument('--hidden_units', type=int, default=50,
                        help='   Nombre d\'unités cachées (défaut: 50)')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='   Taux de dropout (défaut: 0.2)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='   GPU à utiliser (-1 pour CPU, défaut: 0)')

    parser.add_argument('--seed', type=int, default=42,
                        help='   Graine aléatoire (défaut: 42)')

    parser.add_argument('--verbose', action='store_true',
                        help='   Mode verbeux')

    return parser.parse_args()


def main():
    """Fonction principale d'entraînement"""

    # Parse des arguments
    args = parse_arguments()

    # Vérifier les dépendances
    missing_deps = check_dependencies(args.dataset)
    if missing_deps:
        print(f"  Dépendances manquantes: {', '.join(missing_deps)}")
        print(f"  Installez avec: pip install {' '.join(missing_deps)}")
        return

    # Configuration initiale
    print("  Démarrage de l'entraînement RNN...")
    print_training_info(args)

    # Setup GPU
    setup_gpu(args.gpu)

    # Fixer les graines
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Créer la structure de dossiers
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Charger les données
    print(f"  Chargement du dataset {args.dataset.upper()}...")
    try:
        train_data, val_data, test_data, scaler, feature_dim = load_timeseries_dataset(
            args.dataset, args.sequence_length, args.batch_size
        )
    except Exception as e:
        print(f"  Erreur chargement dataset: {e}")
        return

    # Créer le modèle
    print(f"  Construction du modèle {args.model.upper()}...")
    model = create_rnn_model(
        model_type=args.model,
        sequence_length=args.sequence_length,
        feature_dim=feature_dim,
        hidden_units=args.hidden_units,
        dropout=args.dropout
    )

    if args.verbose:
        model.summary()

    # Compiler le modèle
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
    )

    # Entraînement
    print(f" Début de l'entraînement ({args.epochs} époques)...")

    history = model.fit(
        train_data,
        epochs=args.epochs,
        validation_data=val_data,
        callbacks=[early_stopping, reduce_lr],
        verbose=1 if args.verbose else 1
    )

    # Évaluation finale
    print(" Évaluation finale...")
    test_loss, test_mae = model.evaluate(test_data, verbose=0)

    print(f"✅ Entraînement terminé!")
    print(f" Test Loss (MSE): {test_loss:.6f}")
    print(f" Test MAE: {test_mae:.6f}")

    # Sauvegarder les résultats
    save_results(model, history, args, test_loss, test_mae, scaler)

    print(f"\n📁 Résultats sauvegardés:")
    print(f"    results/ - Fichiers de results détaillés")
    print(f"    models/ - Modèles entraînés (.keras)")
    print(f"    visualizations/ - Graphiques et prédictions")


if __name__ == '__main__':
    main()