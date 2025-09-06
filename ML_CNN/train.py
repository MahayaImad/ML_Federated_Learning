"""
Script principal pour l'entraînement de modèles CNN
Usage: python train.py --dataset mnist --model cnn --epochs 50 --gpu 0
"""

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import create_model
from data_loader import load_dataset
from utils import setup_gpu, save_results, print_training_info


def parse_arguments():
    """Parse les arguments de ligne de commande avec des explications détaillées"""
    parser = argparse.ArgumentParser(
        description='Entraînement de modèles CNN sur MNIST/CIFAR-10',
        formatter_class=argparse.RawDescriptionHelpFormatter, # Pour préserver le format exact
        epilog="""
Exemples d'utilisation:

  🔹 MNIST avec CNN (recommandé pour débuter):
     python train.py --dataset mnist --model cnn --epochs 10 --gpu 0

  🔹 CIFAR-10 avec CNN (plus avancé):
     python train.py --dataset cifar --model cnn --epochs 50 --lr 0.001

  🔹 MNIST avec MLP simple:
     python train.py --dataset mnist --model mlp --epochs 20 --batch_size 128

  🔹 Entraînement CPU seulement:
     python train.py --dataset mnist --model cnn --epochs 10 --gpu -1

  🔹 Configuration complète:
     python train.py --dataset cifar --model cnn --epochs 100 --lr 0.01 --batch_size 64 --gpu 0
        """
    )

    # Arguments obligatoires
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'cifar'],
                        help='Dataset à utiliser (mnist: images 28x28 N&B, cifar: images 32x32 couleur)')

    parser.add_argument('--model', type=str, required=True,
                        choices=['cnn', 'mlp'],
                        help='Architecture du modèle (cnn: réseau convolutionnel, mlp: réseau dense)')

    # Arguments optionnels avec valeurs par défaut
    parser.add_argument('--epochs', type=int, default=10,
                        help='Nombre d\'époques d\'entraînement (défaut: 10)')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Taille des lots (défaut: 64, augmenter si beaucoup de RAM)')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Taux d\'apprentissage (défaut: 0.01, diminuer si perte oscille)')

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum pour l\'optimiseur SGD (défaut: 0.9)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU à utiliser (défaut: 0, -1 pour CPU uniquement)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire pour reproductibilité (défaut: 42)')

    parser.add_argument('--verbose', action='store_true',
                        help='Mode verbeux avec plus de détails')

    return parser.parse_args()


def main():
    """Fonction principale d'entraînement"""

    # Parse des arguments
    args = parse_arguments()

    # Configuration initiale
    print("Démarrage de l'entraînement...")
    print_training_info(args)

    # Setup GPU
    setup_gpu(args.gpu)

    # Fixer les graines pour reproductibilité
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Créer la structure de dossiers FL-MIA
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Charger les données
    print(f"Chargement du dataset {args.dataset.upper()}...")
    train_data, test_data, input_shape, num_classes = load_dataset(args.dataset, args.batch_size)

    # Créer le modèle
    print(f"Construction du modèle {args.model.upper()}...")
    model = create_model(args.model, args.dataset, input_shape, num_classes)

    if args.verbose:
        model.summary()

    # Compiler le modèle
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=args.momentum)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entraînement
    print(f"Début de l'entraînement ({args.epochs} époques)...")

    history = model.fit(
        train_data,
        epochs=args.epochs,
        validation_data=test_data,
        verbose=1 if args.verbose else 1
    )

    # Évaluation finale
    print("Évaluation finale...")
    test_loss, test_acc = model.evaluate(test_data, verbose=0)

    print(f"✅ Entraînement terminé!")
    print(f"Accuracy finale: {test_acc * 100:.2f}%")
    print(f"Perte finale: {test_loss:.4f}")

    # Sauvegarder les résultats (format FL-MIA)
    save_results(model, history, args, test_loss, test_acc)

    # Afficher la structure comme FL-MIA
    print(f"\n📁 Résultats sauvegardés:")
    print(f"      results/ - Fichiers de résultas détaillés")
    print(f"      models/ - Modèles entraînés (.keras)")
    print(f"      visualizations/ - Graphiques et courbes")


if __name__ == '__main__':
    main()