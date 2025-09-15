"""
Script principal pour comparaison FL vs autres types d'apprentissage
Usage: python main.py --dataset mnist --comparison fl_vs_central --epochs 20

Exemples rapides:
  python main.py --dataset mnist --comparison fl_vs_central
  python main.py --dataset cifar --comparison fl_vs_distributed --epochs 50
  python main.py --dataset mnist --comparison all --verbose
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')

from data_loader import load_comparison_dataset
from comparisons import compare_fl_vs_central, compare_fl_vs_distributed, compare_all_methods
from utils import setup_gpu, save_comparison_results, print_comparison_info


def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Comparaison Apprentissage Fédéré vs Autres Méthodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  🔹 FL vs Centralisé sur MNIST:
     python main.py --dataset mnist --comparison fl_vs_central --epochs 20

  🔹 FL vs Distribué sur CIFAR-10:
     python main.py --dataset cifar --comparison fl_vs_distributed --epochs 50

  🔹 Comparaison complète:
     python main.py --dataset mnist --comparison all --epochs 30

  🔹 Mode verbose avec graphiques:
     python main.py --dataset cifar --comparison fl_vs_central --verbose --plot
        """
    )

    # Arguments obligatoires
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'cifar'],
                        help='Dataset à utiliser (mnist: 28x28 N&B, cifar: 32x32 couleur)')

    parser.add_argument('--comparison', type=str, required=True,
                        choices=['fl_vs_central', 'fl_vs_distributed', 'all'],
                        help='Type de comparaison à effectuer')

    # Arguments optionnels
    parser.add_argument('--epochs', type=int, default=20,
                        help='Nombre d\'époques (défaut: 20)')

    parser.add_argument('--clients', type=int, default=5,
                        help='Nombre de clients FL (défaut: 5)')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Taille des lots (défaut: 32)')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Taux d\'apprentissage (défaut: 0.001)')

    parser.add_argument('--iid', action='store_true',
                        help='Distribution IID des données (défaut: non-IID)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU à utiliser (-1 pour CPU, défaut: 0)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire (défaut: 42)')

    parser.add_argument('--verbose', action='store_true',
                        help='Mode verbeux avec détails')

    parser.add_argument('--plot', action='store_true',
                        help='Afficher les graphiques de comparaison')

    return parser.parse_args()


def main():
    """Fonction principale de comparaison"""

    # Parse des arguments
    args = parse_arguments()

    # Configuration initiale
    print("🔄 Démarrage de la comparaison d'apprentissage...")
    print_comparison_info(args)

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
    print(f"📊 Chargement du dataset {args.dataset.upper()}...")
    try:
        data_splits = load_comparison_dataset(
            args.dataset, args.clients, args.iid, args.batch_size
        )
    except Exception as e:
        print(f"❌ Erreur chargement dataset: {e}")
        return

    # Exécuter la comparaison selon le type
    print(f"🔬 Lancement comparaison: {args.comparison}")

    try:
        if args.comparison == 'fl_vs_central':
            results = compare_fl_vs_central(data_splits, args)
        elif args.comparison == 'fl_vs_distributed':
            results = compare_fl_vs_distributed(data_splits, args)
        elif args.comparison == 'all':
            results = compare_all_methods(data_splits, args)

        # Afficher les résultats
        print("\n" + "=" * 60)
        print("📊 RÉSULTATS DE COMPARAISON")
        print("=" * 60)

        for method, metrics in results.items():
            acc = metrics.get('final_accuracy', 0.0)
            time_taken = metrics.get('training_time', 0.0)
            print(f"{method:20}: Accuracy = {acc:.4f}, Temps = {time_taken:.2f}s")

        # Sauvegarder les résultats
        save_comparison_results(results, args)

        print(f"\n📁 Résultats sauvegardés:")
        print(f"    results/ - Fichiers de résultats détaillés")
        print(f"    models/ - Modèles entraînés (.keras)")
        print(f"    visualizations/ - Graphiques de comparaison")

    except Exception as e:
        print(f"❌ Erreur lors de la comparaison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()