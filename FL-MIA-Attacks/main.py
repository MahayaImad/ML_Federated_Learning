"""
Script principal pour attaques MIA sur apprentissage fédéré
Usage: python main.py --dataset mnist --attack shadow_model --risk_level high

Exemples rapides:
  python main.py --dataset mnist --attack shadow_model --risk_level medium
  python main.py --dataset cifar --attack threshold --risk_level high --epochs 50
  python main.py --dataset mnist --attack all --verbose
"""

import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')

from data_loader import load_mia_dataset
from attacks import run_shadow_model_attack, run_threshold_attack, run_all_attacks
from utils import setup_gpu, save_attack_results, print_attack_info


def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Attaques MIA (Membership Inference) sur Apprentissage Fédéré',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  🔹 Attaque Shadow Model (risque moyen):
     python main.py --dataset mnist --attack shadow_model --risk_level medium

  🔹 Attaque Threshold (risque élevé):
     python main.py --dataset cifar --attack threshold --risk_level high --epochs 50

  🔹 Toutes les attaques:
     python main.py --dataset mnist --attack all --risk_level low --verbose

  🔹 Évaluation complète avec graphiques:
     python main.py --dataset cifar --attack all --risk_level medium --plot --verbose
        """
    )

    # Arguments obligatoires
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mnist', 'cifar'],
                        help='Dataset à utiliser (mnist: 28x28 N&B, cifar: 32x32 couleur)')

    parser.add_argument('--attack', type=str, required=True,
                        choices=['shadow_model', 'threshold', 'gradient_based', 'all'],
                        help='Type d\'attaque MIA à effectuer')

    parser.add_argument('--risk_level', type=str, required=True,
                        choices=['low', 'medium', 'high'],
                        help='Niveau de risque de l\'attaque (low: faible, medium: moyen, high: élevé)')

    # Arguments optionnels
    parser.add_argument('--epochs', type=int, default=30,
                        help='Nombre d\'époques d\'entraînement (défaut: 30)')

    parser.add_argument('--clients', type=int, default=5,
                        help='Nombre de clients FL (défaut: 5)')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Taille des lots (défaut: 32)')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Taux d\'apprentissage (défaut: 0.001)')

    parser.add_argument('--target_samples', type=int, default=1000,
                        help='Nombre d\'échantillons cibles pour MIA (défaut: 1000)')

    parser.add_argument('--shadow_models', type=int, default=5,
                        help='Nombre de modèles shadow (défaut: 5)')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU à utiliser (-1 pour CPU, défaut: 0)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire (défaut: 42)')

    parser.add_argument('--verbose', action='store_true',
                        help='Mode verbeux avec détails')

    parser.add_argument('--plot', action='store_true',
                        help='Afficher les graphiques d\'analyse')

    return parser.parse_args()


def main():
    """Fonction principale d'attaque MIA"""

    # Parse des arguments
    args = parse_arguments()

    # Configuration initiale
    print("🎯 Démarrage des attaques MIA...")
    print_attack_info(args)

    # Setup GPU
    setup_gpu(args.gpu)

    # Fixer les graines
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Créer la structure de dossiers
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Charger les données pour MIA
    print(f"📊 Préparation dataset pour MIA: {args.dataset.upper()}...")
    try:
        mia_data = load_mia_dataset(
            args.dataset, args.clients, args.target_samples, args.batch_size
        )
    except Exception as e:
        print(f"❌ Erreur préparation dataset: {e}")
        return

    # Exécuter les attaques selon le type
    print(f"🎯 Lancement attaque: {args.attack} (risque: {args.risk_level})")

    try:
        if args.attack == 'shadow_model':
            results = run_shadow_model_attack(mia_data, args)
        elif args.attack == 'threshold':
            results = run_threshold_attack(mia_data, args)
        elif args.attack == 'gradient_based':
            results = run_gradient_based_attack(mia_data, args)
        elif args.attack == 'all':
            results = run_all_attacks(mia_data, args)

        # Afficher les résultats d'attaque
        print("\n" + "=" * 60)
        print("🎯 RÉSULTATS D'ATTAQUE MIA")
        print("=" * 60)

        for attack_name, metrics in results.items():
            accuracy = metrics.get('attack_accuracy', 0.0)
            precision = metrics.get('precision', 0.0)
            recall = metrics.get('recall', 0.0)
            auc_score = metrics.get('auc_score', 0.0)

            print(f"{attack_name:20}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  AUC Score: {auc_score:.4f}")
            print(f"  Niveau de risque: {args.risk_level.upper()}")

        # Évaluation du risque global
        avg_accuracy = np.mean([m.get('attack_accuracy', 0) for m in results.values()])
        risk_assessment = assess_privacy_risk(avg_accuracy, args.risk_level)

        print(f"\n🔍 ÉVALUATION DU RISQUE:")
        print(f"  Accuracy moyenne des attaques: {avg_accuracy:.4f}")
        print(f"  Niveau de risque configuré: {args.risk_level.upper()}")
        print(f"  Évaluation: {risk_assessment}")

        # Sauvegarder les résultats
        save_attack_results(results, args, risk_assessment)

        print(f"\n📁 Résultats sauvegardés:")
        print(f"    results/ - Rapports d'attaque détaillés")
        print(f"    models/ - Modèles cibles et shadow (.keras)")
        print(f"    visualizations/ - Courbes ROC et métriques")

    except Exception as e:
        print(f"❌ Erreur lors de l'attaque: {e}")
        import traceback
        traceback.print_exc()


def assess_privacy_risk(attack_accuracy, risk_level):
    """Évalue le risque de confidentialité"""
    if attack_accuracy > 0.7:
        return "🔴 RISQUE ÉLEVÉ - Modèle vulnérable aux attaques MIA"
    elif attack_accuracy > 0.6:
        return "🟡 RISQUE MOYEN - Failles de confidentialité détectées"
    elif attack_accuracy > 0.55:
        return "🟢 RISQUE FAIBLE - Protection acceptable"
    else:
        return "✅ RISQUE MINIMAL - Bonne protection de la confidentialité"


if __name__ == '__main__':
    main()