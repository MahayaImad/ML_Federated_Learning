"""
Utilitaires pour attaques MIA
"""
import os
import json
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


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


def print_attack_info(args):
    """Affiche les informations de configuration d'attaque"""
    print("=" * 60)
    print("🎯 CONFIGURATION D'ATTAQUE MIA")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset.upper()}")
    print(f"🎯 Type d'attaque: {args.attack}")
    print(f"⚠️  Niveau de risque: {args.risk_level.upper()}")
    print(f"👥 Nombre de clients: {args.clients}")
    print(f"🎯 Échantillons cibles: {args.target_samples}")
    if args.attack in ['shadow_model', 'all']:
        print(f"👤 Modèles shadow: {args.shadow_models}")
    print(f"🔄 Époques: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.lr}")
    print("=" * 60)


def save_attack_results(results, args, risk_assessment):
    """Sauvegarde les résultats d'attaque MIA"""
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    # Nom du fichier basé sur l'attaque
    attack_types = {
        'shadow_model': 'ShadowModel',
        'threshold': 'Threshold',
        'gradient_based': 'GradientBased',
        'all': 'AllAttacks'
    }

    base_name = f"{args.dataset}_{attack_types.get(args.attack, 'attack')}_{args.risk_level}_{args.epochs}ep"

    # 1. Sauvegarder rapport détaillé
    log_path = os.path.join('results', f"{timestamp}_{base_name}.txt")
    save_attack_report(results, args, risk_assessment, log_path)

    # 2. Sauvegarder JSON structuré
    json_path = os.path.join('results', f"{timestamp}_{base_name}.json")
    save_attack_json(results, args, risk_assessment, json_path)

    # 3. Créer visualisations
    plot_path = os.path.join('visualizations', f"{base_name}_{timestamp}.png")
    plot_attack_results(results, args, risk_assessment, plot_path)

    print(f"💾 Rapport sauvegardé: {log_path}")
    print(f"📊 Visualisations sauvegardées: {plot_path}")


def save_attack_report(results, args, risk_assessment, log_path):
    """Sauvegarde le rapport détaillé d'attaque"""
    try:
        with open(log_path, 'w') as f:
            f.write("RAPPORT D'ATTAQUE MIA (Membership Inference Attack)\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Dataset: {args.dataset.upper()}\n")
            f.write(f"Type d'attaque: {args.attack}\n")
            f.write(f"Niveau de risque: {args.risk_level.upper()}\n")
            f.write(f"Configuration FL: {args.clients} clients, {args.epochs} époques\n")
            f.write(f"Échantillons cibles: {args.target_samples}\n")
            if args.attack in ['shadow_model', 'all']:
                f.write(f"Modèles shadow: {args.shadow_models}\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write("\n" + "=" * 60 + "\n\n")

            f.write("RÉSULTATS DES ATTAQUES:\n")
            f.write("-" * 30 + "\n\n")

            for attack_name, metrics in results.items():
                f.write(f"Attaque: {attack_name}\n")
                f.write(f"  Attack Accuracy: {metrics.get('attack_accuracy', 0.0):.6f}\n")
                f.write(f"  Precision: {metrics.get('precision', 0.0):.6f}\n")
                f.write(f"  Recall: {metrics.get('recall', 0.0):.6f}\n")
                f.write(f"  AUC Score: {metrics.get('auc_score', 0.0):.6f}\n")
                f.write(f"  Temps d'attaque: {metrics.get('attack_time', 0.0):.2f}s\n")

                if 'optimal_threshold' in metrics:
                    f.write(f"  Seuil optimal: {metrics['optimal_threshold']:.6f}\n")
                if 'shadow_models_used' in metrics:
                    f.write(f"  Modèles shadow utilisés: {metrics['shadow_models_used']}\n")
                if 'error' in metrics:
                    f.write(f"  ⚠️ Erreur: {metrics['error']}\n")

                f.write("\n")

            f.write("ÉVALUATION DU RISQUE:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{risk_assessment}\n\n")

            # Recommandations de sécurité
            f.write("RECOMMANDATIONS DE SÉCURITÉ:\n")
            f.write("-" * 30 + "\n")

            avg_accuracy = np.mean([m.get('attack_accuracy', 0) for m in results.values()])

            if avg_accuracy > 0.7:
                f.write("🔴 URGENT - Risque élevé détecté:\n")
                f.write("  - Implémenter la confidentialité différentielle\n")
                f.write("  - Augmenter le dropout et la régularisation\n")
                f.write("  - Réduire le nombre de tours de communication\n")
                f.write("  - Considérer l'agrégation sécurisée\n")
            elif avg_accuracy > 0.6:
                f.write("🟡 ATTENTION - Risque modéré:\n")
                f.write("  - Ajouter du bruit aux gradients\n")
                f.write("  - Augmenter la taille des lots\n")
                f.write("  - Limiter les informations partagées\n")
            else:
                f.write("✅ ACCEPTABLE - Risque contrôlé:\n")
                f.write("  - Maintenir les défenses actuelles\n")
                f.write("  - Surveiller les métriques de confidentialité\n")

    except IOError as e:
        print(f"❌ Erreur sauvegarde rapport: {e}")


def save_attack_json(results, args, risk_assessment, json_path):
    """Sauvegarde au format JSON structuré"""
    try:
        json_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'dataset': args.dataset,
                'attack_type': args.attack,
                'risk_level': args.risk_level,
                'clients': args.clients,
                'epochs': args.epochs,
                'target_samples': args.target_samples,
                'shadow_models': getattr(args, 'shadow_models', 0),
                'batch_size': args.batch_size,
                'learning_rate': args.lr
            },
            'attack_results': results,
            'risk_assessment': risk_assessment,
            'privacy_metrics': calculate_privacy_metrics(results)
        }

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)

    except Exception as e:
        print(f"❌ Erreur sauvegarde JSON: {e}")


def plot_attack_results(results, args, risk_assessment, plot_path):
    """Crée les visualisations d'attaque MIA"""

    if not results:
        print("⚠️ Aucun résultat d'attaque à visualiser")
        return

    # Préparer les données
    attack_names = list(results.keys())
    accuracies = [results[attack].get('attack_accuracy', 0.0) for attack in attack_names]
    precisions = [results[attack].get('precision', 0.0) for attack in attack_names]
    recalls = [results[attack].get('recall', 0.0) for attack in attack_names]
    aucs = [results[attack].get('auc_score', 0.0) for attack in attack_names]

    # Créer les sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Couleurs selon le niveau de risque
    risk_colors = {
        'low': '#2E8B57',  # Vert
        'medium': '#FF8C00',  # Orange
        'high': '#DC143C'  # Rouge
    }
    color = risk_colors.get(args.risk_level, '#4169E1')

    # 1. Attack Accuracy
    bars1 = axes[0, 0].bar(attack_names, accuracies, color=color, alpha=0.7)
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Baseline (50%)')
    axes[0, 0].set_title(f'Attack Accuracy - {args.dataset.upper()} ({args.risk_level.upper()})',
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Ajouter les valeurs sur les barres
    for bar, acc in zip(bars1, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Métriques de performance
    x_pos = np.arange(len(attack_names))
    width = 0.25

    bars2 = axes[0, 1].bar(x_pos - width, precisions, width, label='Precision', color='#2E8B57', alpha=0.7)
    bars3 = axes[0, 1].bar(x_pos, recalls, width, label='Recall', color='#FF8C00', alpha=0.7)
    bars4 = axes[0, 1].bar(x_pos + width, aucs, width, label='AUC', color='#DC143C', alpha=0.7)

    axes[0, 1].set_title('Métriques de Performance MIA', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(attack_names, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Évaluation du risque
    axes[1, 0].axis('off')

    # Graphique de risque circulaire
    avg_accuracy = np.mean(accuracies)
    risk_level_num = {'low': 1, 'medium': 2, 'high': 3}[args.risk_level]

    # Cercle de risque
    circle_colors = ['#2E8B57', '#FF8C00', '#DC143C']
    circle_sizes = [0.3, 0.6, 0.9]
    risk_labels = ['Faible', 'Moyen', 'Élevé']

    for i, (size, col, label) in enumerate(zip(circle_sizes, circle_colors, risk_labels)):
        circle = plt.Circle((0.5, 0.5), size, color=col, alpha=0.3, transform=axes[1, 0].transAxes)
        axes[1, 0].add_patch(circle)
        axes[1, 0].text(0.5, 0.5 + size + 0.05, label, ha='center', va='center',
                        transform=axes[1, 0].transAxes, fontweight='bold')

    # Point de risque actuel
    risk_point_size = avg_accuracy * 0.8 + 0.1
    axes[1, 0].scatter(0.5, 0.5, s=1000, c=color, alpha=0.8, transform=axes[1, 0].transAxes)
    axes[1, 0].text(0.5, 0.3, f'Accuracy Moyenne\n{avg_accuracy:.3f}',
                    ha='center', va='center', transform=axes[1, 0].transAxes,
                    fontweight='bold', fontsize=12)

    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title('Évaluation du Risque de Confidentialité', fontsize=14, fontweight='bold')

    # 4. Tableau récapitulatif
    axes[1, 1].axis('off')

    # Créer le tableau de résultats
    table_data = []
    for attack in attack_names:
        metrics = results[attack]
        table_data.append([
            attack,
            f"{metrics.get('attack_accuracy', 0.0):.3f}",
            f"{metrics.get('precision', 0.0):.3f}",
            f"{metrics.get('recall', 0.0):.3f}",
            f"{metrics.get('auc_score', 0.0):.3f}"
        ])

    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Attaque', 'Accuracy', 'Precision', 'Recall', 'AUC'],
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style du tableau
    for i in range(len(attack_names) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor(color)
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F0F8FF' if i % 2 == 0 else 'white')
                # Colorier selon la performance
                if j == 1:  # Accuracy column
                    acc_val = float(table_data[i - 1][1])
                    if acc_val > 0.7:
                        cell.set_facecolor('#FFB6C1')  # Rouge clair
                    elif acc_val > 0.6:
                        cell.set_facecolor('#FFE4B5')  # Orange clair

    axes[1, 1].set_title('Résumé des Performances', fontsize=14, fontweight='bold')

    # Titre général avec évaluation du risque
    risk_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}[args.risk_level]
    fig.suptitle(
        f'{risk_emoji} Attaques MIA - {args.dataset.upper()} (Risque: {args.risk_level.upper()})',
        fontsize=16, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def calculate_privacy_metrics(results):
    """Calcule les métriques de confidentialité"""
    if not results:
        return {}

    accuracies = [r.get('attack_accuracy', 0.5) for r in results.values()]
    aucs = [r.get('auc_score', 0.5) for r in results.values()]

    avg_accuracy = np.mean(accuracies)
    avg_auc = np.mean(aucs)

    # Métrique de confidentialité (plus c'est proche de 0.5, mieux c'est)
    privacy_score = 1 - abs(avg_accuracy - 0.5) * 2

    return {
        'average_attack_accuracy': avg_accuracy,
        'average_auc': avg_auc,
        'privacy_score': privacy_score,
        'vulnerability_level': categorize_vulnerability(avg_accuracy),
        'recommendations': generate_recommendations(avg_accuracy)
    }


def categorize_vulnerability(attack_accuracy):
    """Catégorise le niveau de vulnérabilité"""
    if attack_accuracy > 0.75:
        return "CRITIQUE - Vulnérabilité très élevée"
    elif attack_accuracy > 0.65:
        return "ÉLEVÉ - Vulnérabilité significative"
    elif attack_accuracy > 0.55:
        return "MODÉRÉ - Vulnérabilité détectable"
    else:
        return "FAIBLE - Protection acceptable"


def generate_recommendations(attack_accuracy):
    """Génère des recommandations de sécurité"""
    recommendations = []

    if attack_accuracy > 0.7:
        recommendations.extend([
            "Implémenter la confidentialité différentielle (DP-SGD)",
            "Réduire drastiquement le nombre de tours de communication",
            "Augmenter la taille des lots locaux",
            "Ajouter du bruit aux gradients",
            "Considérer l'agrégation sécurisée (SecAgg)"
        ])
    elif attack_accuracy > 0.6:
        recommendations.extend([
            "Ajouter de la régularisation (dropout, weight decay)",
            "Limiter les informations partagées",
            "Implémenter un mécanisme de clipping des gradients",
            "Surveiller les métriques de confidentialité"
        ])
    else:
        recommendations.extend([
            "Maintenir les défenses actuelles",
            "Effectuer des audits réguliers de confidentialité",
            "Surveiller les nouvelles techniques d'attaque"
        ])

    return recommendations


def plot_roc_curves(attack_results, save_path=None):
    """Crée les courbes ROC pour les attaques (si données disponibles)"""
    # Note: Nécessiterait les probabilités d'attaque pour chaque échantillon
    # Implémentation simplifiée pour le projet académique

    plt.figure(figsize=(10, 8))

    for attack_name, metrics in attack_results.items():
        auc_score = metrics.get('auc_score', 0.5)

        # Simuler une courbe ROC basée sur l'AUC
        fpr = np.linspace(0, 1, 100)
        # Approximation simple de TPR basée sur AUC
        tpr = np.clip(auc_score * fpr + (auc_score - 0.5), 0, 1)

        plt.plot(fpr, tpr, label=f'{attack_name} (AUC = {auc_score:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Baseline (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbes ROC - Attaques MIA')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path.replace('.png', '_roc.png'), dpi=300, bbox_inches='tight')

    plt.close()


def create_privacy_budget_analysis(results, epsilon_values):
    """Analyse du budget de confidentialité différentielle"""
    # Simulation pour le projet académique
    # En pratique, nécessiterait une implémentation complète de DP

    plt.figure(figsize=(10, 6))

    # Simuler l'impact d'epsilon sur l'accuracy d'attaque
    for attack_name in results.keys():
        attack_accuracies = []
        for eps in epsilon_values:
            # Simulation: plus epsilon est petit, plus l'attaque est difficile
            base_acc = results[attack_name].get('attack_accuracy', 0.6)
            privacy_protection = 1 / (1 + eps)
            protected_acc = 0.5 + (base_acc - 0.5) * (1 - privacy_protection)
            attack_accuracies.append(protected_acc)

        plt.plot(epsilon_values, attack_accuracies, marker='o', label=f'{attack_name}')

    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Baseline (50%)')
    plt.xlabel('Epsilon (Budget de Confidentialité)')
    plt.ylabel('Attack Accuracy')
    plt.title('Impact de la Confidentialité Différentielle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, max(epsilon_values))
    plt.ylim(0.4, 1.0)

    return plt