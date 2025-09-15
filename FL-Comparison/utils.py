"""
Utilitaires pour comparaisons FL vs autres méthodes
"""
import os
import json
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


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


def print_comparison_info(args):
    """Affiche les informations de configuration de comparaison"""
    print("=" * 60)
    print("🔧 CONFIGURATION DE COMPARAISON")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset.upper()}")
    print(f"🔄 Type de comparaison: {args.comparison}")
    print(f"👥 Nombre de clients FL: {args.clients}")
    print(f"📈 Distribution: {'IID' if args.iid else 'Non-IID'}")
    print(f"🔄 Époques: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📈 Learning rate: {args.lr}")
    print("=" * 60)


def save_comparison_results(results, args):
    """Sauvegarde les résultats de comparaison"""
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    # Nom du fichier basé sur la comparaison
    comparison_types = {
        'fl_vs_central': 'FLvsCentral',
        'fl_vs_distributed': 'FLvsDistributed',
        'all': 'AllMethods'
    }

    base_name = f"{args.dataset}_{comparison_types.get(args.comparison, 'comparison')}_{args.epochs}ep"

    # 1. Sauvegarder résultats détaillés
    log_path = os.path.join('results', f"{timestamp}_{base_name}.txt")
    save_detailed_results(results, args, log_path)

    # 2. Sauvegarder JSON structuré
    json_path = os.path.join('results', f"{timestamp}_{base_name}.json")
    save_json_results(results, args, json_path)

    # 3. Créer visualisations
    plot_path = os.path.join('visualizations', f"{base_name}_{timestamp}.png")
    plot_comparison_results(results, args, plot_path)

    print(f"💾 Résultats sauvegardés: {log_path}")
    print(f"📊 Graphiques sauvegardés: {plot_path}")


def save_detailed_results(results, args, log_path):
    """Sauvegarde les résultats détaillés au format texte"""
    try:
        with open(log_path, 'w') as f:
            f.write(f"Comparison Results: {args.dataset.upper()}, {args.comparison}\n")
            f.write(f"Configuration: {args.clients} clients, {'IID' if args.iid else 'Non-IID'}\n")
            f.write(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")

            for method, metrics in results.items():
                f.write(f"Method: {method}\n")
                f.write(f"  Final Accuracy: {metrics.get('final_accuracy', 0.0):.6f}\n")
                f.write(f"  Training Time: {metrics.get('training_time', 0.0):.2f}s\n")
                f.write(f"  Communication Cost: {metrics.get('communication_cost', 0)}\n")
                f.write(f"  Convergence Round: {metrics.get('convergence_round', 0)}\n")

                if 'accuracy_std' in metrics:
                    f.write(f"  Accuracy Std: {metrics['accuracy_std']:.6f}\n")

                f.write("\n")

    except IOError as e:
        print(f"❌ Erreur sauvegarde résultats: {e}")


def save_json_results(results, args, json_path):
    """Sauvegarde au format JSON structuré"""
    try:
        json_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'dataset': args.dataset,
                'comparison_type': args.comparison,
                'clients': args.clients,
                'iid': args.iid,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr
            },
            'results': results
        }

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)

    except Exception as e:
        print(f"❌ Erreur sauvegarde JSON: {e}")


def plot_comparison_results(results, args, plot_path):
    """Crée les visualisations de comparaison"""

    if not results:
        print("⚠️ Aucun résultat à visualiser")
        return

    # Préparer les données
    methods = list(results.keys())
    accuracies = [results[method].get('final_accuracy', 0.0) for method in methods]
    times = [results[method].get('training_time', 0.0) for method in methods]
    comm_costs = [results[method].get('communication_cost', 0) for method in methods]

    # Créer les sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Accuracy finale
    bars1 = axes[0, 0].bar(methods, accuracies, color=['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC'][:len(methods)])
    axes[0, 0].set_title(f'Accuracy Finale - {args.dataset.upper()}', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    # Ajouter les valeurs sur les barres
    for bar, acc in zip(bars1, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. Temps d'entraînement
    bars2 = axes[0, 1].bar(methods, times, color=['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC'][:len(methods)])
    axes[0, 1].set_title('Temps d\'Entraînement', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Temps (secondes)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # Ajouter les valeurs
    for bar, time_val in zip(bars2, times):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                        f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')

    # 3. Coût de communication
    bars3 = axes[1, 0].bar(methods, comm_costs, color=['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9932CC'][:len(methods)])
    axes[1, 0].set_title('Coût de Communication', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Paramètres échangés')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Format des coûts en notation scientifique si nécessaire
    for bar, cost in zip(bars3, comm_costs):
        if cost > 1000000:
            label = f'{cost/1000000:.1f}M'
        elif cost > 1000:
            label = f'{cost/1000:.1f}K'
        else:
            label = f'{cost}'
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(comm_costs)*0.01,
                        label, ha='center', va='bottom', fontweight='bold')

    # 4. Tableau récapitulatif
    axes[1, 1].axis('off')

    # Créer le tableau
    table_data = []
    for method in methods:
        acc = results[method].get('final_accuracy', 0.0)
        time_val = results[method].get('training_time', 0.0)
        comm = results[method].get('communication_cost', 0)
        table_data.append([method, f'{acc:.4f}', f'{time_val:.1f}s', f'{comm:,}'])

    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Méthode', 'Accuracy', 'Temps', 'Communication'],
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style du tableau
    for i in range(len(methods) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4169E1')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F0F8FF' if i % 2 == 0 else 'white')

    axes[1, 1].set_title('Résumé des Performances', fontsize=14, fontweight='bold')

    # Titre général
    comparison_titles = {
        'fl_vs_central': 'Fédéré vs Centralisé',
        'fl_vs_distributed': 'Fédéré vs Distribué',
        'all': 'Comparaison Complète'
    }

    fig.suptitle(
        f'{comparison_titles.get(args.comparison, args.comparison)} - {args.dataset.upper()} - {args.iid} : {args.epochs} epochs',
        fontsize=16, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_performance_summary(results):
    """Crée un résumé des performances"""
    if not results:
        return "Aucun résultat disponible"

    summary = []

    # Trouver la meilleure méthode par métrique
    best_accuracy = max(results.items(), key=lambda x: x[1].get('final_accuracy', 0))
    fastest_method = min(results.items(), key=lambda x: x[1].get('training_time', float('inf')))

    summary.append(f"🏆 Meilleure accuracy: {best_accuracy[0]} ({best_accuracy[1].get('final_accuracy', 0):.4f})")
    summary.append(f"⚡ Plus rapide: {fastest_method[0]} ({fastest_method[1].get('training_time', 0):.1f}s)")

    # Communication (seulement pour méthodes avec communication)
    comm_methods = {k: v for k, v in results.items() if v.get('communication_cost', 0) > 0}
    if comm_methods:
        lowest_comm = min(comm_methods.items(), key=lambda x: x[1].get('communication_cost', float('inf')))
        summary.append(f"📡 Moins de communication: {lowest_comm[0]} ({lowest_comm[1].get('communication_cost', 0):,})")

    return "\n".join(summary)