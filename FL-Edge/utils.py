"""
Utilitaires pour simulation FL-Edge
"""
import os
import json
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def setup_gpu(gpu_id):
    """Configure l'utilisation du GPU pour edge"""
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpu_id == -1:
        print("🖥️  Mode CPU uniquement (recommandé pour simulation edge)")
        tf.config.set_visible_devices([], 'GPU')
    elif gpus and gpu_id < len(gpus):
        try:
            # Configuration légère pour simulation edge
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            tf.config.experimental.set_memory_limit(gpus[gpu_id], 1024)  # 1GB limite
            print(f"🎮 GPU configuré pour edge: {gpus[gpu_id].name}")
        except RuntimeError as e:
            print(f"⚠️ Erreur GPU, passage CPU: {e}")
            tf.config.set_visible_devices([], 'GPU')
    else:
        print(f"⚠️ GPU {gpu_id} non trouvé, utilisation CPU")


def print_edge_info(args):
    """Affiche les informations de configuration edge"""
    scenario_descriptions = {
        'iot': 'Internet of Things - Capteurs et objets connectés',
        'mobile': 'Appareils mobiles - Smartphones et tablettes',
        'vehicular': 'Véhicules connectés - Automobiles intelligentes',
        'healthcare': 'Dispositifs médicaux - Équipements de santé',
        'smart_city': 'Ville intelligente - Infrastructure urbaine',
        'industrial': 'Industrie 4.0 - Automatisation industrielle'
    }

    print("=" * 70)
    print("🌐 CONFIGURATION SIMULATION FL-EDGE")
    print("=" * 70)
    print(f"🎯 Scénario: {args.scenario.upper()}")
    print(f"   Description: {scenario_descriptions.get(args.scenario, 'Scénario personnalisé')}")
    print(f"📱 Dispositifs: {args.devices}")
    print(f"🔧 Types: {args.device_types}")
    print(f"📡 Réseau: {args.network_type}")
    print(f"🚶 Mobilité: {args.mobility}")
    print(f"📊 Dataset: {args.dataset.upper()}")
    print(f"🔄 Rounds FL: {args.epochs}")

    # Options avancées
    advanced_options = []
    if args.simulate_failures:
        advanced_options.append("Simulation pannes")
    if args.battery_aware:
        advanced_options.append("Gestion batterie")
    if args.compression:
        advanced_options.append("Compression modèles")
    if args.async_mode:
        advanced_options.append("Mode asynchrone")

    if advanced_options:
        print(f"⚙️  Options: {', '.join(advanced_options)}")

    print(f"🔒 Confidentialité: {args.privacy_level.upper()}")
    print("=" * 70)


def save_edge_results(metrics, server, args):
    """Sauvegarde les résultats de simulation edge"""
    timestamp = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    # Nom du fichier basé sur scénario
    base_name = f"{args.scenario}_{args.devices}dev_{args.epochs}rounds"

    # 1. Sauvegarder métriques détaillées
    results_path = os.path.join('results', f"{timestamp}_{base_name}.txt")
    save_edge_metrics_report(metrics, args, results_path)

    # 2. Sauvegarder JSON structuré
    json_path = os.path.join('results', f"{timestamp}_{base_name}.json")
    save_edge_json(metrics, server, args, json_path)

    # 3. Sauvegarder modèle global
    model_path = os.path.join('models', f"{base_name}_global_model.keras")
    if hasattr(server, 'global_model'):
        server.global_model.save(model_path)
        print(f"💾 Modèle global sauvegardé: {model_path}")

    # 4. Créer visualisations
    plot_path = os.path.join('visualizations', f"{base_name}_{timestamp}.png")
    plot_edge_metrics(metrics, args, plot_path)

    print(f"📊 Résultats edge sauvegardés: {results_path}")


def save_edge_metrics_report(metrics, args, file_path):
    """Sauvegarde un rapport détaillé des métriques edge"""
    try:
        with open(file_path, 'w') as f:
            f.write("RAPPORT SIMULATION FL-EDGE\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Scénario: {args.scenario.upper()}\n")
            f.write(f"Dispositifs: {args.devices}\n")
            f.write(f"Types: {args.device_types}\n")
            f.write(f"Réseau: {args.network_type}\n")
            f.write(f"Mobilité: {args.mobility}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Rounds: {args.epochs}\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write("\n" + "=" * 50 + "\n\n")

            f.write("MÉTRIQUES DE PERFORMANCE:\n")
            f.write("-" * 30 + "\n")

            if metrics.get('accuracy_per_round'):
                final_acc = metrics['accuracy_per_round'][-1]
                max_acc = max(metrics['accuracy_per_round'])
                f.write(f"Accuracy finale: {final_acc:.6f}\n")
                f.write(f"Accuracy maximale: {max_acc:.6f}\n")

            if metrics.get('energy_consumption'):
                total_energy = sum(metrics['energy_consumption'])
                avg_energy = np.mean(metrics['energy_consumption'])
                f.write(f"Énergie totale consommée: {total_energy:.2f}J\n")
                f.write(f"Énergie moyenne/round: {avg_energy:.2f}J\n")

            if metrics.get('communication_costs'):
                total_comm = sum(metrics['communication_costs'])
                avg_comm = np.mean(metrics['communication_costs'])
                f.write(f"Coût communication total: {total_comm:.0f}\n")
                f.write(f"Coût communication moyen: {avg_comm:.0f}\n")

            if metrics.get('network_latencies'):
                avg_latency = np.mean(metrics['network_latencies'])
                max_latency = max(metrics['network_latencies'])
                f.write(f"Latence réseau moyenne: {avg_latency:.1f}ms\n")
                f.write(f"Latence réseau maximale: {max_latency:.1f}ms\n")

            f.write("\nMÉTRIQUES EDGE SPÉCIFIQUES:\n")
            f.write("-" * 30 + "\n")

            if metrics.get('device_participation'):
                avg_participation = np.mean(metrics['device_participation'])
                participation_rate = avg_participation / args.devices
                f.write(f"Participation moyenne: {avg_participation:.1f} dispositifs\n")
                f.write(f"Taux de participation: {participation_rate:.2%}\n")

            if metrics.get('device_dropouts'):
                total_dropouts = sum(metrics['device_dropouts'])
                f.write(f"Total dropouts: {total_dropouts}\n")

            if metrics.get('battery_levels'):
                final_battery = metrics['battery_levels'][-1] if metrics['battery_levels'] else 0
                min_battery = min(metrics['battery_levels']) if metrics['battery_levels'] else 0
                f.write(f"Niveau batterie final: {final_battery:.1f}%\n")
                f.write(f"Niveau batterie minimal: {min_battery:.1f}%\n")

            # Analyse de l'efficacité edge
            f.write("\nANALYSE EFFICACITÉ EDGE:\n")
            f.write("-" * 30 + "\n")

            edge_efficiency = calculate_edge_efficiency_score(metrics, args)
            f.write(f"Score efficacité edge: {edge_efficiency:.2f}/10\n")

            efficiency_breakdown = analyze_efficiency_breakdown(metrics, args)
            for factor, score in efficiency_breakdown.items():
                f.write(f"  - {factor}: {score:.2f}/10\n")

    except IOError as e:
        print(f"❌ Erreur sauvegarde rapport: {e}")


def save_edge_json(metrics, server, args, file_path):
    """Sauvegarde JSON structuré pour analyse ultérieure"""
    try:
        edge_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'scenario': args.scenario,
                'devices': args.devices,
                'device_types': args.device_types,
                'network_type': args.network_type,
                'mobility': args.mobility,
                'dataset': args.dataset,
                'epochs': args.epochs,
                'privacy_level': args.privacy_level,
                'advanced_options': {
                    'simulate_failures': args.simulate_failures,
                    'battery_aware': args.battery_aware,
                    'compression': args.compression,
                    'async_mode': args.async_mode
                }
            },
            'metrics': metrics,
            'analysis': {
                'edge_efficiency_score': calculate_edge_efficiency_score(metrics, args),
                'efficiency_breakdown': analyze_efficiency_breakdown(metrics, args),
                'scenario_specific_insights': get_scenario_insights(metrics, args)
            }
        }

        with open(file_path, 'w') as f:
            json.dump(edge_data, f, indent=2, default=str)

    except Exception as e:
        print(f"❌ Erreur sauvegarde JSON: {e}")


def plot_edge_metrics(metrics, args, plot_path):
    """Crée les visualisations spécifiques edge"""

    if not any(metrics.values()):
        print("⚠️ Aucune métrique à visualiser")
        return

    # Créer figure avec sous-graphiques spécialisés edge
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    rounds = range(1, len(metrics.get('accuracy_per_round', [])) + 1)

    # 1. Évolution accuracy
    if metrics.get('accuracy_per_round'):
        axes[0, 0].plot(rounds, metrics['accuracy_per_round'], 'b-', linewidth=2, marker='o')
        axes[0, 0].set_title('Accuracy du Modèle Global', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Round FL')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)

    # 2. Consommation énergétique
    if metrics.get('energy_consumption'):
        axes[0, 1].plot(rounds, metrics['energy_consumption'], 'r-', linewidth=2, marker='s')
        axes[0, 1].set_title('Consommation Énergétique', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Round FL')
        axes[0, 1].set_ylabel('Énergie (J)')
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Participation des dispositifs
    if metrics.get('device_participation') and metrics.get('device_dropouts'):
        participating = metrics['device_participation']
        dropouts = metrics['device_dropouts']

        axes[0, 2].bar(rounds, participating, alpha=0.7, label='Participants', color='green')
        axes[0, 2].bar(rounds, dropouts, bottom=participating, alpha=0.7, label='Dropouts', color='red')
        axes[0, 2].set_title('Participation des Dispositifs', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Round FL')
        axes[0, 2].set_ylabel('Nombre de dispositifs')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # 4. Latence réseau
    if metrics.get('network_latencies'):
        axes[1, 0].plot(rounds, metrics['network_latencies'], 'orange', linewidth=2, marker='^')
        axes[1, 0].set_title('Latence Réseau', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Round FL')
        axes[1, 0].set_ylabel('Latence (ms)')
        axes[1, 0].grid(True, alpha=0.3)

    # 5. Niveaux de batterie
    if metrics.get('battery_levels'):
        axes[1, 1].plot(rounds, metrics['battery_levels'], 'purple', linewidth=2, marker='d')
        axes[1, 1].axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Seuil critique')
        axes[1, 1].set_title('Niveau de Batterie Moyen', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Round FL')
        axes[1, 1].set_ylabel('Batterie (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 100)

    # 6. Résumé et insights scénario
    axes[1, 2].axis('off')

    # Informations du scénario
    scenario_info = get_scenario_display_info(args.scenario)
    axes[1, 2].text(0.1, 0.9, f'🎯 Scénario: {args.scenario.upper()}',
                    fontsize=14, fontweight='bold', transform=axes[1, 2].transAxes)
    axes[1, 2].text(0.1, 0.8, scenario_info, fontsize=10, transform=axes[1, 2].transAxes