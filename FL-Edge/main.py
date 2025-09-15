"""
Script principal pour simulation FL sur dispositifs Edge
Usage: python main.py --scenario iot --devices 20 --device_types mixed --epochs 30

Exemples rapides:
  python main.py --scenario iot --devices 15 --network_type wifi
  python main.py --scenario mobile --devices 30 --mobility high --network_type 4g
  python main.py --scenario smart_city --devices 50 --simulate_failures
"""

import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('Agg')

from edge_environment import EdgeEnvironment
from edge_devices import create_edge_devices
from communication import NetworkSimulator
from resource_manager import ResourceManager
from data_loader import load_edge_dataset
from utils import setup_gpu, save_edge_results, print_edge_info


def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Simulation d\'Apprentissage Fédéré sur Dispositifs Edge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scénarios disponibles:

  🏭 IoT (Internet of Things):
     python main.py --scenario iot --devices 20 --device_types sensor,gateway

  📱 Mobile (Smartphones/Tablets):
     python main.py --scenario mobile --devices 30 --mobility high --network_type 4g

  🚗 Vehicular (Véhicules connectés):
     python main.py --scenario vehicular --devices 25 --mobility high --area urban

  🏥 Healthcare (Dispositifs médicaux):
     python main.py --scenario healthcare --devices 15 --privacy_level high

  🏙️ Smart City (Ville intelligente):
     python main.py --scenario smart_city --devices 50 --device_types camera,sensor,gateway

  🏭 Industrial (Industrie 4.0):
     python main.py --scenario industrial --devices 20 --reliability_critical
        """
    )

    # Arguments obligatoires
    parser.add_argument('--scenario', type=str, required=True,
                        choices=['iot', 'mobile', 'vehicular', 'healthcare', 'smart_city', 'industrial'],
                        help='Scénario de simulation edge')

    # Arguments optionnels
    parser.add_argument('--devices', type=int, default=20,
                        help='Nombre de dispositifs edge (défaut: 20)')

    parser.add_argument('--device_types', type=str, default='mixed',
                        help='Types de dispositifs (mixed, smartphone, sensor, etc.)')

    parser.add_argument('--network_type', type=str, default='mixed',
                        choices=['wifi', '4g', '5g', 'ethernet', 'mixed'],
                        help='Type de réseau (défaut: mixed)')

    parser.add_argument('--mobility', type=str, default='static',
                        choices=['static', 'low', 'medium', 'high'],
                        help='Niveau de mobilité des dispositifs (défaut: static)')

    parser.add_argument('--epochs', type=int, default=25,
                        help='Nombre de rounds FL (défaut: 25)')

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar', 'synthetic'],
                        help='Dataset à utiliser (défaut: mnist)')

    parser.add_argument('--simulate_failures', action='store_true',
                        help='Simuler pannes et déconnexions')

    parser.add_argument('--battery_aware', action='store_true',
                        help='Gestion intelligente de la batterie')

    parser.add_argument('--privacy_level', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Niveau de protection de la vie privée')

    parser.add_argument('--compression', action='store_true',
                        help='Activer compression des modèles')

    parser.add_argument('--async_mode', action='store_true',
                        help='Mode agrégation asynchrone')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU à utiliser (-1 pour CPU, défaut: 0)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Graine aléatoire (défaut: 42)')

    parser.add_argument('--verbose', action='store_true',
                        help='Mode verbeux avec détails')

    parser.add_argument('--plot', action='store_true',
                        help='Générer graphiques de simulation')

    return parser.parse_args()


def run_edge_fl_simulation(devices, edge_env, network, resource_mgr, args):
    """Exécute la simulation FL sur edge"""

    print(f"🚀 Lancement simulation FL-Edge...")
    print(f"   Scénario: {args.scenario}")
    print(f"   Dispositifs: {len(devices)}")
    print(f"   Réseau: {args.network_type}")
    print(f"   Mobilité: {args.mobility}")

    # Initialiser métriques edge
    edge_metrics = {
        'round_times': [],
        'energy_consumption': [],
        'communication_costs': [],
        'device_participation': [],
        'network_latencies': [],
        'accuracy_per_round': [],
        'device_dropouts': [],
        'battery_levels': []
    }

    # Serveur edge (peut être dans le cloud ou edge gateway)
    from edge_server import EdgeFederatedServer
    server = EdgeFederatedServer(args.scenario)

    # Simulation rounds FL
    for round_num in range(args.epochs):
        print(f"\n📡 Round {round_num + 1}/{args.epochs}")

        # 1. Mise à jour environnement (mobilité, réseau, batterie)
        edge_env.update_environment(devices, round_num)

        # 2. Sélection des dispositifs disponibles
        available_devices = resource_mgr.select_available_devices(
            devices, args.battery_aware, args.simulate_failures
        )

        if len(available_devices) == 0:
            print("   ⚠️ Aucun dispositif disponible ce round")
            continue

        print(f"   📱 {len(available_devices)}/{len(devices)} dispositifs disponibles")

        # 3. Entraînement local sur dispositifs edge
        device_updates = []
        round_energy = 0
        round_latencies = []

        for device in available_devices:
            try:
                # Simuler contraintes de ressources
                if not resource_mgr.can_participate(device):
                    continue

                # Entraînement local avec contraintes edge
                update, energy, latency = device.train_with_constraints(
                    server.global_model, args
                )

                if update is not None:
                    device_updates.append((device.device_id, update))
                    round_energy += energy
                    round_latencies.append(latency)

                    # Mise à jour batterie
                    device.update_battery(energy)

            except Exception as e:
                if args.verbose:
                    print(f"     ⚠️ Erreur dispositif {device.device_id}: {e}")
                continue

        if not device_updates:
            print("   ⚠️ Aucune mise à jour reçue ce round")
            continue

        # 4. Communication vers serveur (avec simulation réseau)
        comm_cost, comm_latency = network.simulate_communication(
            device_updates, args.compression
        )

        # 5. Agrégation sur serveur
        if args.async_mode:
            accuracy = server.async_aggregate(device_updates)
        else:
            accuracy = server.sync_aggregate(device_updates)

        # 6. Enregistrer métriques edge
        edge_metrics['accuracy_per_round'].append(accuracy)
        edge_metrics['energy_consumption'].append(round_energy)
        edge_metrics['communication_costs'].append(comm_cost)
        edge_metrics['network_latencies'].append(np.mean(round_latencies) if round_latencies else 0)
        edge_metrics['device_participation'].append(len(device_updates))
        edge_metrics['device_dropouts'].append(len(devices) - len(available_devices))
        edge_metrics['battery_levels'].append(
            np.mean([d.battery_level for d in devices])
        )

        if args.verbose or round_num % 5 == 0:
            print(f"   📊 Accuracy: {accuracy:.4f}")
            print(f"   ⚡ Énergie: {round_energy:.2f}J")
            print(f"   📡 Latence: {np.mean(round_latencies) if round_latencies else 0:.1f}ms")
            print(f"   🔋 Batterie moyenne: {edge_metrics['battery_levels'][-1]:.1f}%")

    return edge_metrics, server


def main():
    """Fonction principale de simulation edge"""

    # Parse des arguments
    args = parse_arguments()

    # Configuration initiale
    print("🌐 Démarrage simulation FL-Edge...")
    print_edge_info(args)

    # Setup GPU
    setup_gpu(args.gpu)

    # Fixer les graines
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Créer la structure de dossiers
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    try:
        # 1. Créer environnement edge
        print(f"🌍 Création environnement edge: {args.scenario}")
        edge_env = EdgeEnvironment(args.scenario, args.mobility)

        # 2. Créer dispositifs edge
        print(f"📱 Création de {args.devices} dispositifs edge...")
        devices = create_edge_devices(
            args.devices, args.device_types, args.scenario
        )

        # 3. Charger et distribuer données
        print(f"📊 Chargement dataset edge: {args.dataset}")
        device_data = load_edge_dataset(
            args.dataset, devices, args.scenario
        )

        # 4. Initialiser simulateur réseau
        print(f"📡 Initialisation réseau: {args.network_type}")
        network = NetworkSimulator(args.network_type, args.mobility)

        # 5. Créer gestionnaire de ressources
        print("⚡ Initialisation gestionnaire ressources...")
        resource_mgr = ResourceManager(args.privacy_level)

        # 6. Lancer simulation FL-Edge
        edge_metrics, server = run_edge_fl_simulation(
            devices, edge_env, network, resource_mgr, args
        )

        # 7. Afficher résultats finaux
        print("\n" + "=" * 60)
        print("📊 RÉSULTATS SIMULATION FL-EDGE")
        print("=" * 60)

        if edge_metrics['accuracy_per_round']:
            final_accuracy = edge_metrics['accuracy_per_round'][-1]
            avg_energy = np.mean(edge_metrics['energy_consumption'])
            avg_latency = np.mean(edge_metrics['network_latencies'])
            avg_participation = np.mean(edge_metrics['device_participation'])
            total_dropouts = sum(edge_metrics['device_dropouts'])

            print(f"🎯 Accuracy finale: {final_accuracy:.4f}")
            print(f"⚡ Énergie moyenne/round: {avg_energy:.2f}J")
            print(f"📡 Latence moyenne: {avg_latency:.1f}ms")
            print(f"📱 Participation moyenne: {avg_participation:.1f} dispositifs")
            print(f"❌ Total dropouts: {total_dropouts}")

            # 8. Analyser efficacité edge
            edge_efficiency = analyze_edge_efficiency(edge_metrics, args)
            print(f"🏆 Score efficacité edge: {edge_efficiency:.2f}/10")

        else:
            print("❌ Aucune donnée de performance disponible")

        # 9. Sauvegarder résultats
        save_edge_results(edge_metrics, server, args)

        print(f"\n📁 Résultats sauvegardés:")
        print(f"    results/ - Métriques edge détaillées")
        print(f"    models/ - Modèles edge (.keras)")
        print(f"    visualizations/ - Graphiques edge")

    except Exception as e:
        print(f"❌ Erreur simulation edge: {e}")
        import traceback
        traceback.print_exc()


def analyze_edge_efficiency(metrics, args):
    """Analyse l'efficacité de la solution edge"""

    efficiency_score = 0

    # Accuracy (40% du score)
    if metrics['accuracy_per_round']:
        final_acc = metrics['accuracy_per_round'][-1]
        acc_score = min(final_acc * 4, 4.0)  # Max 4 points
        efficiency_score += acc_score

    # Efficacité énergétique (30% du score)
    if metrics['energy_consumption']:
        avg_energy = np.mean(metrics['energy_consumption'])
        # Moins d'énergie = meilleur score
        energy_score = max(0, 3.0 - (avg_energy / 100))
        efficiency_score += energy_score

    # Participation (20% du score)
    if metrics['device_participation']:
        avg_participation = np.mean(metrics['device_participation'])
        participation_rate = avg_participation / args.devices
        participation_score = participation_rate * 2.0  # Max 2 points
        efficiency_score += participation_score

    # Robustesse (10% du score)
    if metrics['device_dropouts']:
        total_dropouts = sum(metrics['device_dropouts'])
        max_possible_dropouts = len(metrics['device_dropouts']) * args.devices
        robustness = 1 - (total_dropouts / max_possible_dropouts) if max_possible_dropouts > 0 else 1
        robustness_score = robustness * 1.0  # Max 1 point
        efficiency_score += robustness_score

    return efficiency_score


if __name__ == '__main__':
    main()