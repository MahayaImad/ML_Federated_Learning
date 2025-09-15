"""
Simulation de l'environnement edge (mobilité, réseau, conditions)
"""
import numpy as np
import random
import math
from typing import List, Dict, Any


class EdgeEnvironment:
    """Simule l'environnement edge avec mobilité, réseau, etc."""

    def __init__(self, scenario, mobility_level='static'):
        self.scenario = scenario
        self.mobility_level = mobility_level
        self.current_time = 0  # Temps de simulation en minutes

        # Paramètres environnementaux
        self.weather_condition = 'clear'
        self.network_congestion = 0.0  # 0-1
        self.electromagnetic_interference = 0.0  # 0-1

        # Événements spéciaux selon scénario
        self.special_events = self._initialize_special_events()

        # Zones géographiques avec propriétés
        self.zones = self._initialize_zones()

        print(f"🌍 Environnement edge initialisé: {scenario} (mobilité: {mobility_level})")

    def _initialize_special_events(self):
        """Initialise les événements spéciaux selon le scénario"""
        events = {
            'iot': [
                {'type': 'power_outage', 'probability': 0.001, 'duration': 30},
                {'type': 'network_maintenance', 'probability': 0.002, 'duration': 60},
            ],
            'mobile': [
                {'type': 'rush_hour', 'probability': 0.1, 'duration': 120},
                {'type': 'event_crowd', 'probability': 0.005, 'duration': 180},
                {'type': 'network_congestion', 'probability': 0.05, 'duration': 45},
            ],
            'vehicular': [
                {'type': 'traffic_jam', 'probability': 0.08, 'duration': 60},
                {'type': 'tunnel_passage', 'probability': 0.02, 'duration': 5},
                {'type': 'weather_impact', 'probability': 0.03, 'duration': 90},
            ],
            'healthcare': [
                {'type': 'emergency', 'probability': 0.01, 'duration': 30},
                {'type': 'equipment_sterilization', 'probability': 0.005, 'duration': 45},
            ],
            'smart_city': [
                {'type': 'public_event', 'probability': 0.02, 'duration': 240},
                {'type': 'infrastructure_maintenance', 'probability': 0.008, 'duration': 120},
            ],
            'industrial': [
                {'type': 'shift_change', 'probability': 0.125, 'duration': 15},  # 3x par jour
                {'type': 'maintenance_window', 'probability': 0.01, 'duration': 180},
            ]
        }
        return events.get(self.scenario, [])

    def _initialize_zones(self):
        """Initialise les zones géographiques avec leurs propriétés"""
        if self.scenario == 'smart_city':
            return {
                'downtown': {
                    'network_quality_base': 0.9,
                    'congestion_factor': 0.7,
                    'device_density': 'high'
                },
                'residential': {
                    'network_quality_base': 0.8,
                    'congestion_factor': 0.3,
                    'device_density': 'medium'
                },
                'industrial': {
                    'network_quality_base': 0.85,
                    'congestion_factor': 0.2,
                    'device_density': 'medium'
                },
                'park': {
                    'network_quality_base': 0.6,
                    'congestion_factor': 0.1,
                    'device_density': 'low'
                }
            }
        elif self.scenario == 'vehicular':
            return {
                'highway': {
                    'network_quality_base': 0.7,
                    'congestion_factor': 0.5,
                    'speed_limit': 130
                },
                'urban_road': {
                    'network_quality_base': 0.8,
                    'congestion_factor': 0.6,
                    'speed_limit': 50
                },
                'tunnel': {
                    'network_quality_base': 0.3,
                    'congestion_factor': 0.8,
                    'speed_limit': 90
                }
            }
        else:
            return {
                'default': {
                    'network_quality_base': 0.7,
                    'congestion_factor': 0.4,
                    'device_density': 'medium'
                }
            }

    def update_environment(self, devices, round_number):
        """Met à jour l'environnement pour un nouveau round"""
        self.current_time += 1  # Incrément de 1 minute par round

        # Mise à jour conditions globales
        self._update_weather()
        self._update_network_conditions()

        # Traitement des événements spéciaux
        self._process_special_events(devices)

        # Mise à jour mobilité des dispositifs
        self._update_device_mobility(devices)

        # Mise à jour conditions locales par zone
        self._update_zone_conditions()

        # Patterns temporels selon scénario
        self._apply_temporal_patterns(devices, round_number)

    def _update_weather(self):
        """Mise à jour des conditions météorologiques"""
        weather_options = ['clear', 'cloudy', 'rainy', 'stormy']

        # Changement météo avec probabilité
        if random.random() < 0.02:  # 2% chance de changement par minute
            self.weather_condition = random.choice(weather_options)

        # Impact sur le réseau selon la météo
        weather_network_impact = {
            'clear': 1.0,
            'cloudy': 0.95,
            'rainy': 0.85,
            'stormy': 0.6
        }

        self.weather_network_factor = weather_network_impact[self.weather_condition]

    def _update_network_conditions(self):
        """Mise à jour des conditions réseau globales"""
        # Congestion réseau avec variation sinusoïdale (simule pics de trafic)
        time_of_day = (self.current_time % 1440) / 1440  # 0-1 pour 24h

        # Pics de congestion aux heures de pointe
        peak_morning = 0.3 + 0.2 * math.sin(2 * math.pi * (time_of_day - 0.35))
        peak_evening = 0.4 + 0.3 * math.sin(2 * math.pi * (time_of_day - 0.75))

        base_congestion = max(peak_morning, peak_evening, 0.1)

        # Ajout de bruit aléatoire
        self.network_congestion = np.clip(
            base_congestion + random.uniform(-0.1, 0.1), 0.0, 1.0
        )

        # Interférences électromagnétiques (plus élevées en zone industrielle)
        if self.scenario == 'industrial':
            self.electromagnetic_interference = random.uniform(0.1, 0.3)
        else:
            self.electromagnetic_interference = random.uniform(0.0, 0.1)

    def _process_special_events(self, devices):
        """Traite les événements spéciaux selon le scénario"""
        for event in self.special_events:
            if random.random() < event['probability']:
                self._trigger_special_event(event, devices)

    def _trigger_special_event(self, event, devices):
        """Déclenche un événement spécial"""
        event_type = event['type']

        if event_type == 'power_outage':
            # Panne de courant: dispositifs non alimentés perdent batterie rapidement
            affected_devices = [d for d in devices if not d.is_charging]
            for device in affected_devices[:len(affected_devices) // 3]:  # 1/3 affectés
                device.battery_level *= 0.8
                device.is_charging = False

        elif event_type == 'network_maintenance':
            # Maintenance réseau: qualité réseau réduite
            for device in devices:
                device.network_quality *= 0.5

        elif event_type == 'rush_hour':
            # Heure de pointe: plus de mobilité, plus de congestion
            mobile_devices = [d for d in devices if d.is_mobile]
            for device in mobile_devices:
                device.velocity = min(device.velocity * 1.5, 120)
                device.network_quality *= 0.7

        elif event_type == 'traffic_jam':
            # Embouteillage: véhicules ralentissent mais réseau stable
            vehicle_devices = [d for d in devices if d.device_type == 'vehicle_ecu']
            for device in vehicle_devices:
                device.velocity *= 0.3
                device.current_load += 0.1  # Plus de temps pour calculs

        elif event_type == 'emergency':
            # Urgence médicale: priorité aux dispositifs critiques
            medical_devices = [d for d in devices if d.device_type == 'medical_device']
            for device in medical_devices:
                device.current_load *= 0.5  # Libère ressources
                device.network_quality = min(1.0, device.network_quality * 1.2)

        elif event_type == 'shift_change':
            # Changement d'équipe industrielle: pic d'activité
            industrial_devices = [d for d in devices if d.device_type == 'industrial_sensor']
            for device in industrial_devices:
                device.current_load = min(0.9, device.current_load + 0.3)

        print(f"🚨 Événement: {event_type} (durée: {event['duration']}min)")

    def _update_device_mobility(self, devices):
        """Met à jour la mobilité des dispositifs"""
        mobility_factors = {
            'static': 0.0,
            'low': 0.1,
            'medium': 0.5,
            'high': 1.0
        }

        mobility_factor = mobility_factors[self.mobility_level]

        for device in devices:
            if device.is_mobile and mobility_factor > 0:
                # Variation de vitesse
                velocity_change = random.uniform(-10, 10) * mobility_factor
                device.velocity = max(0, device.velocity + velocity_change)

                # Limite selon type de dispositif
                max_speeds = {
                    'smartphone_low': 50,
                    'smartphone_high': 50,
                    'tablet': 50,
                    'vehicle_ecu': 130
                }

                max_speed = max_speeds.get(device.device_type, 30)
                device.velocity = min(device.velocity, max_speed)

                # Mise à jour position
                device.update_mobility(time_delta=60)  # 1 minute

                # Impact sur qualité réseau selon vitesse
                speed_impact = 1.0 - (device.velocity / 150) * 0.3
                device.network_quality *= speed_impact
                device.network_quality = np.clip(device.network_quality, 0.1, 1.0)

    def _update_zone_conditions(self):
        """Met à jour les conditions par zone géographique"""
        for zone_name, zone_props in self.zones.items():
            # Variation temporelle des conditions de zone
            time_factor = math.sin(2 * math.pi * self.current_time / 1440)  # Cycle 24h

            # Mise à jour qualité réseau de base
            congestion_impact = 1 - (zone_props.get('congestion_factor', 0.5) * self.network_congestion)
            zone_props['current_network_quality'] = (
                    zone_props.get('network_quality_base', 0.7) *
                    congestion_impact *
                    self.weather_network_factor *
                    (1 + 0.1 * time_factor)  # Variation temporelle
            )

    def _apply_temporal_patterns(self, devices, round_number):
        """Applique des patterns temporels selon le scénario"""

        if self.scenario == 'mobile':
            # Patterns d'usage mobile (plus actif le jour)
            hour_of_day = (self.current_time % 1440) / 60  # 0-24h

            if 8 <= hour_of_day <= 22:  # Jour
                activity_multiplier = 1.2
            else:  # Nuit
                activity_multiplier = 0.6

            mobile_devices = [d for d in devices if d.device_type.startswith('smartphone')]
            for device in mobile_devices:
                device.current_load *= activity_multiplier
                device.current_load = np.clip(device.current_load, 0.05, 0.95)

        elif self.scenario == 'industrial':
            # Patterns industriels (3x8h, moins actif la nuit)
            hour_of_day = (self.current_time % 1440) / 60

            if 6 <= hour_of_day <= 22:  # Jour (2 équipes)
                industrial_activity = 0.8
            else:  # Nuit (équipe réduite)
                industrial_activity = 0.4

            industrial_devices = [d for d in devices if d.device_type == 'industrial_sensor']
            for device in industrial_devices:
                base_load = 0.3 * industrial_activity
                device.current_load = base_load + random.uniform(0, 0.3)

        elif self.scenario == 'healthcare':
            # Santé: activité constante avec pics d'urgence
            if round_number % 50 == 0:  # Pic toutes les 50 minutes
                medical_devices = [d for d in devices if d.device_type == 'medical_device']
                for device in medical_devices:
                    device.current_load = min(0.9, device.current_load + 0.4)

    def get_zone_impact(self, device):
        """Calcule l'impact de la zone sur un dispositif"""
        device_zone = device.location.get('zone', 'default')
        zone_props = self.zones.get(device_zone, self.zones.get('default', {}))

        return {
            'network_quality_modifier': zone_props.get('current_network_quality', 0.7),
            'congestion_level': zone_props.get('congestion_factor', 0.5),
            'interference_level': self.electromagnetic_interference
        }

    def get_environment_summary(self):
        """Retourne un résumé de l'état environnemental"""
        return {
            'current_time_minutes': self.current_time,
            'weather_condition': self.weather_condition,
            'network_congestion': self.network_congestion,
            'electromagnetic_interference': self.electromagnetic_interference,
            'weather_network_factor': getattr(self, 'weather_network_factor', 1.0),
            'active_zones': len(self.zones)
        }

    def simulate_network_failure(self, duration_minutes=10):
        """Simule une panne réseau généralisée"""
        print(f"🔴 Panne réseau simulée ({duration_minutes} minutes)")

        # Programmer la fin de la panne
        failure_end_time = self.current_time + duration_minutes

        return {
            'type': 'network_failure',
            'start_time': self.current_time,
            'end_time': failure_end_time,
            'severity': random.uniform(0.7, 1.0)  # 70-100% de dégradation
        }

    def calculate_environmental_stress(self, devices):
        """Calcule le niveau de stress environnemental global"""
        stress_factors = []

        # Stress météorologique
        weather_stress = {'clear': 0, 'cloudy': 0.2, 'rainy': 0.5, 'stormy': 0.8}
        stress_factors.append(weather_stress[self.weather_condition])

        # Stress de congestion réseau
        stress_factors.append(self.network_congestion)

        # Stress d'interférence électromagnétique
        stress_factors.append(self.electromagnetic_interference)

        # Stress de densité des dispositifs
        active_devices = [d for d in devices if d.current_load > 0]
        device_density_stress = len(active_devices) / max(len(devices), 1)
        stress_factors.append(device_density_stress * 0.5)

        # Stress de batterie globale
        battery_levels = [d.battery_level for d in devices if d.specs['battery_capacity'] != float('inf')]
        if battery_levels:
            avg_battery = np.mean(battery_levels) / 100.0
            battery_stress = 1.0 - avg_battery  # Plus la batterie est faible, plus le stress est élevé
            stress_factors.append(battery_stress * 0.3)

        # Stress temporel (heures de pointe)
        hour = (self.current_time / 60) % 24
        if 8 <= hour <= 10 or 17 <= hour <= 19:  # Heures de pointe
            stress_factors.append(0.4)
        else:
            stress_factors.append(0.1)

        # Calcul du stress global (moyenne pondérée)
        global_stress = np.mean(stress_factors)

        return min(1.0, global_stress)  # Limiter à 1.0 maximum


    def _update_weather(self):
        """Mise à jour des conditions météorologiques"""
        weather_transitions = {
            'clear': ['clear', 'cloudy'],
            'cloudy': ['clear', 'cloudy', 'rainy'],
            'rainy': ['cloudy', 'rainy', 'stormy'],
            'stormy': ['rainy', 'cloudy']
        }

        if random.random() < 0.05:  # 5% chance de changement
            self.weather_condition = random.choice(
                weather_transitions[self.weather_condition]
            )

            # Impact sur le réseau
            weather_network_impact = {
                'clear': 1.0, 'cloudy': 0.95, 'rainy': 0.8, 'stormy': 0.5
            }
            self.weather_network_factor = weather_network_impact[self.weather_condition]

    def _update_network_conditions(self):
        """Met à jour les conditions réseau globales"""
        # Congestion variable selon l'heure
        hour = (self.current_time / 60) % 24
        if 8 <= hour <= 10 or 17 <= hour <= 19:  # Heures de pointe
            base_congestion = 0.7
        else:
            base_congestion = 0.3

        self.network_congestion = base_congestion + random.uniform(-0.2, 0.2)
        self.network_congestion = np.clip(self.network_congestion, 0.0, 1.0)

    def _process_special_events(self, devices):
        """Traite les événements spéciaux"""
        for event in self.special_events:
            if random.random() < event['probability']:
                print(f"🚨 Événement: {event['type']}")
                self._apply_event_impact(event, devices)

    def _apply_event_impact(self, event, devices):
        """Applique l'impact d'un événement"""
        if event['type'] == 'power_outage':
            affected = random.sample(devices, k=min(3, len(devices)))
            for device in affected:
                device.battery_level *= 0.5
        elif event['type'] == 'network_congestion':
            for device in devices:
                device.network_quality *= 0.7

    def _update_device_mobility(self, devices):
        """Met à jour la position des dispositifs mobiles"""
        if self.mobility_level == 'static':
            return

        mobility_factor = {'low': 0.1, 'medium': 0.3, 'high': 0.5}[self.mobility_level]

        for device in devices:
            if device.specs.get('mobile', False):
                # Déplacement aléatoire
                if 'position' not in device.location:
                    device.location['position'] = [0.0, 0.0]

                device.location['position'][0] += random.uniform(-mobility_factor, mobility_factor)
                device.location['position'][1] += random.uniform(-mobility_factor, mobility_factor)

    def _update_zone_conditions(self):
        """Met à jour les conditions par zone"""
        for zone_name, zone_props in self.zones.items():
            # Variation de qualité réseau
            base_quality = zone_props['network_quality_base']
            zone_props['current_network_quality'] = base_quality * self.weather_network_factor