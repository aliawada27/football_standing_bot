#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.ticker as ticker
from pathlib import Path

class PremierLeagueVisualizer:
    """
    Classe pour visualiser les données de Premier League récupérées par PremierLeagueScraper et analysées par PremierLeagueAnalyzer
    """
    def __init__(self, data_dir='data', output_dir=None):
        """
        Initialise le visualiseur de données de Premier League
        
        Args:
            data_dir (str): Répertoire contenant les données à visualiser
            output_dir (str): Répertoire pour sauvegarder les visualisations (par défaut: sous-répertoire 'visualizations' dans data_dir)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir or os.path.join(data_dir, 'visualizations')
        
        # Création du répertoire de sortie s'il n'existe pas
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Données
        self.standings = None
        self.fixtures = None
        self.team_stats = {}
        
        # Style de visualisation
        plt.style.use('seaborn-v0_8-darkgrid')
        self.premier_league_palette = {
            'red': '#FF2882',
            'blue': '#37003C',
            'light_blue': '#00FF87',
            'purple': '#963CFF',
            'green': '#2F3E4F'
        }
        
        # Chargement des données
        self.load_latest_data()
    
    def load_latest_data(self):
        """
        Charge les données les plus récentes dans le visualiseur
        
        Returns:
            bool: True si les données ont été chargées avec succès, False sinon
        """
        try:
            # Récupérer le fichier de classement le plus récent
            standings_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('standings_')])
            if not standings_files:
                print("Aucun fichier de classement trouvé.")
                return False
                
            latest_standings = standings_files[-1]
            self.standings = pd.read_csv(os.path.join(self.data_dir, latest_standings))
            print(f"Classement chargé: {latest_standings}")
            
            # Récupérer le fichier de matchs le plus récent (tous les matchs)
            fixtures_files = sorted([f for f in os.listdir(self.data_dir) if f.startswith('fixtures_all_')])
            if fixtures_files:
                latest_fixtures = fixtures_files[-1]
                self.fixtures = pd.read_csv(os.path.join(self.data_dir, latest_fixtures))
                print(f"Calendrier des matchs chargé: {latest_fixtures}")
            
            # Charger les statistiques des équipes
            stats_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json') and 'stats' in f]
            for stat_file in stats_files:
                team_name = stat_file.split('_stats_')[0].replace('_', ' ')
                with open(os.path.join(self.data_dir, stat_file), 'r', encoding='utf-8') as f:
                    self.team_stats[team_name] = json.load(f)
                print(f"Statistiques chargées pour: {team_name}")
            
            return True
            
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return False
    
    def visualize_league_table(self, top_n=20, save=True, show=True):
        """
        Visualise le classement de la Premier League avec une présentation améliorée
        
        Args:
            top_n (int): Nombre d'équipes à afficher (défaut: toutes)
            save (bool): Sauvegarder la visualisation
            show (bool): Afficher la visualisation
            
        Returns:
            matplotlib.figure.Figure: L'objet figure créé
        """
        if self.standings is None:
            print("Aucune donnée de classement disponible.")
            return None
        
        # Limiter aux top_n premières équipes
        standings_display = self.standings.copy()
        if top_n < len(standings_display):
            standings_display = standings_display.head(top_n)
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Positions pour les zones (Champions League, Europa League, Relégation)
        cl_zone = 4  # Top 4 pour Champions League
        el_zone = 6  # 5-6 pour Europa League
        rel_zone = 17  # 18-20 pour relégation
        
        # Créer des couleurs de fond pour les différentes zones
        colors = []
        for pos in standings_display['Position']:
            if pos <= cl_zone:
                colors.append(self.premier_league_palette['light_blue'])
            elif pos <= el_zone:
                colors.append(self.premier_league_palette['purple'])
            elif pos > rel_zone:
                colors.append(self.premier_league_palette['red'])
            else:
                colors.append('white')
        
        # Créer le graphique à barres horizontales
        bars = ax.barh(standings_display['Team'], standings_display['Points'], color=self.premier_league_palette['blue'])
        
        # Ajouter les positions à côté du nom de l'équipe
        for i, (_, row) in enumerate(standings_display.iterrows()):
            ax.text(-5, i, f"{row['Position']}.", ha='right', va='center', 
                    color='black', fontweight='bold', fontsize=12)
        
        # Ajouter les statistiques W-D-L à droite des barres
        for i, (_, row) in enumerate(standings_display.iterrows()):
            stats_text = f"  {row['Won']}W - {row['Drawn']}D - {row['Lost']}L  ({row['GF']}-{row['GA']}, {row['GD']})"
            ax.text(row['Points'] + 1, i, stats_text, va='center', fontsize=10)
        
        # Ajouter les valeurs dans les barres
        for bar, points in zip(bars, standings_display['Points']):
            width = bar.get_width()
            ax.text(width / 2, bar.get_y() + bar.get_height() / 2, 
                    f"{int(width)}", ha='center', va='center', color='white', fontweight='bold')
        
        # Ajouter une légende pour les zones
        legend_elements = [
            Patch(facecolor=self.premier_league_palette['light_blue'], label='Champions League'),
            Patch(facecolor=self.premier_league_palette['purple'], label='Europa League'),
            Patch(facecolor=self.premier_league_palette['red'], label='Relegation Zone')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Ajouter titre et étiquettes
        plt.title('Premier League Table', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Points', fontsize=14, labelpad=10)
        ax.set_xlim(-5, max(standings_display['Points']) + 25)
        
        # Ajouter une grille verticale pour les points
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)  # Mettre la grille en arrière-plan
        
        # Supprimer les bordures
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Ajuster la mise en page
        plt.tight_layout()
        
        # Sauvegarder l'image
        if save:
            date_str = datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(self.output_dir, f"league_table_{date_str}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Classement visualisé et sauvegardé: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_team_form(self, team_name, num_matches=10, save=True, show=True):
        """
        Visualise la forme récente d'une équipe avec une présentation détaillée
        
        Args:
            team_name (str): Nom de l'équipe
            num_matches (int): Nombre de matchs à considérer
            save (bool): Sauvegarder la visualisation
            show (bool): Afficher la visualisation
            
        Returns:
            matplotlib.figure.Figure: L'objet figure créé
        """
        if self.fixtures is None:
            print("Aucune donnée de matchs disponible.")
            return None
        
        # Filtrer les matchs de l'équipe
        team_matches = self.fixtures[
            ((self.fixtures['HomeTeam'] == team_name) | (self.fixtures['AwayTeam'] == team_name)) & 
            (self.fixtures['Status'] == 'Joué')
        ].copy()
        
        # Convertir les scores en numériques
        for col in ['HomeScore', 'AwayScore']:
            team_matches[col] = pd.to_numeric(team_matches[col], errors='coerce')
        
        # Éliminer les matchs sans score
        team_matches = team_matches.dropna(subset=['HomeScore', 'AwayScore'])
        
        if len(team_matches) == 0:
            print(f"Aucun match joué trouvé pour {team_name}.")
            return None
        
        # Trier par date
        try:
            team_matches['Date'] = pd.to_datetime(team_matches['Date'])
            team_matches = team_matches.sort_values('Date')
        except:
            # Si la conversion de date échoue, on garde l'ordre actuel
            pass
        
        # Limiter au nombre de matchs spécifié
        if len(team_matches) > num_matches:
            team_matches = team_matches.tail(num_matches)
        
        # Préparer les données pour la visualisation
        results = []
        points = []
        opponents = []
        venues = []
        scores = []
        dates = []
        
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                # L'équipe joue à domicile
                opponent = match['AwayTeam']
                venue = 'Home'
                score = f"{int(match['HomeScore'])}-{int(match['AwayScore'])}"
                
                if match['HomeScore'] > match['AwayScore']:
                    results.append('W')
                    points.append(3)
                elif match['HomeScore'] < match['AwayScore']:
                    results.append('L')
                    points.append(0)
                else:
                    results.append('D')
                    points.append(1)
            else:
                # L'équipe joue à l'extérieur
                opponent = match['HomeTeam']
                venue = 'Away'
                score = f"{int(match['AwayScore'])}-{int(match['HomeScore'])}"
                
                if match['AwayScore'] > match['HomeScore']:
                    results.append('W')
                    points.append(3)
                elif match['AwayScore'] < match['HomeScore']:
                    results.append('L')
                    points.append(0)
                else:
                    results.append('D')
                    points.append(1)
            
            opponents.append(opponent)
            venues.append(venue)
            scores.append(score)
            
            # Formater la date
            if isinstance(match['Date'], pd.Timestamp):
                dates.append(match['Date'].strftime('%d %b'))
            else:
                dates.append(str(match['Date']))
        
        # Créer la figure avec plusieurs sous-graphiques
        fig = plt.figure(figsize=(14, 10))
        
        # Grille pour les sous-graphiques
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 2], hspace=0.3)
        
        # Sous-graphique 1: Résultats récents
        ax1 = fig.add_subplot(gs[0])
        
        # Définir les couleurs pour les résultats
        result_colors = {'W': 'green', 'D': 'gray', 'L': 'red'}
        bar_colors = [result_colors[r] for r in results]
        
        # Créer le graphique à barres pour les résultats
        form_bars = ax1.bar(range(len(results)), [1] * len(results), color=bar_colors, width=0.6)
        
        # Ajouter les labels sur chaque barre
        for i, (res, opp, score, date) in enumerate(zip(results, opponents, scores, dates)):
            ax1.text(i, 0.5, res, ha='center', va='center', color='white', fontweight='bold', fontsize=14)
            if venues[i] == 'Home':
                vs_text = f"vs {opp} (H)"
            else:
                vs_text = f"vs {opp} (A)"
            ax1.text(i, 1.2, vs_text, ha='center', va='center', color='black', fontsize=9, rotation=90)
            ax1.text(i, 0.2, score, ha='center', va='center', color='white', fontsize=9)
            ax1.text(i, -0.2, date, ha='center', va='center', color='black', fontsize=8)
        
        # Configurer l'axe des résultats
        ax1.set_ylim(0, 1.5)
        ax1.set_xlim(-0.5, len(results) - 0.5)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f'Recent Form: {team_name}', fontsize=16, fontweight='bold', pad=15)
        
        for spine in ax1.spines.values():
            spine.set_visible(False)
        
        # Sous-graphique 2: Points cumulés
        ax2 = fig.add_subplot(gs[1])
        
        # Calculer les points cumulés
        cumulative_points = np.cumsum(points)
        
        # Créer le graphique de ligne pour les points cumulés
        ax2.plot(range(len(cumulative_points)), cumulative_points, marker='o', linewidth=3, 
                color=self.premier_league_palette['blue'])
        
        # Ajouter les points à chaque marqueur
        for i, p in enumerate(cumulative_points):
            ax2.text(i, p + 0.3, str(p), ha='center', va='bottom', fontweight='bold')
        
        # Configurer l'axe des points
        ax2.set_xlim(-0.5, len(cumulative_points) - 0.5)
        ax2.set_xticks(range(len(cumulative_points)))
        ax2.set_xticklabels([f"Match {i+1}" for i in range(len(cumulative_points))])
        ax2.set_ylabel('Points', fontsize=12)
        ax2.set_title('Cumulative Points', fontsize=14, pad=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Sous-graphique 3: Performances à domicile vs extérieur
        ax3 = fig.add_subplot(gs[2])
        
        # Séparer les résultats à domicile et à l'extérieur
        home_results = [r for r, v in zip(results, venues) if v == 'Home']
        away_results = [r for r, v in zip(results, venues) if v == 'Away']
        
        # Compter les victoires, nuls et défaites
        home_counts = {'W': home_results.count('W'), 'D': home_results.count('D'), 'L': home_results.count('L')}
        away_counts = {'W': away_results.count('W'), 'D': away_results.count('D'), 'L': away_results.count('L')}
        
        # Préparer les données pour le graphique à barres groupées
        performance_data = {
            'Home': [home_counts['W'], home_counts['D'], home_counts['L']],
            'Away': [away_counts['W'], away_counts['D'], away_counts['L']]
        }
        
        # Positions des barres
        categories = ['Wins', 'Draws', 'Losses']
        x = np.arange(len(categories))
        width = 0.35
        
        # Créer les barres
        ax3.bar(x - width/2, performance_data['Home'], width, label='Home', color=self.premier_league_palette['light_blue'])
        ax3.bar(x + width/2, performance_data['Away'], width, label='Away', color=self.premier_league_palette['purple'])
        
        # Ajouter les labels et légendes
        ax3.set_title('Home vs Away Performance', fontsize=14, pad=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(performance_data['Home']):
            ax3.text(i - width/2, v + 0.1, str(v), ha='center', va='bottom')
        
        for i, v in enumerate(performance_data['Away']):
            ax3.text(i + width/2, v + 0.1, str(v), ha='center', va='bottom')
        
        # Ajouter un titre global
        fig.suptitle(f'{team_name} Performance Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # Ajuster la mise en page
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarder l'image
        if save:
            date_str = datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(self.output_dir, f"{team_name.replace(' ', '_').lower()}_form_{date_str}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Forme de l'équipe visualisée et sauvegardée: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_team_comparison(self, team1, team2, save=True, show=True):
        """
        Visualise une comparaison complète entre deux équipes
        
        Args:
            team1 (str): Nom de la première équipe
            team2 (str): Nom de la deuxième équipe
            save (bool): Sauvegarder la visualisation
            show (bool): Afficher la visualisation
            
        Returns:
            matplotlib.figure.Figure: L'objet figure créé
        """
        if self.standings is None:
            print("Aucune donnée de classement disponible.")
            return None
        
        # Vérifier si les équipes existent
        if team1 not in self.standings['Team'].values or team2 not in self.standings['Team'].values:
            print(f"Une ou plusieurs équipes non trouvées: {team1}, {team2}")
            return None
        
        # Extraire les données des équipes
        team1_data = self.standings[self.standings['Team'] == team1].iloc[0]
        team2_data = self.standings[self.standings['Team'] == team2].iloc[0]
        
        # Créer la figure avec plusieurs sous-graphiques
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 2, 2], width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        
        # Sous-graphique 1: Comparaison des positions et points
        ax1 = fig.add_subplot(gs[0, :])
        
        # Données pour le graphique à barres groupées
        standings_metrics = ['Position', 'Points', 'Played', 'Won', 'Drawn', 'Lost', 'GF', 'GA']
        standings_values1 = [team1_data[metric] for metric in standings_metrics]
        standings_values2 = [team2_data[metric] for metric in standings_metrics]
        
        # Positions des barres
        x = np.arange(len(standings_metrics))
        width = 0.35
        
        # Inverser la position (plus bas = meilleur)
        standings_values1[0] = 21 - standings_values1[0]  # 20 équipes max en PL
        standings_values2[0] = 21 - standings_values2[0]
        
        # Créer les barres
        bars1 = ax1.bar(x - width/2, standings_values1, width, label=team1, color=self.premier_league_palette['light_blue'])
        bars2 = ax1.bar(x + width/2, standings_values2, width, label=team2, color=self.premier_league_palette['purple'])
        
        # Ajouter les labels et légendes
        ax1.set_title('Season Performance Comparison', fontsize=16, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        standings_labels = ['Position*', 'Points', 'Played', 'Won', 'Drawn', 'Lost', 'Goals For', 'Goals Against']
        ax1.set_xticklabels(standings_labels)
        ax1.legend()
        
        # Ajouter note pour l'inversion de la position
        ax1.text(0, -1, '* Position is inverted (higher is better)', fontsize=8, style='italic')
        
        # Ajouter les valeurs sur les barres (récupérer les vraies positions)
        for i, bar in enumerate(bars1):
            if i == 0:  # Position
                value = 21 - standings_values1[i]  # Reconvertir
            else:
                value = standings_values1[i]
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(int(value)), 
                     ha='center', va='bottom', fontsize=9)
        
        for i, bar in enumerate(bars2):
            if i == 0:  # Position
                value = 21 - standings_values2[i]  # Reconvertir
            else:
                value = standings_values2[i]
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(int(value)), 
                     ha='center', va='bottom', fontsize=9)
        
        # Sous-graphique 2: Graphique radar des métriques
        ax2 = fig.add_subplot(gs[1, 0], polar=True)
        
        # Choisir les métriques pour le radar
        radar_metrics = ['Points/Game', 'Win %', 'Goal Diff', 'Goals/Game', 'Clean Sheets %']
        
        # Calculer les métriques normalisées
        ppg1 = team1_data['Points'] / team1_data['Played'] if team1_data['Played'] > 0 else 0
        ppg2 = team2_data['Points'] / team2_data['Played'] if team2_data['Played'] > 0 else 0
        
        win_pct1 = team1_data['Won'] / team1_data['Played'] * 100 if team1_data['Played'] > 0 else 0
        win_pct2 = team2_data['Won'] / team2_data['Played'] * 100 if team2_data['Played'] > 0 else 0
        
        goals_per_game1 = team1_data['GF'] / team1_data['Played'] if team1_data['Played'] > 0 else 0
        goals_per_game2 = team2_data['GF'] / team2_data['Played'] if team2_data['Played'] > 0 else 0
        
        # Calculer le pourcentage de clean sheets (estimation car nous n'avons pas les détails des matchs)
        # On estime que 20% des matchs sont des clean sheets pour la simplicité
        clean_sheet_pct1 = team1_data['GA'] / team1_data['Played'] * 20 if team1_data['Played'] > 0 else 0
        clean_sheet_pct2 = team2_data['GA'] / team2_data['Played'] * 20 if team2_data['Played'] > 0 else 0
        
        # Obtenir les données pour le radar
        max_ppg = 3.0  # Maximum théorique de points par match
        max_win_pct = 100.0
        max_gd = max(abs(team1_data['GD']), abs(team2_data['GD'])) * 1.2 or 10
        max_gpg = max(goals_per_game1, goals_per_game2) * 1.2 or 3
        max_cs_pct = 100.0
        
        # Normaliser les données
        radar_values1 = [
            ppg1 / max_ppg,
            win_pct1 / max_win_pct,
            (team1_data['GD'] + max_gd) / (2 * max_gd),  # Normaliser entre 0 et 1
            goals_per_game1 / max_gpg,
            clean_sheet_pct1 / max_cs_pct
        ]
        
        radar_values2 = [
            ppg2 / max_ppg,
            win_pct2 / max_win_pct,
            (team2_data['GD'] + max_gd) / (2 * max_gd),  # Normaliser entre 0 et 1
            goals_per_game2 / max_gpg,
            clean_sheet_pct2 / max_cs_pct
        ]
        
        # Angles pour le radar
        angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
        
        # Fermer le graphique radar
        radar_values1 += [radar_values1[0]]
        radar_values2 += [radar_values2[0]]
        angles += [angles[0]]
        
        # Tracer le radar
        ax2.plot(angles, radar_values1, 'o-', linewidth=2, color=self.premier_league_palette['light_blue'], label=team1)
        ax2.fill(angles, radar_values1, alpha=0.25, color=self.premier_league_palette['light_blue'])
        
        ax2.plot(angles, radar_values2, 'o-', linewidth=2, color=self.premier_league_palette['purple'], label=team2)
        ax2.fill(angles, radar_values2, alpha=0.25, color=self.premier_league_palette['purple'])
        
        # Ajouter les métriques et labels
        ax2.set_thetagrids(np.degrees(angles[:-1]), radar_metrics)
        ax2.set_rlim(0, 1.1)
        ax2.set_title('Performance Metrics Radar', fontsize=14, pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Sous-graphique 3: Comparaison des formes récentes
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Visualisation des résultats récents
        if self.fixtures is not None:
            # Obtenir les 5 derniers résultats de chaque équipe
            team1_results = self._get_recent_results(team1, 5)
            team2_results = self._get_recent_results(team2, 5)
            
            # Positions des résultats
            result_positions = np.arange(5)
            
            # Couleurs des résultats
            team1_colors = ['green' if r == 'W' else ('gray' if r == 'D' else 'red') for r in team1_results]
            team2_colors = ['green' if r == 'W' else ('gray' if r == 'D' else 'red') for r in team2_results]
            
            # Barres pour team1
            ax3.bar(result_positions - 0.2, [0.8] * len(team1_results), width=0.4, color=team1_colors, 
                    alpha=0.8, label=f"{team1} Results")
            
            # Barres pour team2
            ax3.bar(result_positions + 0.2, [0.8] * len(team2_results), width=0.4, color=team2_colors, 
                    alpha=0.8, label=f"{team2} Results")
            
            # Ajouter les labels sur les barres
            for i, result in enumerate(team1_results):
                ax3.text(i - 0.2, 0.4, result, ha='center', va='center', color='white', fontweight='bold')
            
            for i, result in enumerate(team2_results):
                ax3.text(i + 0.2, 0.4, result, ha='center', va='center', color='white', fontweight='bold')
            
            # Configuration de l'axe des résultats
            ax3.set_ylim(0, 1)
            ax3.set_xlim(-0.5, 4.5)
            ax3.set_xticks(result_positions)
            ax3.set_xticklabels(['5th', '4th', '3rd', '2nd', 'Last'])
            ax3.set_yticks([])
            ax3.set_title('Last 5 Results', fontsize=14, pad=10)
            ax3.legend()
            
            # Supprimer les bordures
            for spine in ax3.spines.values():
                spine.set_visible(False)
        
        # Sous-graphique 4: Comparaison des buts marqués/concédés
        ax4 = fig.add_subplot(gs[2, 0])
        
        # Données pour les buts
        goals_metrics = ['Goals For', 'Goals Against', 'Goal Difference']
        goals_values1 = [team1_data['GF'], team1_data['GA'], team1_data['GD']]
        goals_values2 = [team2_data['GF'], team2_data['GA'], team2_data['GD']]
        
        # Positions des barres
        x = np.arange(len(goals_metrics))
        width = 0.35
        
        # Créer les barres
        ax4.bar(x - width/2, goals_values1, width, color=self.premier_league_palette['light_blue'], label=team1)
        ax4.bar(x + width/2, goals_values2, width, color=self.premier_league_palette['purple'], label=team2)
        
        # Ajouter les labels et légendes
        ax4.set_title('Goals Comparison', fontsize=14, pad=10)
        ax4.set_xticks(x)
        ax4.set_xticklabels(goals_metrics)
        ax4.legend()
        ax4.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Sous-graphique 5: Head-to-Head
        ax5 = fig.add_subplot(gs[2, 1])
        
        # Obtenir les confrontations directes
        if self.fixtures is not None:
            h2h_matches = self.fixtures[
                ((self.fixtures['HomeTeam'] == team1) & (self.fixtures['AwayTeam'] == team2)) |
                ((self.fixtures['HomeTeam'] == team2) & (self.fixtures['AwayTeam'] == team1))
            ].copy()
            
            # Convertir les scores en numériques
            for col in ['HomeScore', 'AwayScore']:
                h2h_matches[col] = pd.to_numeric(h2h_matches[col], errors='coerce')
            
            # Filtrer les matchs joués
            h2h_matches = h2h_matches.dropna(subset=['HomeScore', 'AwayScore'])
            
            if len(h2h_matches) > 0:
                # Compter les victoires de chaque équipe
                team1_wins = 0
                team2_wins = 0
                draws = 0
                
                for _, match in h2h_matches.iterrows():
                    if match['HomeTeam'] == team1:
                        if match['HomeScore'] > match['AwayScore']:
                            team1_wins += 1
                        elif match['HomeScore'] < match['AwayScore']:
                            team2_wins += 1
                        else:
                            draws += 1
                    else:
                        if match['HomeScore'] > match['AwayScore']:
                            team2_wins += 1
                        elif match['HomeScore'] < match['AwayScore']:
                            team1_wins += 1
                        else:
                            draws += 1
                
                # Créer un graphique en camembert pour les résultats H2H
                h2h_labels = [f"{team1} Wins", "Draws", f"{team2} Wins"]
                h2h_sizes = [team1_wins, draws, team2_wins]
                h2h_colors = [self.premier_league_palette['light_blue'], 'gray', self.premier_league_palette['purple']]
                
                # Éviter la division par zéro
                if sum(h2h_sizes) > 0:
                    ax5.pie(h2h_sizes, labels=h2h_labels, colors=h2h_colors, autopct='%1.1f%%', 
                           startangle=90, wedgeprops={'edgecolor': 'w'})
                    ax5.set_title('Head-to-Head Results', fontsize=14, pad=10)
                else:
                    ax5.text(0.5, 0.5, "No head-to-head matches found", ha='center', va='center', 
                            fontsize=12, transform=ax5.transAxes)
            else:
                ax5.text(0.5, 0.5, "No head-to-head matches found", ha='center', va='center', 
                        fontsize=12, transform=ax5.transAxes)
        else:
            ax5.text(0.5, 0.5, "No fixtures data available", ha='center', va='center', 
                    fontsize=12, transform=ax5.transAxes)
        
        # Ajouter un titre global
        fig.suptitle(f'{team1} vs {team2} - Season Comparison', fontsize=18, fontweight='bold', y=0.98)
        
        # Ajuster la mise en page
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarder l'image
        if save:
            date_str = datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(self.output_dir, f"{team1.replace(' ', '_').lower()}_vs_{team2.replace(' ', '_').lower()}_{date_str}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Comparaison d'équipes visualisée et sauvegardée: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_goal_distribution(self, save=True, show=True):
        """
        Visualise la distribution des buts dans la ligue
        
        Args:
            save (bool): Sauvegarder la visualisation
            show (bool): Afficher la visualisation
            
        Returns:
            matplotlib.figure.Figure: L'objet figure créé
        """
        if self.fixtures is None:
            print("Aucune donnée de matchs disponible.")
            return None
        
        # Filtrer les matchs joués
        played_matches = self.fixtures[self.fixtures['Status'] == 'Joué'].copy()
        
        # Convertir les scores en numériques
        for col in ['HomeScore', 'AwayScore']:
            played_matches[col] = pd.to_numeric(played_matches[col], errors='coerce')
        
        # Éliminer les matchs sans score
        played_matches = played_matches.dropna(subset=['HomeScore', 'AwayScore'])
        
        if len(played_matches) == 0:
            print("Aucun match joué trouvé.")
            return None
        
        # Créer la figure avec plusieurs sous-graphiques
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Sous-graphique 1: Distribution des buts par match
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Calculer le nombre total de buts par match
        played_matches['TotalGoals'] = played_matches['HomeScore'] + played_matches['AwayScore']
        
        # Créer l'histogramme des buts par match
        sns.histplot(played_matches['TotalGoals'], bins=range(0, int(played_matches['TotalGoals'].max()) + 2), 
                   kde=True, color=self.premier_league_palette['blue'], ax=ax1)
        
        ax1.set_title('Goals per Match Distribution', fontsize=14, pad=10)
        ax1.set_xlabel('Total Goals')
        ax1.set_ylabel('Number of Matches')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Sous-graphique 2: Buts à domicile vs extérieur
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Calculer les moyennes
        home_goals_avg = played_matches['HomeScore'].mean()
        away_goals_avg = played_matches['AwayScore'].mean()
        
        # Créer les barres
        ax2.bar(['Home Goals', 'Away Goals'], [home_goals_avg, away_goals_avg], 
               color=[self.premier_league_palette['light_blue'], self.premier_league_palette['purple']])
        
        ax2.set_title('Average Goals: Home vs Away', fontsize=14, pad=10)
        ax2.set_ylabel('Average Goals per Match')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Ajouter les valeurs sur les barres
        ax2.text(0, home_goals_avg + 0.05, f"{home_goals_avg:.2f}", ha='center', va='bottom')
        ax2.text(1, away_goals_avg + 0.05, f"{away_goals_avg:.2f}", ha='center', va='bottom')
        
        # Sous-graphique 3: Matrice de fréquence des scores
        ax3 = fig.add_subplot(gs[1, :])
        
        # Créer une matrice de fréquence des scores
        max_score = max(played_matches['HomeScore'].max(), played_matches['AwayScore'].max())
        score_matrix = np.zeros((int(max_score) + 1, int(max_score) + 1))
        
        for _, match in played_matches.iterrows():
            home_score = int(match['HomeScore'])
            away_score = int(match['AwayScore'])
            score_matrix[home_score, away_score] += 1
        
        # Normaliser la matrice
        score_matrix = score_matrix / len(played_matches) * 100
        
        # Créer une heatmap
        sns.heatmap(score_matrix, cmap='YlGnBu', annot=True, fmt='.1f', ax=ax3)
        
        ax3.set_title('Score Frequency Matrix (% of matches)', fontsize=14, pad=10)
        ax3.set_xlabel('Away Goals')
        ax3.set_ylabel('Home Goals')
        
        # Ajouter un titre global
        fig.suptitle('Premier League Goal Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # Ajuster la mise en page
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarder l'image
        if save:
            date_str = datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(self.output_dir, f"goal_distribution_{date_str}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Distribution des buts visualisée et sauvegardée: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_team_stats(self, team_name, save=True, show=True):
        """
        Visualise les statistiques détaillées d'une équipe
        
        Args:
            team_name (str): Nom de l'équipe
            save (bool): Sauvegarder la visualisation
            show (bool): Afficher la visualisation
            
        Returns:
            matplotlib.figure.Figure: L'objet figure créé
        """
        if team_name.lower() not in self.team_stats and team_name not in self.team_stats:
            print(f"Aucune statistique détaillée trouvée pour {team_name}.")
            return None
        
        # Récupérer les statistiques de l'équipe
        team_stats = self.team_stats.get(team_name.lower(), self.team_stats.get(team_name))
        
        # Vérifier si les données sont dans le bon format
        if not isinstance(team_stats, dict) or 'stats' not in team_stats or 'top_players' not in team_stats:
            print(f"Format de statistiques incorrect pour {team_name}.")
            return None
        
        # Créer la figure avec plusieurs sous-graphiques
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # Sous-graphique 1: Statistiques générales
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Extraire les statistiques générales (prendre la première catégorie disponible)
        general_stats_key = list(team_stats['stats'].keys())[0] if team_stats['stats'] else None
        
        if general_stats_key and team_stats['stats'][general_stats_key]:
            # Sélectionner jusqu'à 5 statistiques
            general_stats = dict(list(team_stats['stats'][general_stats_key].items())[:5])
            
            # Créer un graphique à barres horizontal
            y_pos = range(len(general_stats))
            ax1.barh(y_pos, list(general_stats.values()), color=self.premier_league_palette['blue'])
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(list(general_stats.keys()))
            ax1.invert_yaxis()  # Les plus hautes valeurs en haut
            
            ax1.set_title(f'General {general_stats_key}', fontsize=14, pad=10)
            ax1.grid(True, linestyle='--', alpha=0.7, axis='x')
        else:
            ax1.text(0.5, 0.5, "No general statistics available", ha='center', va='center', 
                    fontsize=12, transform=ax1.transAxes)
        
        # Sous-graphique 2: Meilleurs joueurs
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Extraire les meilleurs joueurs (prendre la première catégorie disponible)
        top_players_key = list(team_stats['top_players'].keys())[0] if team_stats['top_players'] else None
        
        if top_players_key and team_stats['top_players'][top_players_key]:
            # Sélectionner jusqu'à 5 joueurs
            top_players = team_stats['top_players'][top_players_key][:5]
            
            # Extraire les noms et les valeurs
            player_names = [player['name'] for player in top_players]
            player_values = [player['value'] for player in top_players]
            
            # Créer un graphique à barres horizontal
            y_pos = range(len(player_names))
            ax2.barh(y_pos, player_values, color=self.premier_league_palette['purple'])
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(player_names)
            ax2.invert_yaxis()  # Les plus hautes valeurs en haut
            
            ax2.set_title(f'Top Players: {top_players_key}', fontsize=14, pad=10)
            ax2.grid(True, linestyle='--', alpha=0.7, axis='x')
        else:
            ax2.text(0.5, 0.5, "No player statistics available", ha='center', va='center', 
                    fontsize=12, transform=ax2.transAxes)
        
        # Ajouter des sous-graphiques supplémentaires si d'autres catégories sont disponibles
        row, col = 1, 0
        
        # Parcourir les autres catégories de statistiques générales
        for i, stats_key in enumerate(list(team_stats['stats'].keys())[1:], 1):
            if row >= 3:  # Limiter à 6 sous-graphiques au total
                break
                
            ax = fig.add_subplot(gs[row, col])
            
            if team_stats['stats'][stats_key]:
                # Sélectionner jusqu'à 5 statistiques
                stats = dict(list(team_stats['stats'][stats_key].items())[:5])
                
                # Créer un graphique à barres horizontal
                y_pos = range(len(stats))
                ax.barh(y_pos, list(stats.values()), color=self.premier_league_palette['blue'])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(list(stats.keys()))
                ax.invert_yaxis()  # Les plus hautes valeurs en haut
                
                ax.set_title(f'{stats_key}', fontsize=14, pad=10)
                ax.grid(True, linestyle='--', alpha=0.7, axis='x')
            else:
                ax.text(0.5, 0.5, f"No {stats_key} statistics available", ha='center', va='center', 
                        fontsize=12, transform=ax.transAxes)
            
            # Passer à la colonne/ligne suivante
            col = (col + 1) % 2
            if col == 0:
                row += 1
        
        # Parcourir les autres catégories de meilleurs joueurs
        for i, players_key in enumerate(list(team_stats['top_players'].keys())[1:], 1):
            if row >= 3:  # Limiter à 6 sous-graphiques au total
                break
                
            ax = fig.add_subplot(gs[row, col])
            
            if team_stats['top_players'][players_key]:
                # Sélectionner jusqu'à 5 joueurs
                players = team_stats['top_players'][players_key][:5]
                
                # Extraire les noms et les valeurs
                player_names = [player['name'] for player in players]
                player_values = [player['value'] for player in players]
                
                # Créer un graphique à barres horizontal
                y_pos = range(len(player_names))
                ax.barh(y_pos, player_values, color=self.premier_league_palette['purple'])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(player_names)
                ax.invert_yaxis()  # Les plus hautes valeurs en haut
                
                ax.set_title(f'Top Players: {players_key}', fontsize=14, pad=10)
                ax.grid(True, linestyle='--', alpha=0.7, axis='x')
            else:
                ax.text(0.5, 0.5, f"No {players_key} statistics available", ha='center', va='center', 
                        fontsize=12, transform=ax.transAxes)
            
            # Passer à la colonne/ligne suivante
            col = (col + 1) % 2
            if col == 0:
                row += 1
        
        # Ajouter un titre global
        fig.suptitle(f'{team_name} - Detailed Statistics', fontsize=18, fontweight='bold', y=0.98)
        
        # Ajuster la mise en page
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarder l'image
        if save:
            date_str = datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(self.output_dir, f"{team_name.replace(' ', '_').lower()}_detailed_stats_{date_str}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Statistiques détaillées visualisées et sauvegardées: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def visualize_season_projections(self, save=True, show=True):
        """
        Visualise les projections de fin de saison
        
        Args:
            save (bool): Sauvegarder la visualisation
            show (bool): Afficher la visualisation
            
        Returns:
            matplotlib.figure.Figure: L'objet figure créé
        """
        if self.standings is None:
            print("Aucune donnée de classement disponible.")
            return None
        
        # Créer des projections simples basées sur les points par match actuels
        projected_standings = self.standings.copy()
        
        # Calculer les points par match pour chaque équipe
        projected_standings['PPM'] = projected_standings['Points'] / projected_standings['Played']
        
        # Nombre de matchs total dans la saison
        total_matches = 38
        
        # Projeter les points de fin de saison
        projected_standings['RemainingMatches'] = total_matches - projected_standings['Played']
        projected_standings['ProjectedPoints'] = projected_standings['Points'] + (projected_standings['PPM'] * projected_standings['RemainingMatches'])
        projected_standings['ProjectedPoints'] = projected_standings['ProjectedPoints'].round().astype(int)
        
        # Trier par points projetés
        projected_standings = projected_standings.sort_values('ProjectedPoints', ascending=False).reset_index(drop=True)
        projected_standings['ProjectedPosition'] = projected_standings.index + 1
        
        # Calculer la différence de position
        position_map = {row['Team']: row['Position'] for _, row in self.standings.iterrows()}
        projected_standings['PositionChange'] = projected_standings.apply(
            lambda x: position_map[x['Team']] - x['ProjectedPosition'], axis=1
        )
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Positions pour les zones (Champions League, Europa League, Relégation)
        cl_pos = 4  # Top 4 pour Champions League
        el_pos = 6  # 5-6 pour Europa League
        rel_pos = 17  # 18-20 pour relégation
        
        # Barres pour les points actuels
        bars_current = ax.barh(projected_standings['Team'], projected_standings['Points'], 
                              color=self.premier_league_palette['blue'], alpha=0.6, label='Current Points')
        
        # Barres pour les points projetés (à partir des points actuels)
        bars_remaining = ax.barh(projected_standings['Team'], 
                                projected_standings['ProjectedPoints'] - projected_standings['Points'], 
                                left=projected_standings['Points'], 
                                color=self.premier_league_palette['light_blue'], alpha=0.8, 
                                label='Projected Additional Points')
        
        # Légende
        ax.legend(loc='lower right')
        
        # Ajouter des zones de couleur pour les différentes qualifications
        height = 0.6
        
        # Champions League (top 4)
        for i, team in enumerate(projected_standings['Team'][:cl_pos]):
            ax.axhspan(i - height/2, i + height/2, color=self.premier_league_palette['light_blue'], alpha=0.2, zorder=0)
            
        # Europa League (5-6)
        for i, team in enumerate(projected_standings['Team'][cl_pos:el_pos], cl_pos):
            ax.axhspan(i - height/2, i + height/2, color=self.premier_league_palette['purple'], alpha=0.2, zorder=0)
            
        # Relégation (18-20)
        for i, team in enumerate(projected_standings['Team'][-3:], len(projected_standings) - 3):
            ax.axhspan(i - height/2, i + height/2, color=self.premier_league_palette['red'], alpha=0.2, zorder=0)
        
        # Ajouter les positions actuelles et projetées, ainsi que les changements à côté du nom de l'équipe
        for i, (_, team) in enumerate(projected_standings.iterrows()):
            current_pos = position_map[team['Team']]
            projected_pos = team['ProjectedPosition']
            pos_change = team['PositionChange']
            
            # Flèche pour indiquer la direction du changement
            if pos_change > 0:
                change_text = f"↑{pos_change}"
                change_color = 'green'
            elif pos_change < 0:
                change_text = f"↓{abs(pos_change)}"
                change_color = 'red'
            else:
                change_text = "="
                change_color = 'gray'
            
            # Texte avec position actuelle et projetée
            ax.text(-5, i, f"{current_pos}→{projected_pos} {change_text}", 
                   ha='right', va='center', fontsize=10, color=change_color)
            
            # Ajouter les points projetés à la fin des barres
            ax.text(team['ProjectedPoints'] + 1, i, f"{team['ProjectedPoints']} pts", 
                   ha='left', va='center', fontsize=10)
        
        # Configurer l'axe
        ax.set_xlabel('Points', fontsize=14, labelpad=10)
        ax.set_xlim(-10, max(projected_standings['ProjectedPoints']) + 20)
        
        # Ajouter une légende pour les zones
        legend_elements = [
            Patch(facecolor=self.premier_league_palette['light_blue'], alpha=0.2, label='Champions League (Top 4)'),
            Patch(facecolor=self.premier_league_palette['purple'], alpha=0.2, label='Europa League (5-6)'),
            Patch(facecolor=self.premier_league_palette['red'], alpha=0.2, label='Relegation Zone (18-20)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Ajouter une grille verticale pour les points
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)  # Mettre la grille en arrière-plan
        
        # Supprimer les bordures
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Ajouter un titre
        fig.suptitle('Premier League Season Projections', fontsize=18, fontweight='bold', y=0.98)
        plt.title(f"Based on current points-per-game ({projected_standings['Played'].mean():.1f} games played on average)", 
                 fontsize=12, pad=10)
        
        # Ajuster la mise en page
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Sauvegarder l'image
        if save:
            date_str = datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(self.output_dir, f"season_projections_{date_str}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Projections de fin de saison visualisées et sauvegardées: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
        
    def _get_recent_results(self, team_name, num_matches=5):
        """
        Obtient les résultats récents d'une équipe
        
        Args:
            team_name (str): Nom de l'équipe
            num_matches (int): Nombre de matchs à considérer
            
        Returns:
            list: Liste des résultats ('W', 'D', 'L')
        """
        if self.fixtures is None:
            return []
        
        # Filtrer les matchs de l'équipe
        team_matches = self.fixtures[
            ((self.fixtures['HomeTeam'] == team_name) | (self.fixtures['AwayTeam'] == team_name)) & 
            (self.fixtures['Status'] == 'Joué')
        ].copy()
        
        # Convertir les scores en numériques
        for col in ['HomeScore', 'AwayScore']:
            team_matches[col] = pd.to_numeric(team_matches[col], errors='coerce')
        
        # Éliminer les matchs sans score
        team_matches = team_matches.dropna(subset=['HomeScore', 'AwayScore'])
        
        if len(team_matches) == 0:
            return ['?'] * num_matches
        
        # Trier par date
        try:
            team_matches['Date'] = pd.to_datetime(team_matches['Date'])
            team_matches = team_matches.sort_values('Date', ascending=False)
        except:
            # Si la conversion de date échoue, on garde l'ordre actuel
            pass
        
        # Prendre les N derniers matchs
        recent_matches = team_matches.head(num_matches)
        
        # Déterminer les résultats
        results = []
        
        for _, match in recent_matches.iterrows():
            if match['HomeTeam'] == team_name:
                # L'équipe joue à domicile
                if match['HomeScore'] > match['AwayScore']:
                    results.append('W')
                elif match['HomeScore'] < match['AwayScore']:
                    results.append('L')
                else:
                    results.append('D')
            else:
                # L'équipe joue à l'extérieur
                if match['AwayScore'] > match['HomeScore']:
                    results.append('W')
                elif match['AwayScore'] < match['HomeScore']:
                    results.append('L')
                else:
                    results.append('D')
            
        # Compléter avec des '?' si nécessaire
        while len(results) < num_matches:
            results.append('?')
        
        return results

    def generate_dashboard(self, team_name=None, save=True, show=True):
        """
        Génère un tableau de bord avec plusieurs visualisations
        
        Args:
            team_name (str, optional): Nom de l'équipe pour un tableau de bord spécifique
            save (bool): Sauvegarder la visualisation
            show (bool): Afficher la visualisation
            
        Returns:
            matplotlib.figure.Figure: L'objet figure créé
        """
        if team_name:
            return self._generate_team_dashboard(team_name, save, show)
        else:
            return self._generate_league_dashboard(save, show)
    
    def _generate_team_dashboard(self, team_name, save=True, show=True):
        """
        Génère un tableau de bord spécifique à une équipe
        
        Args:
            team_name (str): Nom de l'équipe
            save (bool): Sauvegarder la visualisation
            show (bool): Afficher la visualisation
            
        Returns:
            matplotlib.figure.Figure: L'objet figure créé
        """
        if self.standings is None or team_name not in self.standings['Team'].values:
            print(f"Aucune donnée disponible pour {team_name}.")
            return None
        
        # Créer une figure avec une grille de sous-graphiques
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.2], wspace=0.4, hspace=0.5)
        
        # Extraire les données de l'équipe
        team_data = self.standings[self.standings['Team'] == team_name].iloc[0]
        
        # Sous-graphique 1: Position au classement (grand)
        ax1 = fig.add_subplot(gs[0, :3])
        
        position = team_data['Position']
        
        # Couleur basée sur la position
        if position <= 4:
            color = self.premier_league_palette['light_blue']
            position_text = "Champions League"
        elif position <= 6:
            color = self.premier_league_palette['purple']
            position_text = "Europa League"
        elif position >= 18:
            color = self.premier_league_palette['red']
            position_text = "Relegation Zone"
        else:
            color = 'gray'
            position_text = "Mid-table"
        
        # Afficher la position
        ax1.text(0.5, 0.5, str(position), ha='center', va='center', fontsize=100, color=color)
        ax1.text(0.5, 0.8, "POSITION", ha='center', va='center', fontsize=18, fontweight='bold')
        ax1.text(0.5, 0.2, position_text, ha='center', va='center', fontsize=14, color=color)
        
        # Supprimer les axes
        ax1.axis('off')
        
        # Sous-graphique 2: Points (grand)
        ax2 = fig.add_subplot(gs[0, 3:])
        
        points = team_data['Points']
        matches_played = team_data['Played']
        ppg = points / matches_played if matches_played > 0 else 0
        
        # Afficher les points
        ax2.text(0.5, 0.5, str(points), ha='center', va='center', fontsize=100, color=self.premier_league_palette['blue'])
        ax2.text(0.5, 0.8, "POINTS", ha='center', va='center', fontsize=18, fontweight='bold')
        ax2.text(0.5, 0.2, f"{ppg:.2f} points per game", ha='center', va='center', fontsize=14)
        
        # Supprimer les axes
        ax2.axis('off')
        
        # Sous-graphique 3: Résumé des résultats
        ax3 = fig.add_subplot(gs[1, :2])
        
        # Données pour le graphique à secteurs
        labels = ['Won', 'Drawn', 'Lost']
        sizes = [team_data['Won'], team_data['Drawn'], team_data['Lost']]
        colors = ['green', 'gray', 'red']
        
        # Créer le graphique à secteurs
        ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'w'})
        ax3.set_title('Match Results', fontsize=14, pad=10)
        
        # Sous-graphique 4: Buts
        ax4 = fig.add_subplot(gs[1, 2:4])
        
        # Données pour le graphique à barres
        goal_labels = ['Goals For', 'Goals Against', 'Goal Diff']
        goal_values = [team_data['GF'], team_data['GA'], team_data['GD']]
        
        # Couleurs en fonction des valeurs
        goal_colors = ['green', 'red', 'blue' if team_data['GD'] >= 0 else 'red']
        
        # Créer le graphique à barres
        bars = ax4.bar(goal_labels, goal_values, color=goal_colors)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{int(height) if height >= 0 else int(height)}",
                    ha='center', va='bottom')
        
        ax4.set_title('Goal Statistics', fontsize=14, pad=10)
        ax4.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Sous-graphique 5: Forme récente
        ax5 = fig.add_subplot(gs[1, 4:])
        
        # Obtenir les résultats récents
        recent_results = self._get_recent_results(team_name, 5)
        
        # Définir les couleurs pour les résultats
        result_colors = {'W': 'green', 'D': 'gray', 'L': 'red', '?': 'white'}
        bar_colors = [result_colors[r] for r in recent_results]
        
        # Créer le graphique à barres pour les résultats
        form_bars = ax5.bar(range(len(recent_results)), [1] * len(recent_results), color=bar_colors, width=0.6)
        
        # Ajouter les labels sur chaque barre
        for i, res in enumerate(recent_results):
            ax5.text(i, 0.5, res, ha='center', va='center', color='white' if res != '?' else 'black', 
                    fontweight='bold', fontsize=14)
        
        # Configurer l'axe des résultats
        ax5.set_ylim(0, 1)
        ax5.set_xlim(-0.5, len(recent_results) - 0.5)
        ax5.set_xticks(range(len(recent_results)))
        ax5.set_xticklabels(['5th', '4th', '3rd', '2nd', 'Last'])
        ax5.set_yticks([])
        ax5.set_title('Recent Form', fontsize=14, pad=10)
        
        # Supprimer les bordures
        for spine in ax5.spines.values():
            spine.set_visible(False)
        
        # Sous-graphique 6: Projection de fin de saison
        ax6 = fig.add_subplot(gs[2, :])
        
        # Calculer les projections
        matches_remaining = 38 - team_data['Played']
        projected_points = team_data['Points'] + (ppg * matches_remaining)
        
        # Créer le graphique à barres empilées
        current = ax6.barh(['Projected Season'], [team_data['Points']], color=self.premier_league_palette['blue'], 
                          label='Current Points')
        remaining = ax6.barh(['Projected Season'], [projected_points - team_data['Points']], left=[team_data['Points']], 
                            color=self.premier_league_palette['light_blue'], alpha=0.7, 
                            label='Projected Additional Points')
        
        # Ajouter les annotations
        ax6.text(team_data['Points'] / 2, 0, f"{team_data['Points']}", ha='center', va='center', 
                color='white', fontweight='bold', fontsize=12)
        
        ax6.text(team_data['Points'] + (projected_points - team_data['Points']) / 2, 0, 
                f"+{int(projected_points - team_data['Points'])}", 
                ha='center', va='center', color='black', fontsize=12)
        
        ax6.text(projected_points + 2, 0, f"Total: {int(projected_points)}", 
                ha='left', va='center', fontsize=12)
        
        # Ajouter les annotations de progression
        ax6.text(0, -0.4, f"Played: {team_data['Played']} / 38 matches ({team_data['Played']/38*100:.1f}%)", 
                ha='left', va='center', fontsize=10)
        
        # Ajouter des lignes verticales pour les zones
        cl_points = 70  # Approximation points pour Champions League
        el_points = 60  # Approximation points pour Europa League
        rel_points = 40  # Approximation points pour éviter la relégation
        
        # Champions League
        ax6.axvline(x=cl_points, color=self.premier_league_palette['light_blue'], linestyle='--', alpha=0.7)
        ax6.text(cl_points, 0.5, "CL", ha='right', va='center', color=self.premier_league_palette['light_blue'], 
                fontweight='bold')
        
        # Europa League
        ax6.axvline(x=el_points, color=self.premier_league_palette['purple'], linestyle='--', alpha=0.7)
        ax6.text(el_points, 0.5, "EL", ha='right', va='center', color=self.premier_league_palette['purple'], 
                fontweight='bold')
        
        # Relégation
        ax6.axvline(x=rel_points, color=self.premier_league_palette['red'], linestyle='--', alpha=0.7)
        ax6.text(rel_points, 0.5, "REL", ha='right', va='center', color=self.premier_league_palette['red'], 
                fontweight='bold')
        
        ax6.set_title('Season Projection', fontsize=14, pad=10)
        ax6.set_xlabel('Points', fontsize=12)
        ax6.grid(True, linestyle='--', alpha=0.7, axis='x')
        ax6.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
        
        # Supprimer les bordures
        for spine in ax6.spines.values():
            spine.set_visible(False)
        
        # Supprimer les yticks
        ax6.set_yticks([])
        
        # Ajouter un titre global
        fig.suptitle(f'{team_name} - Season Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # Ajuster la mise en page
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarder l'image
        if save:
            date_str = datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(self.output_dir, f"{team_name.replace(' ', '_').lower()}_dashboard_{date_str}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Tableau de bord visualisé et sauvegardé: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig
    
    def _generate_league_dashboard(self, save=True, show=True):
        """
        Génère un tableau de bord global de la ligue
        
        Args:
            save (bool): Sauvegarder la visualisation
            show (bool): Afficher la visualisation
            
        Returns:
            matplotlib.figure.Figure: L'objet figure créé
        """
        if self.standings is None:
            print("Aucune donnée de classement disponible.")
            return None
        
        # Créer une figure avec une grille de sous-graphiques
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1.2], wspace=0.4, hspace=0.5)
        
        # Sous-graphique 1: Top 6 équipes
        ax1 = fig.add_subplot(gs[0, :3])
        
        # Sélectionner les 6 premières équipes
        top_teams = self.standings.head(6)
        
        # Créer un graphique à barres horizontales
        bars = ax1.barh(top_teams['Team'], top_teams['Points'], color=self.premier_league_palette['blue'])
        
        # Ajouter les positions à côté du nom de l'équipe
        for i, (_, row) in enumerate(top_teams.iterrows()):
            ax1.text(-2, i, f"{row['Position']}.", ha='right', va='center', 
                    color='black', fontweight='bold', fontsize=10)
        
        # Ajouter les valeurs dans les barres
        for bar, points in zip(bars, top_teams['Points']):
            width = bar.get_width()
            ax1.text(width / 2, bar.get_y() + bar.get_height() / 2, 
                    f"{int(width)}", ha='center', va='center', color='white', fontweight='bold')
        
        ax1.set_title('Top 6 Teams', fontsize=14, pad=10)
        ax1.set_xlim(-5, max(top_teams['Points']) + 5)
        ax1.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # Supprimer les bordures
        for spine in ax1.spines.values():
            spine.set_visible(False)
        
        # Sous-graphique 2: Équipes en zone de relégation
        ax2 = fig.add_subplot(gs[0, 3:])
        
        # Sélectionner les 3 dernières équipes
        bottom_teams = self.standings.tail(3)
        
        # Créer un graphique à barres horizontales
        bars = ax2.barh(bottom_teams['Team'], bottom_teams['Points'], color=self.premier_league_palette['red'])
        
        # Ajouter les positions à côté du nom de l'équipe
        for i, (_, row) in enumerate(bottom_teams.iterrows()):
            ax2.text(-2, i, f"{row['Position']}.", ha='right', va='center', 
                    color='black', fontweight='bold', fontsize=10)
        
        # Ajouter les valeurs dans les barres
        for bar, points in zip(bars, bottom_teams['Points']):
            width = bar.get_width()
            ax2.text(width / 2, bar.get_y() + bar.get_height() / 2, 
                    f"{int(width)}", ha='center', va='center', color='white', fontweight='bold')
        
        ax2.set_title('Relegation Zone', fontsize=14, pad=10)
        ax2.set_xlim(-5, max(bottom_teams['Points']) + 5)
        ax2.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # Supprimer les bordures
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        # Sous-graphique 3: Meilleures attaques
        ax3 = fig.add_subplot(gs[1, :2])
        
        # Sélectionner les 5 équipes avec le plus de buts marqués
        best_attack = self.standings.sort_values('GF', ascending=False).head(5)
        
        # Créer un graphique à barres horizontales
        ax3.barh(best_attack['Team'], best_attack['GF'], color=self.premier_league_palette['light_blue'])
        
        ax3.set_title('Best Attacks', fontsize=14, pad=10)
        ax3.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # Sous-graphique 4: Meilleures défenses
        ax4 = fig.add_subplot(gs[1, 2:4])
        
        # Sélectionner les 5 équipes avec le moins de buts encaissés
        best_defence = self.standings.sort_values('GA').head(5)
        
        # Créer un graphique à barres horizontales
        ax4.barh(best_defence['Team'], best_defence['GA'], color=self.premier_league_palette['purple'])
        
        ax4.set_title('Best Defences (Fewest Goals Conceded)', fontsize=14, pad=10)
        ax4.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # Sous-graphique 5: Meilleures différences de buts
        ax5 = fig.add_subplot(gs[1, 4:])
        
        # Sélectionner les 5 équipes avec la meilleure différence de buts
        best_gd = self.standings.sort_values('GD', ascending=False).head(5)
        
        # Créer un graphique à barres horizontales
        bars = ax5.barh(best_gd['Team'], best_gd['GD'], 
                       color=[self.premier_league_palette['green'] if gd >= 0 else self.premier_league_palette['red'] 
                             for gd in best_gd['GD']])
        
        ax5.set_title('Best Goal Difference', fontsize=14, pad=10)
        ax5.grid(True, linestyle='--', alpha=0.7, axis='x')
        
        # Sous-graphique 6: Visualisation de la distribution des points
        ax6 = fig.add_subplot(gs[2, :])
        
        # Créer un graphique en nuage de points
        points = self.standings['Points']
        positions = self.standings['Position']
        
        # Scatter plot avec taille proportionnelle aux points
        sizes = points * 5  # Ajuster la taille en fonction des points
        
        scatter = ax6.scatter(positions, points, s=sizes, c=points, cmap='viridis', 
                            alpha=0.7, edgecolors='w')
        
        # Ajouter les noms des équipes
        for i, (_, row) in enumerate(self.standings.iterrows()):
            ax6.text(row['Position'], row['Points'] + 1, row['Team'], 
                    ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Ajouter les lignes horizontales pour les zones
        cl_line = 70  # Approximation ligne Champions League
        el_line = 60  # Approximation ligne Europa League
        rel_line = 40  # Approximation ligne relégation
        
        ax6.axhline(y=cl_line, color=self.premier_league_palette['light_blue'], linestyle='--', alpha=0.7)
        ax6.text(1, cl_line + 1, "Typical Champions League threshold", 
                ha='left', va='bottom', color=self.premier_league_palette['light_blue'], fontsize=10)
        
        ax6.axhline(y=el_line, color=self.premier_league_palette['purple'], linestyle='--', alpha=0.7)
        ax6.text(1, el_line + 1, "Typical Europa League threshold", 
                ha='left', va='bottom', color=self.premier_league_palette['purple'], fontsize=10)
        
        ax6.axhline(y=rel_line, color=self.premier_league_palette['red'], linestyle='--', alpha=0.7)
        ax6.text(1, rel_line - 2, "Typical relegation threshold", 
                ha='left', va='top', color=self.premier_league_palette['red'], fontsize=10)
        
        # Ajouter une barre de couleur
        cbar = plt.colorbar(scatter, ax=ax6, pad=0.01)
        cbar.set_label('Points')
        
        # Configurer l'axe
        ax6.set_title('Points Distribution by Position', fontsize=14, pad=10)
        ax6.set_xlabel('Position', fontsize=12)
        ax6.set_ylabel('Points', fontsize=12)
        ax6.set_xlim(0.5, len(self.standings) + 0.5)
        ax6.set_ylim(0, max(points) + 10)
        ax6.invert_xaxis()  # Position 1 à gauche
        ax6.grid(True, linestyle='--', alpha=0.7)
        
        # Ajouter un titre global
        fig.suptitle('Premier League Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # Ajuster la mise en page
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarder l'image
        if save:
            date_str = datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(self.output_dir, f"league_dashboard_{date_str}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Tableau de bord de la ligue visualisé et sauvegardé: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig