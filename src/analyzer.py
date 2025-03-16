import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from tabulate import tabulate

class PremierLeagueAnalyzer:
    """
    Classe pour analyser les données de Premier League récupérées par PremierLeagueScraper
    """
    def __init__(self, data_dir='data'):
        """
        Initialise l'analyseur de données de Premier League
        
        Args:
            data_dir (str): Répertoire contenant les données à analyser
        """
        self.data_dir = data_dir
        self.standings = None
        self.fixtures = None
        self.team_stats = {}
        
        # Vérifier si le répertoire existe
        if not os.path.exists(data_dir):
            print(f"Le répertoire {data_dir} n'existe pas. Veuillez exécuter le scraper d'abord.")
            os.makedirs(data_dir)
    
    def load_latest_data(self):
        """
        Charge les données les plus récentes dans l'analyseur
        
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
    
    def get_standings(self):
        """
        Affiche le classement actuel de la Premier League avec un formatage amélioré
        
        Returns:
            pandas.DataFrame: DataFrame contenant le classement
        """
        if self.standings is None:
            print("Aucune donnée de classement disponible.")
            return None
        
        # Créer une copie pour ne pas modifier l'original
        standings_display = self.standings.copy()
        
        # Ajouter une colonne pour le ratio de points par match
        standings_display['PPM'] = round(standings_display['Points'] / standings_display['Played'], 2)
        
        # Réorganiser les colonnes
        columns = ['Position', 'Team', 'Played', 'Won', 'Drawn', 'Lost', 'GF', 'GA', 'GD', 'Points', 'PPM']
        standings_display = standings_display[columns]
        
        # Utiliser tabulate pour un affichage formaté
        print(tabulate(standings_display, headers='keys', tablefmt='pretty', showindex=False))
        
        return standings_display
    
    def team_performance(self, team_name):
        """
        Analyse détaillée des performances d'une équipe
        
        Args:
            team_name (str): Nom de l'équipe à analyser
            
        Returns:
            dict: Dictionnaire contenant diverses métriques de performance
        """
        if self.standings is None or self.fixtures is None:
            print("Données insuffisantes pour analyser les performances.")
            return None
        
        # Vérifier si l'équipe existe dans nos données
        if team_name not in self.standings['Team'].values:
            print(f"Équipe non trouvée: {team_name}")
            return None
        
        # Extraire les données de l'équipe du classement
        team_data = self.standings[self.standings['Team'] == team_name].iloc[0]
        
        # Filtrer les matchs de l'équipe
        team_matches = self.fixtures[
            (self.fixtures['HomeTeam'] == team_name) | 
            (self.fixtures['AwayTeam'] == team_name)
        ].copy()
        
        # Convertir les scores en numériques (s'ils ne le sont pas déjà)
        for col in ['HomeScore', 'AwayScore']:
            if team_matches[col].dtype == 'object':
                team_matches[col] = pd.to_numeric(team_matches[col], errors='coerce')
        
        # Calculer les résultats des matchs joués
        played_matches = team_matches.dropna(subset=['HomeScore', 'AwayScore'])
        
        # Initialiser les compteurs
        home_wins = away_wins = home_losses = away_losses = home_draws = away_draws = 0
        home_goals_scored = away_goals_scored = home_goals_conceded = away_goals_conceded = 0
        
        # Analyser chaque match joué
        for _, match in played_matches.iterrows():
            if match['HomeTeam'] == team_name:
                # L'équipe joue à domicile
                if match['HomeScore'] > match['AwayScore']:
                    home_wins += 1
                elif match['HomeScore'] < match['AwayScore']:
                    home_losses += 1
                else:
                    home_draws += 1
                
                home_goals_scored += match['HomeScore']
                home_goals_conceded += match['AwayScore']
            else:
                # L'équipe joue à l'extérieur
                if match['AwayScore'] > match['HomeScore']:
                    away_wins += 1
                elif match['AwayScore'] < match['HomeScore']:
                    away_losses += 1
                else:
                    away_draws += 1
                
                away_goals_scored += match['AwayScore']
                away_goals_conceded += match['HomeScore']
        
        # Calculer les points à domicile et à l'extérieur
        home_points = home_wins * 3 + home_draws
        away_points = away_wins * 3 + away_draws
        
        # Calculer les totaux
        total_home_matches = home_wins + home_draws + home_losses
        total_away_matches = away_wins + away_draws + away_losses
        
        # Résultats à venir
        upcoming_matches = team_matches[team_matches['Status'] != 'Joué']
        
        # Compiler toutes les métriques
        performance = {
            'team': team_name,
            'position': team_data['Position'],
            'points': team_data['Points'],
            'matches': {
                'played': team_data['Played'],
                'won': team_data['Won'],
                'drawn': team_data['Drawn'],
                'lost': team_data['Lost']
            },
            'goals': {
                'scored': team_data['GF'],
                'conceded': team_data['GA'],
                'difference': team_data['GD']
            },
            'home': {
                'played': total_home_matches,
                'won': home_wins,
                'drawn': home_draws,
                'lost': home_losses,
                'points': home_points,
                'goals_scored': home_goals_scored,
                'goals_conceded': home_goals_conceded,
                'points_per_game': round(home_points / total_home_matches, 2) if total_home_matches > 0 else 0
            },
            'away': {
                'played': total_away_matches,
                'won': away_wins,
                'drawn': away_draws,
                'lost': away_losses,
                'points': away_points,
                'goals_scored': away_goals_scored,
                'goals_conceded': away_goals_conceded,
                'points_per_game': round(away_points / total_away_matches, 2) if total_away_matches > 0 else 0
            },
            'upcoming_matches': upcoming_matches.to_dict('records')
        }
        
        # Ajouter les statistiques de l'équipe si disponibles
        if team_name.lower() in self.team_stats:
            performance['detailed_stats'] = self.team_stats[team_name.lower()]
        
        return performance
    
    def print_team_performance(self, team_name):
        """
        Affiche un rapport détaillé sur les performances d'une équipe
        
        Args:
            team_name (str): Nom de l'équipe à analyser
        """
        performance = self.team_performance(team_name)
        
        if not performance:
            return
        
        print(f"\n{'=' * 50}")
        print(f"RAPPORT DE PERFORMANCE: {performance['team'].upper()}")
        print(f"{'=' * 50}")
        
        print(f"\nPosition actuelle: {performance['position']}")
        print(f"Points totaux: {performance['points']}")
        
        print("\nRÉSULTATS GLOBAUX:")
        print(f"Matchs joués: {performance['matches']['played']}")
        print(f"Victoires: {performance['matches']['won']} ({round(performance['matches']['won'] / performance['matches']['played'] * 100, 1)}%)")
        print(f"Nuls: {performance['matches']['drawn']} ({round(performance['matches']['drawn'] / performance['matches']['played'] * 100, 1)}%)")
        print(f"Défaites: {performance['matches']['lost']} ({round(performance['matches']['lost'] / performance['matches']['played'] * 100, 1)}%)")
        
        print("\nBUTS:")
        print(f"Marqués: {performance['goals']['scored']} (moyenne: {round(performance['goals']['scored'] / performance['matches']['played'], 2)} par match)")
        print(f"Encaissés: {performance['goals']['conceded']} (moyenne: {round(performance['goals']['conceded'] / performance['matches']['played'], 2)} par match)")
        print(f"Différence: {performance['goals']['difference']}")
        
        print("\nPERFORMANCE À DOMICILE:")
        home = performance['home']
        if home['played'] > 0:
            print(f"Matchs joués: {home['played']}")
            print(f"Résultats: {home['won']}V {home['drawn']}N {home['lost']}D")
            print(f"Points: {home['points']} ({home['points_per_game']} par match)")
            print(f"Buts marqués: {home['goals_scored']} (moyenne: {round(home['goals_scored'] / home['played'], 2)} par match)")
            print(f"Buts encaissés: {home['goals_conceded']} (moyenne: {round(home['goals_conceded'] / home['played'], 2)} par match)")
        else:
            print("Aucun match à domicile joué.")
        
        print("\nPERFORMANCE À L'EXTÉRIEUR:")
        away = performance['away']
        if away['played'] > 0:
            print(f"Matchs joués: {away['played']}")
            print(f"Résultats: {away['won']}V {away['drawn']}N {away['lost']}D")
            print(f"Points: {away['points']} ({away['points_per_game']} par match)")
            print(f"Buts marqués: {away['goals_scored']} (moyenne: {round(away['goals_scored'] / away['played'], 2)} par match)")
            print(f"Buts encaissés: {away['goals_conceded']} (moyenne: {round(away['goals_conceded'] / away['played'], 2)} par match)")
        else:
            print("Aucun match à l'extérieur joué.")
        
        print("\nPROCHAINS MATCHS:")
        if performance['upcoming_matches']:
            for i, match in enumerate(performance['upcoming_matches'], 1):
                if i > 5:  # Limiter à 5 prochains matchs
                    break
                print(f"{i}. {match['HomeTeam']} vs {match['AwayTeam']} ({match['Date']})")
        else:
            print("Aucun match à venir.")
    
    def plot_team_form(self, team_name):
        """
        Génère un graphique montrant la forme récente d'une équipe
        
        Args:
            team_name (str): Nom de l'équipe à analyser
            
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
        
        # Déterminer le résultat de chaque match
        results = []
        points = []
        cumulative_points = 0
        goal_difference = []
        
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                # L'équipe joue à domicile
                if match['HomeScore'] > match['AwayScore']:
                    results.append('W')
                    cumulative_points += 3
                    points.append(3)
                elif match['HomeScore'] < match['AwayScore']:
                    results.append('L')
                    points.append(0)
                else:
                    results.append('D')
                    cumulative_points += 1
                    points.append(1)
                
                goal_difference.append(match['HomeScore'] - match['AwayScore'])
            else:
                # L'équipe joue à l'extérieur
                if match['AwayScore'] > match['HomeScore']:
                    results.append('W')
                    cumulative_points += 3
                    points.append(3)
                elif match['AwayScore'] < match['HomeScore']:
                    results.append('L')
                    points.append(0)
                else:
                    results.append('D')
                    cumulative_points += 1
                    points.append(1)
                
                goal_difference.append(match['AwayScore'] - match['HomeScore'])
        
        # Créer un DataFrame pour les résultats
        form_data = pd.DataFrame({
            'Match': range(1, len(results) + 1),
            'Result': results,
            'Points': points,
            'CumulativePoints': np.cumsum(points),
            'GoalDifference': goal_difference
        })
        
        # Configurer la figure
        plt.figure(figsize=(12, 8))
        
        # Sous-figure 1: Résultats récents
        plt.subplot(2, 1, 1)
        colors = ['green' if r == 'W' else 'red' if r == 'L' else 'gray' for r in results[-5:]]
        plt.bar(range(len(results[-5:])), [1] * len(results[-5:]), color=colors)
        plt.xticks(range(len(results[-5:])), results[-5:])
        plt.title(f"5 derniers résultats de {team_name}")
        plt.ylim(0, 1.2)
        plt.axis('off')
        
        # Sous-figure 2: Points cumulés
        plt.subplot(2, 1, 2)
        plt.plot(form_data['Match'], form_data['CumulativePoints'], marker='o', linewidth=2, color='blue')
        plt.title(f"Progression des points sur la saison: {team_name}")
        plt.xlabel("Match joué")
        plt.ylabel("Points cumulés")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ajuster la disposition
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/{team_name.replace(' ', '_').lower()}_form.png")
        
        return plt.gcf()

if __name__ == "__main__":
    # Exemple d'utilisation
    analyzer = PremierLeagueAnalyzer()
    
    # Charger les données
    if analyzer.load_latest_data():
        print("\nDonnées chargées avec succès!")
        
        # Afficher le classement
        print("\n--- Classement actuel ---")
        analyzer.get_standings()
        
        # Analyser une équipe
        team_name = "Liverpool"  # Remplacer par une équipe disponible dans vos données
        print(f"\n--- Analyse de {team_name} ---")
        analyzer.print_team_performance(team_name)
        
        # Comparer deux équipes
        team2_name = "Manchester City"  # Remplacer par une équipe disponible dans vos données
        print(f"\n--- Comparaison: {team_name} vs {team2_name} ---")
        analyzer.compare_teams(team_name, team2_name)
        
        # Générer un rapport de ligue
        print("\n--- Rapport de la Premier League ---")
        analyzer.generate_league_report()
        
        # Identifier les tendances
        print("\n--- Tendances et anomalies ---")
        analyzer.identify_trends()
        
        # Prédire un match
        print(f"\n--- Prédiction: {team_name} vs {team2_name} ---")
        analyzer.predict_match(team_name, team2_name)
        
        # Aperçu de la prochaine journée
        print("\n--- Aperçu de la prochaine journée ---")
        analyzer.generate_match_day_preview()
        
        # Projections de fin de saison
        print("\n--- Projections de fin de saison ---")
        analyzer.generate_season_projections()
        
        # Créer des visualisations
        print("\n--- Création de visualisations ---")
        analyzer.plot_team_form(team_name)
        analyzer.plot_team_comparison(team_name, team2_name)
        analyzer.plot_league_table()
        
        print("\nAnalyse terminée! Vous pouvez trouver les visualisations dans le répertoire de données.")
    else:
        print("\nÉchec du chargement des données. Veuillez d'abord exécuter le scraper.")

    
    def compare_teams(self, team1, team2):
        """
        Compare les performances de deux équipes
        
        Args:
            team1 (str): Nom de la première équipe
            team2 (str): Nom de la deuxième équipe
            
        Returns:
            dict: Dictionnaire contenant la comparaison
        """
        if self.standings is None:
            print("Aucune donnée de classement disponible.")
            return None
        
        # Vérifier si les équipes existent
        if team1 not in self.standings['Team'].values or team2 not in self.standings['Team'].values:
            print(f"Une ou plusieurs équipes non trouvées: {team1}, {team2}")
            return None
        
        # Obtenir les performances des deux équipes
        team1_perf = self.team_performance(team1)
        team2_perf = self.team_performance(team2)
        
        if not team1_perf or not team2_perf:
            return None
        
        # Trouver les confrontations directes
        if self.fixtures is not None:
            head_to_head = self.fixtures[
                ((self.fixtures['HomeTeam'] == team1) & (self.fixtures['AwayTeam'] == team2)) |
                ((self.fixtures['HomeTeam'] == team2) & (self.fixtures['AwayTeam'] == team1))
            ].copy()
            
            # Convertir les scores en numériques
            for col in ['HomeScore', 'AwayScore']:
                if head_to_head[col].dtype == 'object':
                    head_to_head[col] = pd.to_numeric(head_to_head[col], errors='coerce')
        else:
            head_to_head = pd.DataFrame()
        
        # Compiler la comparaison
        comparison = {
            'teams': [team1, team2],
            'positions': [team1_perf['position'], team2_perf['position']],
            'points': [team1_perf['points'], team2_perf['points']],
            'matches_played': [team1_perf['matches']['played'], team2_perf['matches']['played']],
            'wins': [team1_perf['matches']['won'], team2_perf['matches']['won']],
            'draws': [team1_perf['matches']['drawn'], team2_perf['matches']['drawn']],
            'losses': [team1_perf['matches']['lost'], team2_perf['matches']['lost']],
            'goals_scored': [team1_perf['goals']['scored'], team2_perf['goals']['scored']],
            'goals_conceded': [team1_perf['goals']['conceded'], team2_perf['goals']['conceded']],
            'goal_difference': [team1_perf['goals']['difference'], team2_perf['goals']['difference']],
            'head_to_head': head_to_head.to_dict('records')
        }
        
        # Calculer les résultats des confrontations directes
        team1_h2h_wins = team2_h2h_wins = h2h_draws = 0
        
        for _, match in head_to_head.dropna(subset=['HomeScore', 'AwayScore']).iterrows():
            if match['HomeTeam'] == team1:
                if match['HomeScore'] > match['AwayScore']:
                    team1_h2h_wins += 1
                elif match['HomeScore'] < match['AwayScore']:
                    team2_h2h_wins += 1
                else:
                    h2h_draws += 1
            else:
                if match['HomeScore'] > match['AwayScore']:
                    team2_h2h_wins += 1
                elif match['HomeScore'] < match['AwayScore']:
                    team1_h2h_wins += 1
                else:
                    h2h_draws += 1
        
        comparison['head_to_head_summary'] = {
            'matches': len(head_to_head.dropna(subset=['HomeScore', 'AwayScore'])),
            'wins': [team1_h2h_wins, team2_h2h_wins],
            'draws': h2h_draws
        }
        
        # Afficher la comparaison
        print(f"\n{'=' * 50}")
        print(f"COMPARAISON: {team1.upper()} vs {team2.upper()}")
        print(f"{'=' * 50}")
        
        print(f"\nPosition au classement: {team1}: {team1_perf['position']} | {team2}: {team2_perf['position']}")
        print(f"Points: {team1}: {team1_perf['points']} | {team2}: {team2_perf['points']}")
        print(f"Matchs joués: {team1}: {team1_perf['matches']['played']} | {team2}: {team2_perf['matches']['played']}")
        
        print("\nRÉSULTATS:")
        print(f"Victoires: {team1}: {team1_perf['matches']['won']} | {team2}: {team2_perf['matches']['won']}")
        print(f"Nuls: {team1}: {team1_perf['matches']['drawn']} | {team2}: {team2_perf['matches']['drawn']}")
        print(f"Défaites: {team1}: {team1_perf['matches']['lost']} | {team2}: {team2_perf['matches']['lost']}")
        
        print("\nBUTS:")
        print(f"Marqués: {team1}: {team1_perf['goals']['scored']} | {team2}: {team2_perf['goals']['scored']}")
        print(f"Encaissés: {team1}: {team1_perf['goals']['conceded']} | {team2}: {team2_perf['goals']['conceded']}")
        print(f"Différence: {team1}: {team1_perf['goals']['difference']} | {team2}: {team2_perf['goals']['difference']}")
        
        print("\nCONFRONTATIONS DIRECTES:")
        if comparison['head_to_head_summary']['matches'] > 0:
            print(f"Total des matchs: {comparison['head_to_head_summary']['matches']}")
            print(f"Victoires: {team1}: {team1_h2h_wins} | {team2}: {team2_h2h_wins}")
            print(f"Nuls: {h2h_draws}")
        else:
            print("Aucune confrontation directe cette saison.")
        
        return comparison
    
    def plot_team_comparison(self, team1, team2):
        """
        Génère un graphique radar comparant deux équipes
        
        Args:
            team1 (str): Nom de la première équipe
            team2 (str): Nom de la deuxième équipe
            
        Returns:
            matplotlib.figure.Figure: L'objet figure créé
        """
        comparison = self.compare_teams(team1, team2)
        
        if not comparison:
            return None
        
        # Définir les catégories pour le graphique radar
        categories = ['Points', 'Victoires', 'Nuls', 'Diff. buts', 'Buts marqués', 'Clean sheets']
        
        # Compter les clean sheets
        team1_perf = self.team_performance(team1)
        team2_perf = self.team_performance(team2)
        
        team1_clean_sheets = 0
        team2_clean_sheets = 0
        
        if self.fixtures is not None:
            team1_matches = self.fixtures[
                ((self.fixtures['HomeTeam'] == team1) | (self.fixtures['AwayTeam'] == team1)) &
                (self.fixtures['Status'] == 'Joué')
            ].copy()
            
            team2_matches = self.fixtures[
                ((self.fixtures['HomeTeam'] == team2) | (self.fixtures['AwayTeam'] == team2)) &
                (self.fixtures['Status'] == 'Joué')
            ].copy()
            
            # Convertir les scores en numériques
            for col in ['HomeScore', 'AwayScore']:
                team1_matches[col] = pd.to_numeric(team1_matches[col], errors='coerce')
                team2_matches[col] = pd.to_numeric(team2_matches[col], errors='coerce')
            
            # Compter les clean sheets pour team1
            for _, match in team1_matches.dropna(subset=['HomeScore', 'AwayScore']).iterrows():
                if match['HomeTeam'] == team1 and match['AwayScore'] == 0:
                    team1_clean_sheets += 1
                elif match['AwayTeam'] == team1 and match['HomeScore'] == 0:
                    team1_clean_sheets += 1
            
            # Compter les clean sheets pour team2
            for _, match in team2_matches.dropna(subset=['HomeScore', 'AwayScore']).iterrows():
                if match['HomeTeam'] == team2 and match['AwayScore'] == 0:
                    team2_clean_sheets += 1
                elif match['AwayTeam'] == team2 and match['HomeScore'] == 0:
                    team2_clean_sheets += 1
        
        # Normaliser les valeurs
        max_points = max(comparison['points'])
        max_wins = max(comparison['wins'])
        max_draws = max(comparison['draws'])
        max_gd = max(abs(comparison['goal_difference'][0]), abs(comparison['goal_difference'][1]))
        max_goals = max(comparison['goals_scored'])
        max_cs = max(team1_clean_sheets, team2_clean_sheets)
        
        team1_values = [
            comparison['points'][0] / max_points if max_points > 0 else 0,
            comparison['wins'][0] / max_wins if max_wins > 0 else 0,
            comparison['draws'][0] / max_draws if max_draws > 0 else 0,
            (comparison['goal_difference'][0] + max_gd) / (2 * max_gd) if max_gd > 0 else 0.5,
            comparison['goals_scored'][0] / max_goals if max_goals > 0 else 0,
            team1_clean_sheets / max_cs if max_cs > 0 else 0
        ]
        
        team2_values = [
            comparison['points'][1] / max_points if max_points > 0 else 0,
            comparison['wins'][1] / max_wins if max_wins > 0 else 0,
            comparison['draws'][1] / max_draws if max_draws > 0 else 0,
            (comparison['goal_difference'][1] + max_gd) / (2 * max_gd) if max_gd > 0 else 0.5,
            comparison['goals_scored'][1] / max_goals if max_goals > 0 else 0,
            team2_clean_sheets / max_cs if max_cs > 0 else 0
        ]
        
        # Créer le graphique radar
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Fermer le graphique
        
        team1_values += team1_values[:1]
        team2_values += team2_values[:1]
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        ax.plot(angles, team1_values, 'o-', linewidth=2, label=team1)
        ax.fill(angles, team1_values, alpha=0.25)
        
        ax.plot(angles, team2_values, 'o-', linewidth=2, label=team2)
        ax.fill(angles, team2_values, alpha=0.25)
        
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        ax.set_ylim(0, 1.1)
        ax.grid(True)
        
        ax.set_title(f"Comparaison: {team1} vs {team2}")
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/{team1.replace(' ', '_').lower()}_vs_{team2.replace(' ', '_').lower()}.png")
        
        return fig
    
    def predict_match(self, home_team, away_team):
        """
        Prédit le résultat d'un match en fonction des performances passées des équipes
        
        Args:
            home_team (str): Nom de l'équipe jouant à domicile
            away_team (str): Nom de l'équipe jouant à l'extérieur
            
        Returns:
            dict: Dictionnaire contenant la prédiction
        """
        if self.standings is None or self.fixtures is None:
            print("Données insuffisantes pour faire une prédiction.")
            return None
        
        # Vérifier si les équipes existent
        if home_team not in self.standings['Team'].values or away_team not in self.standings['Team'].values:
            print(f"Une ou plusieurs équipes non trouvées: {home_team}, {away_team}")
            return None
        
        # Obtenir les performances des deux équipes
        home_perf = self.team_performance(home_team)
        away_perf = self.team_performance(away_team)
        
        if not home_perf or not away_perf:
            return None
        
        # Facteurs à prendre en compte pour la prédiction
        
        # 1. Avantage du terrain (moyenne de points à domicile vs moyenne de points à l'extérieur)
        home_advantage = home_perf['home']['points_per_game'] if home_perf['home']['played'] > 0 else 0
        away_disadvantage = 3 - away_perf['away']['points_per_game'] if away_perf['away']['played'] > 0 else 1.5
        
        # 2. Position au classement (plus c'est bas, plus la valeur est haute)
        position_factor = (away_perf['position'] - home_perf['position']) * 0.05
        
        # 3. Forme récente (5 derniers matchs)
        home_recent_form = self._calculate_recent_form(home_team)
        away_recent_form = self._calculate_recent_form(away_team)
        
        # 4. Confrontations directes
        h2h_factor = self._calculate_h2h_factor(home_team, away_team)
        
        # 5. Efficacité offensive et défensive
        home_attack = home_perf['goals']['scored'] / home_perf['matches']['played'] if home_perf['matches']['played'] > 0 else 1
        home_defense = home_perf['goals']['conceded'] / home_perf['matches']['played'] if home_perf['matches']['played'] > 0 else 1
        away_attack = away_perf['goals']['scored'] / away_perf['matches']['played'] if away_perf['matches']['played'] > 0 else 1
        away_defense = away_perf['goals']['conceded'] / away_perf['matches']['played'] if away_perf['matches']['played'] > 0 else 1
        
        # Calculer les scores estimés (formule simple basée sur les facteurs ci-dessus)
        estimated_home_goals = ((home_attack * away_defense) / 1.5) * (1 + home_advantage * 0.1 + position_factor + home_recent_form * 0.2 + h2h_factor * 0.1)
        estimated_away_goals = ((away_attack * home_defense) / 1.8) * (1 + (1/away_disadvantage) * 0.1 - position_factor + away_recent_form * 0.2 - h2h_factor * 0.1)
        
        # Calculer les probabilités
        home_win_prob = 0.4 + home_advantage * 0.1 + position_factor * 0.5 + (home_recent_form - away_recent_form) * 0.1 + h2h_factor * 0.1
        draw_prob = 0.3 - abs(position_factor) * 0.3 - abs(home_recent_form - away_recent_form) * 0.05
        away_win_prob = 1 - home_win_prob - draw_prob
        
        # Ajuster les probabilités pour qu'elles soient entre 0 et 1
        home_win_prob = max(0.05, min(0.9, home_win_prob))
        away_win_prob = max(0.05, min(0.9, away_win_prob))
        draw_prob = 1 - home_win_prob - away_win_prob
        
        # Arrondir les estimations de buts
        home_goals_rounded = round(max(0, estimated_home_goals), 1)
        away_goals_rounded = round(max(0, estimated_away_goals), 1)
        
        # Déterminer le score le plus probable
        probable_home_goals = round(home_goals_rounded)
        probable_away_goals = round(away_goals_rounded)
        
        # Déterminer le résultat le plus probable
        if home_win_prob > draw_prob and home_win_prob > away_win_prob:
            likely_result = f"Victoire {home_team}"
            if probable_home_goals == probable_away_goals:
                probable_home_goals += 1
        elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
            likely_result = f"Victoire {away_team}"
            if probable_home_goals == probable_away_goals:
                probable_away_goals += 1
        else:
            likely_result = "Match nul"
            probable_home_goals = probable_away_goals
        
        # Compiler la prédiction
        prediction = {
            'match': f"{home_team} vs {away_team}",
            'estimated_score': f"{home_goals_rounded:.1f} - {away_goals_rounded:.1f}",
            'probable_score': f"{probable_home_goals} - {probable_away_goals}",
            'probabilities': {
                'home_win': round(home_win_prob * 100, 1),
                'draw': round(draw_prob * 100, 1),
                'away_win': round(away_win_prob * 100, 1)
            },
            'likely_result': likely_result,
            'factors': {
                'home_advantage': home_advantage,
                'position_difference': position_factor,
                'home_recent_form': home_recent_form,
                'away_recent_form': away_recent_form,
                'head_to_head': h2h_factor
            }
        }
        
        # Afficher la prédiction
        print(f"\n{'=' * 50}")
        print(f"PRÉDICTION DE MATCH: {home_team} vs {away_team}")
        print(f"{'=' * 50}")
        
        print(f"\nScore estimé: {home_goals_rounded:.1f} - {away_goals_rounded:.1f}")
        print(f"Score probable: {probable_home_goals} - {probable_away_goals}")
        print(f"Résultat probable: {likely_result}")
        
        print("\nPROBABILITÉS:")
        print(f"Victoire {home_team}: {prediction['probabilities']['home_win']}%")
        print(f"Match nul: {prediction['probabilities']['draw']}%")
        print(f"Victoire {away_team}: {prediction['probabilities']['away_win']}%")
        
        return prediction
        
    def _calculate_recent_form(self, team_name, num_matches=5):
        """
        Calcule un score de forme récente basé sur les derniers matchs
        
        Args:
            team_name (str): Nom de l'équipe
            num_matches (int): Nombre de matchs à considérer
            
        Returns:
            float: Score de forme entre -1 et 1
        """
        if self.fixtures is None:
            return 0
        
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
            return 0
        
        # Trier par date et prendre les N derniers matchs
        try:
            team_matches['Date'] = pd.to_datetime(team_matches['Date'])
            team_matches = team_matches.sort_values('Date', ascending=False)
        except:
            # Si la conversion de date échoue, on garde l'ordre actuel
            pass
        
        recent_matches = team_matches.head(num_matches)
        
        if len(recent_matches) == 0:
            return 0
        
        # Calculer le score de forme
        form_score = 0
        weights = [1.0, 0.8, 0.6, 0.4, 0.2]  # Plus de poids aux matchs récents
        
        for i, (_, match) in enumerate(recent_matches.iterrows()):
            weight = weights[i] if i < len(weights) else 0.2
            
            if match['HomeTeam'] == team_name:
                if match['HomeScore'] > match['AwayScore']:
                    form_score += 1 * weight
                elif match['HomeScore'] < match['AwayScore']:
                    form_score -= 1 * weight
                # Match nul: pas de changement
            else:
                if match['AwayScore'] > match['HomeScore']:
                    form_score += 1 * weight
                elif match['AwayScore'] < match['HomeScore']:
                    form_score -= 1 * weight
                # Match nul: pas de changement
        
        # Normaliser entre -1 et 1
        matches_considered = min(len(recent_matches), len(weights))
        normalized_form = form_score / sum(weights[:matches_considered])
        
        return normalized_form
    
    def _calculate_h2h_factor(self, home_team, away_team):
        """
        Calcule un facteur basé sur les confrontations directes
        
        Args:
            home_team (str): Nom de l'équipe à domicile
            away_team (str): Nom de l'équipe à l'extérieur
            
        Returns:
            float: Facteur h2h entre -0.5 et 0.5
        """
        if self.fixtures is None:
            return 0
        
        # Trouver les confrontations directes
        h2h_matches = self.fixtures[
            ((self.fixtures['HomeTeam'] == home_team) & (self.fixtures['AwayTeam'] == away_team)) |
            ((self.fixtures['HomeTeam'] == away_team) & (self.fixtures['AwayTeam'] == home_team))
        ].copy()
        
        # Convertir les scores en numériques
        for col in ['HomeScore', 'AwayScore']:
            h2h_matches[col] = pd.to_numeric(h2h_matches[col], errors='coerce')
        
        # Éliminer les matchs sans score
        h2h_matches = h2h_matches.dropna(subset=['HomeScore', 'AwayScore'])
        
        if len(h2h_matches) == 0:
            return 0
        
        # Calculer le facteur h2h
        home_wins = 0
        away_wins = 0
        
        for _, match in h2h_matches.iterrows():
            if match['HomeTeam'] == home_team:
                if match['HomeScore'] > match['AwayScore']:
                    home_wins += 1
                elif match['HomeScore'] < match['AwayScore']:
                    away_wins += 1
            else:
                if match['HomeScore'] > match['AwayScore']:
                    away_wins += 1
                elif match['HomeScore'] < match['AwayScore']:
                    home_wins += 1
        
        # Calculer le biais h2h
        total_decisive_matches = home_wins + away_wins
        
        if total_decisive_matches == 0:
            return 0
        
        h2h_bias = (home_wins - away_wins) / total_decisive_matches
        
        # Réduire l'impact (entre -0.5 et 0.5)
        return h2h_bias * 0.5
        
    def generate_league_report(self):
        """
        Génère un rapport complet sur l'état actuel de la Premier League
        
        Returns:
            dict: Dictionnaire contenant diverses statistiques de la ligue
        """
        if self.standings is None:
            print("Aucune donnée de classement disponible.")
            return None
        
        # Statistiques globales
        total_matches_played = self.standings['Played'].sum() / 2  # Diviser par 2 car chaque match est compté deux fois
        total_goals_scored = self.standings['GF'].sum()
        avg_goals_per_match = total_goals_scored / total_matches_played if total_matches_played > 0 else 0
        
        # Équipes avec les meilleures attaques/défenses
        best_attack = self.standings.loc[self.standings['GF'].idxmax()]
        best_defense = self.standings.loc[self.standings['GA'].idxmin()]
        
        # Calculer les équipes en forme
        form_scores = []
        for team in self.standings['Team']:
            form_scores.append({
                'team': team,
                'form_score': self._calculate_recent_form(team)
            })
        
        form_df = pd.DataFrame(form_scores)
        in_form_teams = form_df.sort_values('form_score', ascending=False).head(3)
        out_of_form_teams = form_df.sort_values('form_score').head(3)
        
        # Compiler le rapport
        report = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'general_stats': {
                'teams': len(self.standings),
                'matches_played': int(total_matches_played),
                'goals_scored': int(total_goals_scored),
                'avg_goals_per_match': round(avg_goals_per_match, 2)
            },
            'top_teams': {
                'leader': self.standings.iloc[0]['Team'],
                'best_attack': {
                    'team': best_attack['Team'],
                    'goals': int(best_attack['GF'])
                },
                'best_defense': {
                    'team': best_defense['Team'],
                    'goals_conceded': int(best_defense['GA'])
                }
            },
            'form_guide': {
                'in_form': in_form_teams['team'].tolist(),
                'out_of_form': out_of_form_teams['team'].tolist()
            }
        }
        
        # Ajouter des statistiques sur les matchs si disponibles
        if self.fixtures is not None:
            played_matches = self.fixtures[self.fixtures['Status'] == 'Joué'].copy()
            
            # Convertir les scores en numériques
            for col in ['HomeScore', 'AwayScore']:
                played_matches[col] = pd.to_numeric(played_matches[col], errors='coerce')
            
            played_matches = played_matches.dropna(subset=['HomeScore', 'AwayScore'])
            
            if len(played_matches) > 0:
                # Statistiques des matchs
                home_wins = len(played_matches[played_matches['HomeScore'] > played_matches['AwayScore']])
                away_wins = len(played_matches[played_matches['HomeScore'] < played_matches['AwayScore']])
                draws = len(played_matches[played_matches['HomeScore'] == played_matches['AwayScore']])
                
                home_win_pct = home_wins / len(played_matches) * 100 if len(played_matches) > 0 else 0
                
                report['match_stats'] = {
                    'home_wins': home_wins,
                    'away_wins': away_wins,
                    'draws': draws,
                    'home_win_percentage': round(home_win_pct, 1)
                }
                
                # Trouver le plus grand score
                played_matches['TotalGoals'] = played_matches['HomeScore'] + played_matches['AwayScore']
                highest_scoring = played_matches.loc[played_matches['TotalGoals'].idxmax()]
                
                report['match_stats']['highest_scoring_match'] = {
                    'teams': f"{highest_scoring['HomeTeam']} vs {highest_scoring['AwayTeam']}",
                    'score': f"{int(highest_scoring['HomeScore'])} - {int(highest_scoring['AwayScore'])}",
                    'total_goals': int(highest_scoring['TotalGoals'])
                }
        
        # Afficher le rapport
        print(f"\n{'=' * 50}")
        print(f"RAPPORT DE LA PREMIER LEAGUE - {report['date']}")
        print(f"{'=' * 50}")
        
        print("\nSTATISTIQUES GÉNÉRALES:")
        print(f"Nombre d'équipes: {report['general_stats']['teams']}")
        print(f"Matchs joués: {report['general_stats']['matches_played']}")
        print(f"Buts marqués: {report['general_stats']['goals_scored']}")
        print(f"Moyenne de buts par match: {report['general_stats']['avg_goals_per_match']}")
        
        print("\nÉQUIPES PERFORMANTES:")
        print(f"Leader actuel: {report['top_teams']['leader']}")
        print(f"Meilleure attaque: {report['top_teams']['best_attack']['team']} ({report['top_teams']['best_attack']['goals']} buts)")
        print(f"Meilleure défense: {report['top_teams']['best_defense']['team']} ({report['top_teams']['best_defense']['goals_conceded']} buts encaissés)")
        
        print("\nGUIDE DE FORME:")
        print(f"Équipes en forme: {', '.join(report['form_guide']['in_form'])}")
        print(f"Équipes en difficulté: {', '.join(report['form_guide']['out_of_form'])}")
        
        if 'match_stats' in report:
            print("\nSTATISTIQUES DE MATCHS:")
            print(f"Victoires à domicile: {report['match_stats']['home_wins']} ({report['match_stats']['home_win_percentage']}%)")
            print(f"Victoires à l'extérieur: {report['match_stats']['away_wins']}")
            print(f"Matchs nuls: {report['match_stats']['draws']}")
            print(f"Match avec le plus de buts: {report['match_stats']['highest_scoring_match']['teams']} ({report['match_stats']['highest_scoring_match']['score']})")
        
        return report
    
    def identify_trends(self):
        """
        Identifie les tendances et les anomalies dans les données de la Premier League
        
        Returns:
            dict: Dictionnaire contenant les tendances identifiées
        """
        if self.standings is None or self.fixtures is None:
            print("Données insuffisantes pour identifier les tendances.")
            return None
        
        trends = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'team_trends': [],
            'league_trends': [],
            'anomalies': []
        }
        
        # Convertir les scores en numériques
        played_matches = self.fixtures[self.fixtures['Status'] == 'Joué'].copy()
        for col in ['HomeScore', 'AwayScore']:
            played_matches[col] = pd.to_numeric(played_matches[col], errors='coerce')
        
        played_matches = played_matches.dropna(subset=['HomeScore', 'AwayScore'])
        
        # Tendances d'équipes
        for _, team_row in self.standings.iterrows():
            team = team_row['Team']
            team_perf = self.team_performance(team)
            
            if not team_perf:
                continue
            
            # Vérifier les séries d'invincibilité ou de défaites
            team_form = []
            team_matches = played_matches[
                (played_matches['HomeTeam'] == team) | (played_matches['AwayTeam'] == team)
            ].copy()
            
            # Trier par date si possible
            try:
                team_matches['Date'] = pd.to_datetime(team_matches['Date'])
                team_matches = team_matches.sort_values('Date', ascending=False)
            except:
                pass
            
            # Analyser les 5 derniers matchs
            for _, match in team_matches.head(5).iterrows():
                if match['HomeTeam'] == team:
                    if match['HomeScore'] > match['AwayScore']:
                        team_form.append('W')
                    elif match['HomeScore'] < match['AwayScore']:
                        team_form.append('L')
                    else:
                        team_form.append('D')
                else:
                    if match['AwayScore'] > match['HomeScore']:
                        team_form.append('W')
                    elif match['AwayScore'] < match['HomeScore']:
                        team_form.append('L')
                    else:
                        team_form.append('D')
            
            # Vérifier les séries
            if len(team_form) >= 3:
                if all(result == 'W' for result in team_form[:3]):
                    trends['team_trends'].append({
                        'team': team,
                        'trend': 'winning_streak',
                        'description': f"{team} est sur une série de {team_form[:3].count('W')} victoires consécutives."
                    })
                elif all(result == 'L' for result in team_form[:3]):
                    trends['team_trends'].append({
                        'team': team,
                        'trend': 'losing_streak',
                        'description': f"{team} est sur une série de {team_form[:3].count('L')} défaites consécutives."
                    })
                elif all(result != 'L' for result in team_form[:5]):
                    trends['team_trends'].append({
                        'team': team,
                        'trend': 'unbeaten_run',
                        'description': f"{team} est invaincu depuis {len(team_form)} matchs."
                    })
            
            # Vérifier les anomalies de performance
            position = team_row['Position']
            
            # Équipe bien classée mais mauvaise forme récente
            if position <= 6 and self._calculate_recent_form(team) < -0.3:
                trends['anomalies'].append({
                    'team': team,
                    'type': 'form_position_mismatch',
                    'description': f"{team} est bien classé ({position}e) mais traverse une mauvaise période."
                })
            
            # Équipe mal classée mais bonne forme récente
            if position >= 15 and self._calculate_recent_form(team) > 0.3:
                trends['anomalies'].append({
                    'team': team,
                    'type': 'form_position_mismatch',
                    'description': f"{team} est mal classé ({position}e) mais montre des signes d'amélioration."
                })
            
            # Écart important entre performances à domicile et à l'extérieur
            home_ppg = team_perf['home']['points_per_game'] if team_perf['home']['played'] > 0 else 0
            away_ppg = team_perf['away']['points_per_game'] if team_perf['away']['played'] > 0 else 0
            
            if abs(home_ppg - away_ppg) > 1.5 and team_perf['home']['played'] >= 3 and team_perf['away']['played'] >= 3:
                trends['team_trends'].append({
                    'team': team,
                    'trend': 'home_away_disparity',
                    'description': f"{team} montre un grand écart entre ses performances à domicile ({home_ppg} pts/match) et à l'extérieur ({away_ppg} pts/match)."
                })
        
        # Tendances de la ligue
        if len(played_matches) > 0:
            # Nombre moyen de buts par match
            played_matches['TotalGoals'] = played_matches['HomeScore'] + played_matches['AwayScore']
            avg_goals = played_matches['TotalGoals'].mean()
            
            # Déterminer si le nombre de buts est élevé (> 2.8) ou faible (< 2.2)
            if avg_goals > 2.8:
                trends['league_trends'].append({
                    'type': 'high_scoring',
                    'description': f"La ligue est actuellement à haute intensité offensive avec une moyenne de {avg_goals:.2f} buts par match."
                })
            elif avg_goals < 2.2:
                trends['league_trends'].append({
                    'type': 'low_scoring',
                    'description': f"La ligue est actuellement défensive avec une moyenne de seulement {avg_goals:.2f} buts par match."
                })
            
            # Tendance des victoires à domicile/extérieur
            home_wins = len(played_matches[played_matches['HomeScore'] > played_matches['AwayScore']])
            away_wins = len(played_matches[played_matches['HomeScore'] < played_matches['AwayScore']])
            draws = len(played_matches[played_matches['HomeScore'] == played_matches['AwayScore']])
            
            home_win_pct = home_wins / len(played_matches) * 100
            
            if home_win_pct > 60:
                trends['league_trends'].append({
                    'type': 'strong_home_advantage',
                    'description': f"L'avantage du terrain est actuellement très marqué avec {home_win_pct:.1f}% de victoires à domicile."
                })
            elif home_win_pct < 40:
                trends['league_trends'].append({
                    'type': 'weak_home_advantage',
                    'description': f"L'avantage du terrain est actuellement faible avec seulement {home_win_pct:.1f}% de victoires à domicile."
                })
            
            # Pourcentage de matchs nuls
            draw_pct = draws / len(played_matches) * 100
            
            if draw_pct > 30:
                trends['league_trends'].append({
                    'type': 'high_draw_rate',
                    'description': f"Le taux de matchs nuls est élevé ({draw_pct:.1f}%), indiquant des équipes proches en termes de niveau."
                })
            elif draw_pct < 15:
                trends['league_trends'].append({
                    'type': 'low_draw_rate',
                    'description': f"Le taux de matchs nuls est faible ({draw_pct:.1f}%), indiquant des matchs généralement décisifs."
                })
        
        # Afficher les tendances
        print(f"\n{'=' * 50}")
        print(f"ANALYSE DES TENDANCES - {trends['date']}")
        print(f"{'=' * 50}")
        
        if trends['team_trends']:
            print("\nTENDANCES D'ÉQUIPES:")
            for i, trend in enumerate(trends['team_trends'], 1):
                print(f"{i}. {trend['description']}")
        
        if trends['league_trends']:
            print("\nTENDANCES DE LA LIGUE:")
            for i, trend in enumerate(trends['league_trends'], 1):
                print(f"{i}. {trend['description']}")
        
        if trends['anomalies']:
            print("\nANOMALIES DÉTECTÉES:")
            for i, anomaly in enumerate(trends['anomalies'], 1):
                print(f"{i}. {anomaly['description']}")
        
        return trends
    
    def generate_match_day_preview(self, match_day=None):
        """
        Génère un aperçu des matchs à venir pour une journée spécifique
        
        Args:
            match_day (str): Date de la journée (format: 'YYYY-MM-DD')
            
        Returns:
            dict: Dictionnaire contenant les aperçus des matchs
        """
        if self.fixtures is None:
            print("Aucune donnée de matchs disponible.")
            return None
        
        # Filtrer les matchs à venir
        upcoming_matches = self.fixtures[self.fixtures['Status'] != 'Joué'].copy()
        
        if len(upcoming_matches) == 0:
            print("Aucun match à venir trouvé.")
            return None
        
        # Si aucune journée n'est spécifiée, prendre la prochaine
        if match_day is None:
            try:
                # Convertir les dates et trier
                upcoming_matches['Date'] = pd.to_datetime(upcoming_matches['Date'])
                upcoming_matches = upcoming_matches.sort_values('Date')
                match_day = upcoming_matches['Date'].min().strftime('%Y-%m-%d')
            except:
                # Si la conversion échoue, prendre la première journée disponible
                match_day = upcoming_matches['Date'].iloc[0]
        
        # Filtrer les matchs pour la journée spécifiée
        day_matches = upcoming_matches[upcoming_matches['Date'] == match_day].copy()
        
        if len(day_matches) == 0:
            print(f"Aucun match trouvé pour le {match_day}.")
            return None
        
        # Compiler les aperçus de matchs
        match_previews = []
        
        for _, match in day_matches.iterrows():
            # Prédire le résultat du match
            prediction = self.predict_match(match['HomeTeam'], match['AwayTeam'])
            
            if prediction:
                # Obtenir les performances récentes des équipes
                home_team_form = self._calculate_recent_form(match['HomeTeam'])
                away_team_form = self._calculate_recent_form(match['AwayTeam'])
                
                # Déterminer qui est le favori
                if prediction['probabilities']['home_win'] > prediction['probabilities']['away_win'] + 10:
                    favorite = match['HomeTeam']
                    underdog = match['AwayTeam']
                elif prediction['probabilities']['away_win'] > prediction['probabilities']['home_win'] + 10:
                    favorite = match['AwayTeam']
                    underdog = match['HomeTeam']
                else:
                    favorite = None
                    underdog = None
                
                match_preview = {
                    'match': f"{match['HomeTeam']} vs {match['AwayTeam']}",
                    'date': match_day,
                    'stadium': match['Stadium'] if 'Stadium' in match else 'Non spécifié',
                    'prediction': {
                        'probable_score': prediction['probable_score'],
                        'home_win_prob': prediction['probabilities']['home_win'],
                        'draw_prob': prediction['probabilities']['draw'],
                        'away_win_prob': prediction['probabilities']['away_win']
                    },
                    'team_form': {
                        'home': 'En forme' if home_team_form > 0.3 else 'En difficulté' if home_team_form < -0.3 else 'Moyenne',
                        'away': 'En forme' if away_team_form > 0.3 else 'En difficulté' if away_team_form < -0.3 else 'Moyenne'
                    },
                    'favorite': favorite,
                    'underdog': underdog,
                    'key_battle': f"L'attaque de {match['HomeTeam']} vs la défense de {match['AwayTeam']}"
                }
                
                match_previews.append(match_preview)
        
        # Compiler l'aperçu de la journée
        preview = {
            'date': match_day,
            'matches': match_previews,
            'match_count': len(match_previews)
        }
        
        # Afficher l'aperçu
        print(f"\n{'=' * 50}")
        print(f"APERÇU DE LA JOURNÉE - {preview['date']}")
        print(f"{'=' * 50}")
        
        print(f"\n{preview['match_count']} matchs programmés:")
        
        for i, match_preview in enumerate(preview['matches'], 1):
            print(f"\n{i}. {match_preview['match']} - {match_preview['stadium']}")
            
            if match_preview['favorite']:
                print(f"   Favori: {match_preview['favorite']}")
            else:
                print("   Match équilibré")
            
            print(f"   Prédiction: {match_preview['prediction']['probable_score']} ({match_preview['prediction']['home_win_prob']}% - {match_preview['prediction']['draw_prob']}% - {match_preview['prediction']['away_win_prob']}%)")
            print(f"   Forme: {match_preview['team_form']['home']} vs {match_preview['team_form']['away']}")
            print(f"   Clé du match: {match_preview['key_battle']}")
        
        return preview
    
    def generate_season_projections(self):
        """
        Génère des projections pour la fin de saison en se basant sur les tendances actuelles
        
        Returns:
            dict: Dictionnaire contenant les projections de fin de saison
        """
        if self.standings is None:
            print("Aucune donnée de classement disponible.")
            return None
        
        # Nombre de matchs joués et à jouer
        avg_matches_played = self.standings['Played'].mean()
        remaining_matches = 38 - avg_matches_played  # Saison de 38 matchs en Premier League
        
        # Projections de classement
        projected_standings = self.standings.copy()
        
        # Calculer les points par match pour chaque équipe
        projected_standings['PPM'] = projected_standings['Points'] / projected_standings['Played']
        
        # Projeter les points de fin de saison
        projected_standings['ProjectedPoints'] = projected_standings['Points'] + (projected_standings['PPM'] * remaining_matches)
        projected_standings['ProjectedPoints'] = projected_standings['ProjectedPoints'].round().astype(int)
        
        # Trier par points projetés
        projected_standings = projected_standings.sort_values('ProjectedPoints', ascending=False).reset_index(drop=True)
        projected_standings['ProjectedPosition'] = projected_standings.index + 1
        
        # Calculer la différence de position
        position_map = {row['Team']: row['Position'] for _, row in self.standings.iterrows()}
        projected_standings['PositionChange'] = projected_standings.apply(
            lambda x: position_map[x['Team']] - x['ProjectedPosition'], axis=1
        )
        
        # Identifier les équipes en lice pour le titre, qualifications européennes et relégation
        title_contenders = projected_standings[projected_standings['ProjectedPosition'] <= 3]['Team'].tolist()
        champions_league = projected_standings[projected_standings['ProjectedPosition'] <= 4]['Team'].tolist()
        europa_league = projected_standings[(projected_standings['ProjectedPosition'] > 4) & (projected_standings['ProjectedPosition'] <= 6)]['Team'].tolist()
        relegation_candidates = projected_standings[projected_standings['ProjectedPosition'] >= 18]['Team'].tolist()
        
        # Compiler les projections
        projections = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'matches_played_avg': round(avg_matches_played, 1),
            'remaining_matches_avg': round(remaining_matches, 1),
            'projected_standings': projected_standings[['ProjectedPosition', 'Team', 'Played', 'Points', 'ProjectedPoints', 'PositionChange']].to_dict('records'),
            'title_contenders': title_contenders,
            'champions_league': champions_league,
            'europa_league': europa_league,
            'relegation_candidates': relegation_candidates
        }
        
        # Afficher les projections
        print(f"\n{'=' * 50}")
        print(f"PROJECTIONS DE FIN DE SAISON - {projections['date']}")
        print(f"{'=' * 50}")
        
        print(f"\nMoyenne de matchs joués: {projections['matches_played_avg']}")
        print(f"Moyenne de matchs restants: {projections['remaining_matches_avg']}")
        
        print("\nPROJECTION DU CLASSEMENT FINAL:")
        projection_df = pd.DataFrame(projections['projected_standings'])
        print(tabulate(projection_df, headers='keys', tablefmt='pretty', showindex=False))
        
        print("\nÉQUIPES EN LICE POUR LE TITRE:")
        for team in title_contenders:
            print(f"- {team}")
        
        print("\nQUALIFICATION CHAMPIONS LEAGUE:")
        for team in champions_league:
            print(f"- {team}")
        
        print("\nQUALIFICATION EUROPA LEAGUE:")
        for team in europa_league:
            print(f"- {team}")
        
        print("\nCANDIDATS À LA RELÉGATION:")
        for team in relegation_candidates:
            print(f"- {team}")
        
        return projections
    
    def export_data_to_csv(self, output_dir=None):
        """
        Exporte les données analysées en fichiers CSV
        
        Args:
            output_dir (str): Répertoire de sortie (par défaut, crée un sous-répertoire 'exports')
            
        Returns:
            bool: True si l'exportation a réussi, False sinon
        """
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, 'exports')
        
        # Créer le répertoire de sortie s'il n'existe pas
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            # Date pour les noms de fichiers
            date_str = datetime.now().strftime("%Y%m%d")
            
            # Exporter le classement
            if self.standings is not None:
                self.standings.to_csv(os.path.join(output_dir, f"standings_{date_str}.csv"), index=False)
                print(f"Classement exporté: {os.path.join(output_dir, f'standings_{date_str}.csv')}")
            
            # Exporter les matchs
            if self.fixtures is not None:
                self.fixtures.to_csv(os.path.join(output_dir, f"fixtures_{date_str}.csv"), index=False)
                print(f"Matchs exportés: {os.path.join(output_dir, f'fixtures_{date_str}.csv')}")
            
            # Exporter les projections
            projections = self.generate_season_projections()
            if projections:
                projection_df = pd.DataFrame(projections['projected_standings'])
                projection_df.to_csv(os.path.join(output_dir, f"projections_{date_str}.csv"), index=False)
                print(f"Projections exportées: {os.path.join(output_dir, f'projections_{date_str}.csv')}")
            
            # Exporter les tendances
            trends = self.identify_trends()
            if trends:
                # Créer des DataFrames pour les tendances
                team_trends_df = pd.DataFrame(trends['team_trends'])
                league_trends_df = pd.DataFrame(trends['league_trends'])
                anomalies_df = pd.DataFrame(trends['anomalies'])
                
                if not team_trends_df.empty:
                    team_trends_df.to_csv(os.path.join(output_dir, f"team_trends_{date_str}.csv"), index=False)
                    print(f"Tendances d'équipes exportées: {os.path.join(output_dir, f'team_trends_{date_str}.csv')}")
                
                if not league_trends_df.empty:
                    league_trends_df.to_csv(os.path.join(output_dir, f"league_trends_{date_str}.csv"), index=False)
                    print(f"Tendances de ligue exportées: {os.path.join(output_dir, f'league_trends_{date_str}.csv')}")
                
                if not anomalies_df.empty:
                    anomalies_df.to_csv(os.path.join(output_dir, f"anomalies_{date_str}.csv"), index=False)
                    print(f"Anomalies exportées: {os.path.join(output_dir, f'anomalies_{date_str}.csv')}")
            
            # Exporter les statistiques d'équipe
            for team, stats in self.team_stats.items():
                team_name = team.replace(' ', '_').lower()
                with open(os.path.join(output_dir, f"{team_name}_stats_{date_str}.json"), 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=4)
                print(f"Statistiques de {team} exportées: {os.path.join(output_dir, f'{team_name}_stats_{date_str}.json')}")
            
            return True
        
        except Exception as e:
            print(f"Erreur lors de l'exportation des données: {e}")
            return False
    
    def plot_league_table(self):
        """
        Génère une visualisation graphique du classement de la Premier League
        
        Returns:
            matplotlib.figure.Figure: L'objet figure créé
        """
        if self.standings is None:
            print("Aucune donnée de classement disponible.")
            return None
        
        # Créer une copie du classement pour la visualisation
        vis_standings = self.standings.copy()
        
        # Limiter aux 10 premières équipes pour la lisibilité
        vis_standings = vis_standings.head(10)
        
        # Créer la figure
        plt.figure(figsize=(12, 8))
        
        # Barres pour les points
        bars = plt.bar(vis_standings['Team'], vis_standings['Points'], color='skyblue')
        
        # Ajouter les valeurs au-dessus des barres
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"{int(height)}",
                    ha='center', va='bottom')
        
        # Ajouter les positions au-dessus du texte des points
        for i, (_, row) in enumerate(vis_standings.iterrows()):
            plt.text(i, row['Points'] + 5, f"{row['Position']}.", ha='center', va='bottom', 
                    color='darkblue', fontweight='bold', fontsize=12)
        
        # Ajouter les lignes horizontales pour les qualifications
        plt.axhline(y=vis_standings['Points'].iloc[3], color='green', linestyle='--', alpha=0.7)
        plt.text(len(vis_standings) - 1, vis_standings['Points'].iloc[3] + 1, 'Champions League', 
                ha='right', va='bottom', color='green')
        
        if len(vis_standings) > 5:
            plt.axhline(y=vis_standings['Points'].iloc[5], color='orange', linestyle='--', alpha=0.7)
            plt.text(len(vis_standings) - 1, vis_standings['Points'].iloc[5] + 1, 'Europa League', 
                    ha='right', va='bottom', color='orange')
        
        # Ajouter les titres et étiquettes
        plt.title('Classement Premier League', fontsize=15)
        plt.xlabel('Équipes')
        plt.ylabel('Points')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ajuster la disposition
        plt.tight_layout()
        
        # Sauvegarder l'image
        plt.savefig(f"{self.data_dir}/league_table.png")
        
        return plt.gcf()