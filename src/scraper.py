import requests
import pandas as pd
import time
import os
from datetime import datetime
import json

class PremierLeagueScraper:
    def __init__(self, cache_dir='data', api_key=None):
        """
        Initialise le scraper de Premier League utilisant l'API football-data.org
        
        Args:
            cache_dir (str): Répertoire pour stocker les données téléchargées
            api_key (str): Clé API pour football-data.org (optionnel pour certaines requêtes)
        """
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {
            "898ed9dd596b4e5884921b784015d0cb": api_key
        }
        self.premier_league_id = 2021  # ID de la Premier League dans l'API
        self.season = '2023'  # Saison en cours (à mettre à jour selon la saison)
        self.cache_dir = cache_dir
        
        # Création du dossier de cache s'il n'existe pas
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def _make_request(self, endpoint, params=None, delay=1):
        """
        Effectue une requête API avec gestion des erreurs et délai
        
        Args:
            endpoint (str): Point de terminaison de l'API
            params (dict): Paramètres de requête
            delay (int): Délai en secondes entre les requêtes
            
        Returns:
            dict: Réponse JSON de l'API
        """
        time.sleep(delay)  # Pause pour respecter les limites de l'API
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print("Limite d'API atteinte. Attente de 60 secondes avant de réessayer...")
                time.sleep(60)
                return self._make_request(endpoint, params, 0)  # Réessayer sans délai supplémentaire
            else:
                print(f"Erreur API: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête vers {endpoint}: {e}")
            return None
    
    def scrape_standings(self):
        """
        Récupère le classement actuel de la Premier League
        
        Returns:
            pandas.DataFrame: DataFrame contenant le classement
        """
        print("Récupération du classement de Premier League...")
        
        endpoint = f"competitions/{self.premier_league_id}/standings"
        data = self._make_request(endpoint)
        
        if not data or 'standings' not in data:
            print("Impossible de récupérer le classement")
            return None
        
        # Extraire le classement total (pas à domicile/extérieur)
        standings_data = next((item for item in data['standings'] if item['type'] == 'TOTAL'), None)
        
        if not standings_data or 'table' not in standings_data:
            print("Structure des données inattendue")
            return None
        
        teams_data = []
        
        for entry in standings_data['table']:
            team = entry['team']
            team_data = {
                'Position': entry['position'],
                'Team': team['name'],
                'TeamID': team['id'],
                'Played': entry['playedGames'],
                'Won': entry['won'],
                'Drawn': entry['draw'],
                'Lost': entry['lost'],
                'GF': entry['goalsFor'],
                'GA': entry['goalsAgainst'],
                'GD': entry['goalDifference'],
                'Points': entry['points']
            }
            teams_data.append(team_data)
        
        df = pd.DataFrame(teams_data)
        
        # Sauvegarde des données dans le cache
        today = datetime.now().strftime("%Y-%m-%d")
        df.to_csv(f"{self.cache_dir}/standings_{today}.csv", index=False)
        
        return df
    
    def scrape_team_stats(self, team_id=None, team_name=None):
        """
        Récupère les statistiques détaillées d'une équipe
        
        Args:
            team_id (int): ID de l'équipe dans l'API
            team_name (str): Nom de l'équipe (si l'ID n'est pas fourni)
            
        Returns:
            dict: Dictionnaire contenant les statistiques de l'équipe
        """
        if not team_id and not team_name:
            print("Veuillez fournir soit l'ID de l'équipe, soit son nom")
            return None
        
        # Si nous n'avons que le nom, nous devons d'abord obtenir l'ID
        if not team_id:
            standings = self.scrape_standings()
            if standings is None:
                return None
                
            team_row = standings[standings['Team'] == team_name]
            
            if team_row.empty:
                print(f"Équipe non trouvée: {team_name}")
                return None
                
            team_id = team_row['TeamID'].values[0]
        
        print(f"Récupération des statistiques pour l'équipe ID {team_id}...")
        
        # Récupérer les informations de base de l'équipe
        endpoint = f"teams/{team_id}"
        team_data = self._make_request(endpoint)
        
        if not team_data:
            return None
        
        # Récupérer les matchs de l'équipe pour calculer des statistiques supplémentaires
        endpoint = f"teams/{team_id}/matches"
        params = {"competitions": self.premier_league_id, "season": self.season}
        matches_data = self._make_request(endpoint, params)
        
        if not matches_data or 'matches' not in matches_data:
            return None
        
        matches = matches_data['matches']
        
        # Calculer des statistiques avancées
        total_matches = len([m for m in matches if m['status'] == 'FINISHED'])
        
        goals_for = 0
        goals_against = 0
        clean_sheets = 0
        
        for match in matches:
            if match['status'] != 'FINISHED':
                continue
                
            is_home = match['homeTeam']['id'] == team_id
            
            if is_home:
                goals_for += match['score']['fullTime']['home'] or 0
                goals_against += match['score']['fullTime']['away'] or 0
                if (match['score']['fullTime']['away'] or 0) == 0:
                    clean_sheets += 1
            else:
                goals_for += match['score']['fullTime']['away'] or 0
                goals_against += match['score']['fullTime']['home'] or 0
                if (match['score']['fullTime']['home'] or 0) == 0:
                    clean_sheets += 1
        
        # Compiler les statistiques
        team_stats = {
            'id': team_id,
            'name': team_data.get('name', ''),
            'shortName': team_data.get('shortName', ''),
            'tla': team_data.get('tla', ''),
            'crest': team_data.get('crest', ''),
            'stats': {
                'General': {
                    'Matchs joués': total_matches,
                    'Buts marqués': goals_for,
                    'Buts encaissés': goals_against,
                    'Différence de buts': goals_for - goals_against,
                    'Clean sheets': clean_sheets
                }
            },
            'squad': team_data.get('squad', []),
            'matches': matches
        }
        
        # Ajouter des statistiques pour les joueurs si disponibles
        if 'squad' in team_data:
            team_stats['top_players'] = {
                'Squad': [{'name': player['name'], 'value': player.get('position', 'N/A')} 
                          for player in team_data['squad'][:10]]
            }
        
        # Sauvegarder au format JSON pour préserver la structure imbriquée
        today = datetime.now().strftime("%Y-%m-%d")
        team_file_name = team_data.get('shortName', 'team').replace(' ', '_').lower()
        
        with open(f"{self.cache_dir}/{team_file_name}_stats_{today}.json", 'w', encoding='utf-8') as f:
            json.dump(team_stats, f, ensure_ascii=False, indent=4)
        
        return team_stats
    
    def scrape_fixtures(self, team_id=None):
        """
        Récupère les matchs passés et à venir
        
        Args:
            team_id (int, optional): Si spécifié, récupère uniquement les matchs de cette équipe
            
        Returns:
            pandas.DataFrame: DataFrame contenant les matchs
        """
        print("Récupération des matchs...")
        
        if team_id:
            endpoint = f"teams/{team_id}/matches"
            params = {"competitions": self.premier_league_id, "season": self.season}
        else:
            endpoint = f"competitions/{self.premier_league_id}/matches"
            params = {"season": self.season}
        
        data = self._make_request(endpoint, params)
        
        if not data or 'matches' not in data:
            print("Impossible de récupérer les matchs")
            return None
        
        fixtures_data = []
        
        for match in data['matches']:
            match_data = {
                'MatchID': match['id'],
                'Date': match['utcDate'],
                'HomeTeam': match['homeTeam']['name'],
                'AwayTeam': match['awayTeam']['name'],
                'HomeID': match['homeTeam']['id'],
                'AwayID': match['awayTeam']['id'],
                'Status': match['status']
            }
            
            # Ajouter les scores si le match est terminé
            if match['status'] == 'FINISHED':
                match_data['HomeScore'] = match['score']['fullTime']['home']
                match_data['AwayScore'] = match['score']['fullTime']['away']
                match_data['Status'] = 'Joué'
            else:
                match_data['HomeScore'] = None
                match_data['AwayScore'] = None
                if match['status'] == 'SCHEDULED':
                    match_data['Status'] = datetime.fromisoformat(match['utcDate'].replace('Z', '+00:00')).strftime('%H:%M')
                elif match['status'] == 'POSTPONED':
                    match_data['Status'] = 'Reporté'
                else:
                    match_data['Status'] = match['status']
            
            # Ajouter le stade si disponible
            match_data['Stadium'] = match.get('venue', 'Non spécifié')
            
            fixtures_data.append(match_data)
        
        df = pd.DataFrame(fixtures_data)
        
        # Sauvegarde des données dans le cache
        today = datetime.now().strftime("%Y-%m-%d")
        file_name = f"fixtures_{'team_' + str(team_id) if team_id else 'all'}_{today}.csv"
        df.to_csv(f"{self.cache_dir}/{file_name}", index=False)
        
        return df
    
    def scrape_all_teams_stats(self):
        """
        Récupère les statistiques de toutes les équipes
        
        Returns:
            dict: Dictionnaire avec les noms d'équipes comme clés et leurs statistiques comme valeurs
        """
        standings = self.scrape_standings()
        
        if standings is None:
            return None
            
        all_teams_stats = {}
        
        print("Récupération des stats d'équipes...")
        total_teams = len(standings)
        
        for i, (_, team) in enumerate(standings.iterrows(), 1):
            print(f"Équipe {i}/{total_teams}: {team['Team']}")
            team_stats = self.scrape_team_stats(team_id=team['TeamID'], team_name=team['Team'])
            if team_stats:
                all_teams_stats[team['Team']] = team_stats
                
            # Pause pour éviter d'atteindre les limites de l'API
            time.sleep(2)
        
        return all_teams_stats