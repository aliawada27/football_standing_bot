#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from scraper import PremierLeagueScraper
from analyzer import PremierLeagueAnalyzer
from visualizer import PremierLeagueVisualizer

def main():
    """
    Programme principal pour le scraping, l'analyse et la visualisation des données de Premier League
    """
    # Création du parser d'arguments
    parser = argparse.ArgumentParser(description='Scraper, Analyseur et Visualiseur de données de Premier League')
    
    # Sous-commandes
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande scrape
    scrape_parser = subparsers.add_parser('scrape', help='Récupérer les données de Premier League')
    scrape_parser.add_argument('--all', action='store_true', help='Récupérer toutes les données disponibles')
    scrape_parser.add_argument('--standings', action='store_true', help='Récupérer le classement')
    scrape_parser.add_argument('--fixtures', action='store_true', help='Récupérer les matchs')
    scrape_parser.add_argument('--team', type=str, help='Récupérer les statistiques d\'une équipe spécifique')
    scrape_parser.add_argument('--cache-dir', type=str, default='data', help='Répertoire de cache (défaut: data)')
    
    # Commande analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analyser les données de Premier League')
    analyze_parser.add_argument('--report', action='store_true', help='Générer un rapport de ligue')
    analyze_parser.add_argument('--trends', action='store_true', help='Identifier les tendances')
    analyze_parser.add_argument('--team', type=str, help='Analyser une équipe spécifique')
    analyze_parser.add_argument('--compare', type=str, nargs=2, metavar=('TEAM1', 'TEAM2'), help='Comparer deux équipes')
    analyze_parser.add_argument('--predict', type=str, nargs=2, metavar=('HOME', 'AWAY'), help='Prédire le résultat d\'un match')
    analyze_parser.add_argument('--preview', action='store_true', help='Générer un aperçu de la prochaine journée')
    analyze_parser.add_argument('--projections', action='store_true', help='Générer des projections de fin de saison')
    analyze_parser.add_argument('--data-dir', type=str, default='data', help='Répertoire de données (défaut: data)')
    analyze_parser.add_argument('--export', action='store_true', help='Exporter les données analysées')
    analyze_parser.add_argument('--visual', action='store_true', help='Générer des visualisations')
    
    # Commande visualize
    visualize_parser = subparsers.add_parser('visualize', help='Visualiser les données de Premier League')
    visualize_parser.add_argument('--data-dir', type=str, default='data', help='Répertoire de données (défaut: data)')
    visualize_parser.add_argument('--output-dir', type=str, help='Répertoire de sortie pour les visualisations (défaut: data/visualizations)')
    visualize_parser.add_argument('--no-show', action='store_true', help='Ne pas afficher les visualisations (seulement sauvegarder)')
    visualize_parser.add_argument('--no-save', action='store_true', help='Ne pas sauvegarder les visualisations (seulement afficher)')
    
    visualize_subparsers = visualize_parser.add_subparsers(dest='viz_command', help='Types de visualisations disponibles')
    
    # Sous-commandes de visualisation
    table_parser = visualize_subparsers.add_parser('table', help='Visualiser le tableau de classement')
    table_parser.add_argument('--top', type=int, default=20, help='Nombre d\'équipes à afficher (défaut: toutes)')
    
    form_parser = visualize_subparsers.add_parser('form', help='Visualiser la forme d\'une équipe')
    form_parser.add_argument('team', type=str, help='Nom de l\'équipe')
    form_parser.add_argument('--matches', type=int, default=10, help='Nombre de matchs à considérer (défaut: 10)')
    
    compare_parser = visualize_subparsers.add_parser('compare', help='Comparer deux équipes')
    compare_parser.add_argument('team1', type=str, help='Nom de la première équipe')
    compare_parser.add_argument('team2', type=str, help='Nom de la deuxième équipe')
    
    goals_parser = visualize_subparsers.add_parser('goals', help='Visualiser la distribution des buts')
    
    stats_parser = visualize_subparsers.add_parser('stats', help='Visualiser les statistiques détaillées d\'une équipe')
    stats_parser.add_argument('team', type=str, help='Nom de l\'équipe')
    
    projections_parser = visualize_subparsers.add_parser('projections', help='Visualiser les projections de fin de saison')
    
    dashboard_parser = visualize_subparsers.add_parser('dashboard', help='Générer un tableau de bord')
    dashboard_parser.add_argument('--team', type=str, help='Nom de l\'équipe (facultatif, pour un tableau de bord spécifique)')
    
    all_parser = visualize_subparsers.add_parser('all', help='Générer toutes les visualisations possibles')
    all_parser.add_argument('--team', type=str, help='Nom de l\'équipe pour des visualisations spécifiques')
    
    # Récupération des arguments
    args = parser.parse_args()
    
    # Si aucune commande n'est spécifiée, afficher l'aide
    if not args.command:
        parser.print_help()
        return
    
    # Exécution des commandes
    if args.command == 'scrape':
        # Initialisation du scraper
        scraper = PremierLeagueScraper(cache_dir=args.cache_dir)
        
        # Scraping selon les options
        if args.all or args.standings:
            standings = scraper.scrape_standings()
            if standings is not None:
                print(f"Classement récupéré: {len(standings)} équipes")
        
        if args.all or args.fixtures:
            fixtures = scraper.scrape_fixtures()
            if fixtures is not None:
                print(f"Matchs récupérés: {len(fixtures)} matchs")
        
        if args.all:
            all_teams_stats = scraper.scrape_all_teams_stats()
            if all_teams_stats:
                print(f"Statistiques récupérées pour {len(all_teams_stats)} équipes")
        elif args.team:
            team_stats = scraper.scrape_team_stats(team_name=args.team)
            if team_stats:
                print(f"Statistiques récupérées pour {args.team}")
    
    elif args.command == 'analyze':
        # Initialisation de l'analyseur
        analyzer = PremierLeagueAnalyzer(data_dir=args.data_dir)
        
        # Chargement des données
        if analyzer.load_latest_data():
            print("Données chargées avec succès!")
            
            # Analyse selon les options
            if args.report:
               # analyzer.generate_league_report() ////// CHANGER SI ON UTILISE LE API
            
            if args.trends:
                analyzer.identify_trends()
            
            if args.team:
                analyzer.print_team_performance(args.team)
                
                if args.visual:
                    # Initialisation du visualiseur si nécessaire
                    visualizer = PremierLeagueVisualizer(data_dir=args.data_dir)
                    visualizer.visualize_team_form(args.team)
                    visualizer.visualize_team_stats(args.team)
                    print(f"Graphiques créés pour {args.team}")
            
            if args.compare:
                team1, team2 = args.compare
                analyzer.compare_teams(team1, team2)
                
                if args.visual:
                    # Initialisation du visualiseur si nécessaire
                    visualizer = PremierLeagueVisualizer(data_dir=args.data_dir)
                    visualizer.visualize_team_comparison(team1, team2)
                    print(f"Graphique de comparaison créé pour {team1} vs {team2}")
            
            if args.predict:
                home_team, away_team = args.predict
                analyzer.predict_match(home_team, away_team)
            
            if args.preview:
                analyzer.generate_match_day_preview()
            
            if args.projections:
                analyzer.generate_season_projections()
                
                if args.visual:
                    # Initialisation du visualiseur si nécessaire
                    visualizer = PremierLeagueVisualizer(data_dir=args.data_dir)
                    visualizer.visualize_season_projections()
                    print("Graphique des projections créé")
            
            if args.visual and not (args.team or args.compare or args.projections):
                # Initialisation du visualiseur si nécessaire
                visualizer = PremierLeagueVisualizer(data_dir=args.data_dir)
                visualizer.visualize_league_table()
                visualizer.visualize_goal_distribution()
                visualizer.generate_dashboard()
                print("Visualisations générales créées")
            
            if args.export:
                if analyzer.export_data_to_csv():
                    print("Données exportées avec succès!")
        else:
            print("Échec du chargement des données. Veuillez d'abord exécuter le scraper.")
    
    elif args.command == 'visualize':
        # Initialisation du visualiseur
        output_dir = args.output_dir
        visualizer = PremierLeagueVisualizer(data_dir=args.data_dir, output_dir=output_dir)
        
        # Options pour l'affichage et la sauvegarde
        show = not args.no_show
        save = not args.no_save
        
        # Si aucune sous-commande n'est spécifiée, afficher l'aide
        if not hasattr(args, 'viz_command') or not args.viz_command:
            visualize_parser.print_help()
            return
        
        # Exécution des commandes de visualisation
        if args.viz_command == 'table':
            visualizer.visualize_league_table(top_n=args.top, save=save, show=show)
            
        elif args.viz_command == 'form':
            visualizer.visualize_team_form(args.team, num_matches=args.matches, save=save, show=show)
            
        elif args.viz_command == 'compare':
            visualizer.visualize_team_comparison(args.team1, args.team2, save=save, show=show)
            
        elif args.viz_command == 'goals':
            visualizer.visualize_goal_distribution(save=save, show=show)
            
        elif args.viz_command == 'stats':
            visualizer.visualize_team_stats(args.team, save=save, show=show)
            
        elif args.viz_command == 'projections':
            visualizer.visualize_season_projections(save=save, show=show)
            
        elif args.viz_command == 'dashboard':
            visualizer.generate_dashboard(team_name=args.team, save=save, show=show)
            
        elif args.viz_command == 'all':
            if args.team:
                print(f"Génération de toutes les visualisations pour {args.team}...")
                visualizer.visualize_team_form(args.team, save=save, show=show)
                visualizer.visualize_team_stats(args.team, save=save, show=show)
                visualizer.generate_dashboard(team_name=args.team, save=save, show=show)
                # Trouver un adversaire pour la comparaison (premier au classement si c'est une autre équipe)
                if visualizer.standings is not None:
                    opponent = visualizer.standings.iloc[0]['Team'] if visualizer.standings.iloc[0]['Team'] != args.team else visualizer.standings.iloc[1]['Team']
                    visualizer.visualize_team_comparison(args.team, opponent, save=save, show=show)
            else:
                print("Génération de toutes les visualisations de la ligue...")
                visualizer.visualize_league_table(save=save, show=show)
                visualizer.visualize_goal_distribution(save=save, show=show)
                visualizer.visualize_season_projections(save=save, show=show)
                visualizer.generate_dashboard(save=save, show=show)

if __name__ == "__main__":
    main()