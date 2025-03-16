# Premier League Data Analysis Tool

Un outil complet de scraping, analyse et visualisation des données de la Premier League.

## Fonctionnalités

- Scraping des données en temps réel
- Analyse détaillée des performances des équipes
- Visualisations avancées
- Prédictions de matchs
- Projections de fin de saison

## Prérequis

- Python 
- Bibliothèques : pandas, numpy, requests, beautifulsoup4, matplotlib, seaborn

## Installation

```bash
git clone https://github.com/aliawada27/premier-league-analyzer.git
cd premier-league-analyzer
pip install -r requirements.txt
```

## Utilisation

### Scraper les données

```bash
python main.py scrape --all
```

### Analyser les données

```bash
# Rapport général de la ligue
python main.py analyze --report

# Analyse d'une équipe spécifique
python main.py analyze --team "Manchester City"

# Générer des visualisations
python main.py analyze --visual
```
