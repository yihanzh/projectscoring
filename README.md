# Projet 7 - OpenClassrooms

## Parcours Data Scientist - Projet 7 : Implémentez un modèle de scoring

### Source de données :
- [Données sur Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)

### Tableau de bord déployé sur Heroku :
- [Lien vers le tableau de bord](https://testyihanscoring1-94660854c600.herokuapp.com/)

## Contexte :

Nous sommes des Data Scientists au sein de la société financière "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas d'historique de prêt. L'entreprise souhaite mettre en œuvre un outil de "scoring crédit" pour calculer la probabilité qu'un client rembourse son crédit, puis classifier la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s'appuyant sur des sources de données variées (données comportementales, données provenant des institutions financières, etc.).

De plus, les chargés de relation client ont remonté le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d'octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l'entreprise veut incarner. Prêt à dépenser décide donc de développer un tableau de bord interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d'octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

## Notre mission :

- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- Construire un tableau de bord interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d'améliorer la connaissance client des chargés de relation client.
- Mettre en production le modèle de scoring de prédiction à l'aide d'une API, ainsi que le tableau de bord interactif qui appelle l'API pour les prédictions.

## Description des dossiers et fichiers :

- **Notebook** : contient le code du feature engineering, de la préparation des données, de la modélisation et de l'export du modèle.
- **Dossier data** : dossier contenant les données de validation pour les modèles.
- **app.py** : Ce script Python définit une API Flask pour interagir avec un modèle de prédiction.
- **Dossier API** : 
  - *model.py* : Ce script Python définit une classe Model utilisée pour charger un modèle de prédiction, effectuer des prédictions, générer des graphiques SHAP (SHapley Additive exPlanations), des distributions de variables et des graphiques bi-variés.
- **Dossier dashboard** : dossier contenant les fichiers liés au fonctionnement du tableau de bord.
  - *dashboard.py* : Ce script Python est un tableau de bord interactif développé avec Streamlit pour visualiser les prédictions d'un modèle de scoring de crédit.
  - *setup_streamlit.sh* : Ce script Shell crée un fichier de configuration pour Streamlit, un outil utilisé pour développer des applications web en Python.
  - *logo.png* : image du logo de la société "Prêt à dépenser" utilisé dans le tableau de bord.
- **Dossier data_drift** : 
  - *data_drift_evidently.py* : Ce script Python utilise la bibliothèque Evidently pour détecter les dérives de données entre deux ensembles de données, généralement une référence (utilisée pour l'entraînement du modèle) et des données actuelles (en production).
- **Dossier models** : les modèles de prédiction.
- **Procfile** : Fichier pour le déploiement de l'API et du tableau de bord sur Heroku. Il décrit les processus de démarrage pour une application web.
- **Requirements.txt** : Ce fichier contient la liste des librairies requises pour le projet.
- **unit_tests.py** : tests unitaires pour le workflow d'intégration continue, déclenchés par Github Action, réalisés avant le déploiement sur Heroku.
- **Dossier .github/workflows** :
  - *unit_tests.yml* : fichier de configuration YAML utilisé pour définir un workflow dans GitHub Actions. 
- **data_drift_report.html** : rapport au format HTML sur la dérive des données entre les données d'entraînement et les données de production. (télécharger et ouvrir dans un navigateur)
- **Vidéo de démonstration du Tableau de bord**
  https://youtu.be/XySE7vcahZg
