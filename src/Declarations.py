 # I- Importations de mmodules

## I-1 Modules pour l'importation des données api
import requests
import openpyxl
import zipfile


## I-2 Modules pour l'analyse de données et manipulations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


## I-3 Modules pour la visualisation des données
import geopandas as gpd
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px

## I-4. Imports liés à la modélisation et à la statistique
import statsmodels.api as sm
import statsmodels.tsa.filters.hp_filter as smf
import statsmodels.tsa.ardl as sma
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

## I-5. Imprts liés au clusteriing 

import plotly.graph_objects as go 
import plotly.graph_objs as go
from pywaffle import Waffle
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from time import time
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import silhouette_visualizer
import operator
from sklearn import manifold
import plotly.express as px
from sklearn.mixture import GaussianMixture


# II- Déclarations de fonctions

## II-1- Fonctions pour extraire les codes ISO alpha-3 des pays du monde

def get_worldbank_country_codes(url):
    """
    Récupère les codes ISO alpha-3 des pays dans la base de données de la Banque mondiale,
    en excluant les régions et unions économiques.
    
    Returns:
        list: Chaine de caractère des codes ISO alpha-3 des pays uniquement.
    """
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Erreur lors de la requête API : {response.status_code}")
    
    data = response.json()
    
    # Vérification de la présence des données
    if len(data) < 2 or not data[1]:
        raise Exception("Aucune donnée disponible sur les pays")
    
    # Convertir les données en DataFrame
    df = pd.json_normalize(data[1])
    
    # Filtrer uniquement les entités qui sont des pays
    countries_df = df[df['region.id'] != 'NA']  # Exclure les entités non classifiées (il s'agit d'ensemble de pays. Ex: East Asia & Pacific (Code : EAS) : Région, pas un pays)
    countries_df = countries_df[countries_df['region.id'] != '']  # Exclure les entités sans région (c'est des ensembles de pays également)
    countries_df = countries_df[countries_df['incomeLevel.id'] != '']  # Exclure les entités sans niveau de revenu (Les régions ou entités globales (comme "World") n'ont pas de classification de revenu)
    
    # Extraire les codes ISO alpha-3 des pays
    country_codes = countries_df['id'].tolist()

    # Convertir la liste en chaine de caractère avec des points virgules
    country_list = ";".join(country_codes)
    
    return country_list



## II-2- Fonctions pour importer les données des pays pour un indicateur donné


def fetch_worldbank_data(countries, indicator, start_year, end_year):
    """
    Télécharge les données de l'API de la Banque mondiale pour un indicateur.
    
    Args:
        countries (str): Liste de codes ISO alpha-3 des pays séparés par ';' (ex: 'USA;FRA;DEU').
        indicator (str): Code de l'indicateur (ex: 'NY.GDP.MKTP.KD').
        start_year (int): Année de début pour la plage de données.
        end_year (int): Année de fin pour la plage de données.
    
    Returns:
        pd.DataFrame: Base de données contenant les pays, les codes ISO alpha-3 des pays,
                      le nom de l'indicateur, l'année et la valeur de l'indicateur.
    """
    # Dictionnaire des indicateurs
    indicators = {
        'NY.GDP.MKTP.KD': 'GDP (constant 2015 US$)',
        'NE.GDI.FTOT.KD': 'Gross Fixed Capital Formation (constant 2015 US$)',
        'SP.POP.1564.TO': 'Population ages 15-64, total',
        'NE.GDI.FTOT.ZS': 'Gross Fixed Capital Formation (% GDP)'
    }
    
    # URL pour télécharger les données
    url = f"http://api.worldbank.org/v2/country/{countries}/indicator/{indicator}?date={start_year}:{end_year}&format=json&per_page=12000"
    response = requests.get(url)

    # Vérifier si la requête a réussi
    if response.status_code != 200:
        raise Exception(f"Erreur lors de la requête API : {response.status_code}")
    
    data = response.json()
    
    # Vérifier si des données sont disponibles
    if len(data) < 2 or not data[1]:
        print(f"Aucune donnée disponible pour l'indicateur {indicator} et les pays {countries}")
        return pd.DataFrame()  # Retourner un DataFrame vide
    else:
        print("Téléchargement de données réussi")
    
    # Extraire et normaliser les données
    records = data[1]
    df = pd.json_normalize(records)

    # Récupérer le nom de l'indicateur
    indicator_name = indicators.get(indicator, indicator)  # Utiliser le nom ou le code si non trouvé

    # Garder les colonnes pertinentes et les renommer
    df = df[['country.value', 'countryiso3code', 'date', 'value']].rename(columns={
        'country.value': 'Country',
        'countryiso3code': 'Country_Code',
        'date': 'Year',
        'value': indicator_name  # Nom de l'indicateur dans la colonne
    })

    # Vérifier que les codes pays sont au format ISO alpha-3 (facultatif)
    df = df[df['Country_Code'].str.len() == 3]
    
    return df



# II-3- Fonction pour afficher le pourcentage de valeurs manquantes pour chaque pays
def calculate_missing_percentage(df, column_name):
    """
    Calcule le pourcentage de valeurs manquantes pour chaque pays dans une colonne spécifiée.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        column_name (str): Le nom de la colonne dont le pourcentage de valeurs manquantes doit être calculé.
    
    Returns:
        pd.DataFrame: DataFrame contenant les pays et leur pourcentage de valeurs manquantes pour la colonne spécifiée.
    """
    # Calculer le nombre total d'années de données disponibles
    total_years = df['Year'].nunique()

    # Calculer le pourcentage de valeurs manquantes par pays
    missing_percentage_by_country = df.groupby('Country')[column_name].apply(
        lambda x: x.isnull().sum() / total_years * 100)
    
    return missing_percentage_by_country



# II-4 Fonction pour supprimer les pays avec plus de 20% de valeurs manquantes
def remove_countries_with_missing_values(df, column_name, threshold=20):
    """
    Supprime les pays dont les valeurs manquantes dans une colonne spécifiée dépassent un certain seuil (20% par défaut).
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        column_name (str): Le nom de la colonne sur laquelle le seuil de valeurs manquantes est appliqué.
        threshold (float): Seuil de pourcentage de valeurs manquantes (par défaut 0.2 pour 20%).
    
    Returns:
        pd.DataFrame: DataFrame avec les pays dont le pourcentage de valeurs manquantes est inférieur au seuil.
    """
    # Calculer le pourcentage de valeurs manquantes par pays
    missing_percentage_by_country = calculate_missing_percentage(df, column_name)
    
    # Filtrer les pays dont les valeurs manquantes sont inférieures au seuil (en pourcentage)
    valid_countries = missing_percentage_by_country[missing_percentage_by_country <= threshold].index
    invalid_countries = missing_percentage_by_country[missing_percentage_by_country > threshold].index
    
    # Afficher les pays supprimés
    print(f"Pays supprimés (avec plus de {threshold}% de valeurs manquantes):")
    print(list(invalid_countries))  # Afficher la liste des pays supprimés
    
    # Filtrer le DataFrame pour ne garder que les pays valides
    filtered_df = df[df['Country'].isin(valid_countries)]
    
    return filtered_df



# II-5- Visualiser les valeurs manquantes 

def visualize_missing_values(df, column_name):
    """
    Visualise les valeurs manquantes pour chaque pays pour une colonne spécifiée.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        column_name (str): Le nom de la colonne dont les valeurs manquantes sont visualisées.
    """

    # Pivot le DataFrame pour avoir 'Country' en colonnes et 'Year' en lignes
    df_pivot = df.pivot_table(index='Year', columns='Country', values=column_name)
    
    # Créer un masque où les valeurs manquantes sont marquées par True
    missing_data = df_pivot.isnull()
    
    # Tracer la heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(missing_data, cmap='Blues', cbar=False, linewidths=0.5, linecolor='gray')
    
    # Ajouter des titres et des étiquettes
    plt.title(f"Visualisation des valeurs manquantes pour {column_name}", fontsize=14)
    plt.xlabel('Pays', fontsize=12)
    plt.ylabel('Année', fontsize=12)
    
    plt.show()





# II-6- Imputer les valeurs manquantes par la médiane
def impute_missing_values_by_median(df, column_name):
    """
    Impute les valeurs manquantes pour chaque pays dans la colonne spécifiée par la médiane.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données avec 'Country', 'Year', et la colonne cible.
        column_name (str): Le nom de la colonne pour laquelle l'imputation par la médiane doit être effectuée.
    
    Returns:
        pd.DataFrame: DataFrame avec les valeurs manquantes imputées.
    """
    # Appliquer l'imputation de la médiane par pays
    def impute_country(country_df):
        median_value = country_df[column_name].median()  # Calcul de la médiane pour la colonne spécifiée
        country_df[column_name] = country_df[column_name].fillna(median_value)  # Remplacer les valeurs manquantes
        return country_df

    # Grouper par pays et appliquer l'imputation
    df_imputed = df.groupby('Country').apply(impute_country)
    
    # Réinitialiser l'index pour aplatir le DataFrame
    df_imputed = df_imputed.reset_index(drop=True)

    return df_imputed



# II-7- Analyse des séries temporelles
def analyse_serie_temporelle(data, indicateur, pays):
    df = data.copy()
    # Utilisation de la fonction pivot pour remodeler le dataframe
    df = df.pivot(index=['Year'], columns='Country_Code', values=indicateur)
    df['Year'] = pd.to_datetime(df.index)

    # Sélectionner la série temporelle du pays spécifique
    serie_temporelle = df[pays].dropna()

    # Moyenne mobile d'ordre 6
    rolling_mean = serie_temporelle.rolling(window=6).mean()

    # Visualisation de la série temporelle et de la moyenne mobile dans le même plot
    plt.figure(figsize=(10, 6))
    plt.plot(serie_temporelle, label=f'Série Temporelle - {indicateur} en {pays}', color='blue')
    plt.plot(rolling_mean, label='Moyenne Mobile', color='red')
    plt.title(f'Série Temporelle et Moyenne Mobile - {indicateur} en {pays}')
    plt.legend()
    plt.show()

    # Décomposition saisonnière
    decomposition = seasonal_decompose(serie_temporelle, model='multiplicative', period=4)


    # Visualisation des composants décomposés
    plt.figure(figsize=(12, 8))

    # Série Temporelle
    plt.subplot(4, 1, 1)
    plt.plot(serie_temporelle)
    plt.title(f'Série Temporelle - {indicateur} en {pays}')

    # Tendance
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend)
    plt.title('Tendance')

    # Saisonnalité
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal)
    plt.title('Saisonnalité')

    # Résidus
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid)
    plt.title('Résidus')

    plt.tight_layout()
    plt.show()

    # Test de stationnarité (Augmented Dickey-Fuller)
    result = adfuller(serie_temporelle)
    print(f'Test de Dickey-Fuller Augmenté:\nStatistique de test = {result[0]}\nValeur critique (5%) = {result[4]["5%"]}')




## II-8- Calcul du taux de croissance moyen par pays

import pandas as pd


def calculer_taux_croissance_moyen(df, year_col, country_col, population_col):
    """
    Calcule le taux moyen de croissance de la population pour chaque pays entre 1990 et 2023.
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        year_col (str): Nom de la colonne représentant les années.
        country_col (str): Nom de la colonne représentant les pays.
        population_col (str): Nom de la colonne représentant la population en âge de travailler.
    Returns:
        pd.DataFrame: Un nouveau DataFrame avec le pays et son taux de croissance moyen (n).
    """
    # Trier les données par pays et année
    df = df.sort_values(by=[country_col, year_col]).copy()
    
    # Calculer le taux moyen de croissance par pays
    taux_croissance_moyen = df.groupby(country_col).apply(
        lambda group: (group[population_col].iloc[-1] / group[population_col].iloc[0]) ** (1 / (group[year_col].iloc[-1] - group[year_col].iloc[0])) - 1
    ).reset_index(name='Taux_croissance_moyen')
    
    return taux_croissance_moyen



## II-9- Calcul du taux d'épargne moyen par pays 
def calculer_taux_epargne_moyen(df, year_col, country_col, capital_travailleur, pib_travailleur):
    """
    Calcule le taux d'épargne moyen (s) pour chaque pays entre 1990 et 2023.
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        year_col (str): Nom de la colonne représentant les années.
        country_col (str): Nom de la colonne représentant les pays.
        gfcf_col (str): Nom de la colonne représentant la formation brute de capital fixe (GFCF).
        pib_col (str): Nom de la colonne représentant le PIB total.
    Returns:
        pd.DataFrame: Un nouveau DataFrame avec le pays et son taux d'épargne moyen (s).
    """
    # Calculer le taux d'épargne pour chaque ligne
    df = df.copy()
    df['Taux_epargne'] = df[capital_travailleur] / df[pib_travailleur]
    
    # Calculer la moyenne du taux d'épargne par pays
    taux_epargne_moyen = df.groupby(country_col)['Taux_epargne'].mean().reset_index()
    taux_epargne_moyen.rename(columns={'Taux_epargne': 'Taux_epargne_moyen'}, inplace=True)
    
    return taux_epargne_moyen


## II-10 Former échantillons des organisations économiques OCDE, UA
def former_echantillons(df, country_col, liste1, liste2):
    """
    Sépare le jeu de données en deux échantillons : un pour les pays de l'OCDE et un pour les pays de l'Union Africaine.
    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        country_col (str): Nom de la colonne représentant les pays.
        liste_ocde (list): Liste des pays appartenant à l'OCDE (codes ISO ou noms).
        liste_ua (list): Liste des pays appartenant à l'Union Africaine (codes ISO ou noms).
    Returns:
        tuple: Deux DataFrames, l'un pour les pays de l'OCDE et l'autre pour les pays de l'UA.
    """
    # Filtrer les données pour les pays de l'OCDE
    df1 = df[df[country_col].isin(liste1)].copy()
    
    # Filtrer les données pour les pays de l'Union Africaine
    df2 = df[df[country_col].isin(liste2)].copy()
    
    return df1, df2


## II-11 Regression

def perform_regression(data, dependent_var, independent_vars):
    """
    Effectue une régression linéaire OLS sur les données spécifiées.

    Arguments :
    - data : DataFrame contenant les données
    - dependent_var : Nom de la colonne pour la variable dépendante (y)
    - independent_vars : Liste des noms des colonnes pour les variables indépendantes (X)

    Retourne :
    - Un résumé du modèle de régression
    """
    # Extraire la variable dépendante (y) et les variables indépendantes (X)
    y = data[[dependent_var]]
    X = data[independent_vars]

    # Ajouter une constante pour inclure l'intercept dans le modèle
    X = sm.add_constant(X)

    # Effectuer la régression linéaire
    model = sm.OLS(y, X).fit()

    # Afficher le résumé des résultats
    return model.summary()

## II-12 Matrice de variance covariances

def plot_correlation_matrix(data):
    """
    Génère une heatmap des corrélations pour les variables spécifiées.
    
    Arguments :
    - data : DataFrame contenant les données
    - pib_col : Nom de la colonne pour le PIB par tête
    - epargne_col : Nom de la colonne pour le taux d'épargne
    - croissance_col : Nom de la colonne pour le taux de croissance
    - g_sigma : Constante ajoutée à la croissance (par défaut 0.05)
    
    Retourne :
    - La matrice de corrélation
    """

    # Calculer la matrice de corrélation
    correlation_matrix = data[['log_PIB_par_tete', 'log_s', 'log_n_g_sigma']].corr()
    
    # Visualiser la heatmap des corrélations
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Heatmap des corrélations")
    plt.show()
    
    return correlation_matrix



## II-13 Fonction pour afficher les résultats du K-Means
def afficher_resultats_kmeans(pipeline, donnees, donnees_originales):
    """
    Affichage des résultats du K-Means et assignation des étiquettes de clusters aux données originales.

    Args :
        pipeline : Le pipeline contenant K-Means.
        donnees : Les données prétraitées (uniquement les caractéristiques numériques).
        donnees_originales : Le DataFrame original contenant la colonne 'Country'.
    """
    t0 = time()
    pipeline.fit(donnees)
    temps_execution = time() - t0

    resultats = [pipeline['clusterer'].init, temps_execution, pipeline['clusterer'].n_iter_, pipeline['clusterer'].inertia_]

    donnees_kmeans = pipeline['preprocessor'].transform(donnees)
    etiquettes_predites_kmeans = pipeline['clusterer'].labels_

    resultats += [silhouette_score(donnees_kmeans, etiquettes_predites_kmeans, metric='euclidean')]
    print('init\t\ttemps\tnb_iter\tinertie\tsilhouette')
    print("{:9s}\t{:.3f}s\t{}\t{:.0f}\t{:.3f}".format(*resultats))

    # Assigne les étiquettes de clusters au DataFrame original
    donnees_originales['Cluster'] = etiquettes_predites_kmeans
    print("\nAssignations de clusters :\n", donnees_originales[['Country', 'Cluster']])




## II-14 Fonction pour analyser les silhouettes

def silouhette_analysis(pipeline, data, k_range):
    """
    Analyse de silhouette pour évaluer la qualité du clustering pour différents nombres de clusters.
    
    Args :
        pipeline : Le pipeline contenant le modèle de clustering.
        data : Les données à analyser.
        k_range : Une plage de valeurs pour le nombre de clusters à tester (tuple ou liste avec deux valeurs).
    
    Returns :
        scores_dict : Un dictionnaire contenant les scores de silhouette pour chaque nombre de clusters.
    """

    scores_dict = {}  # Initialise un dictionnaire pour stocker les scores de silhouette pour chaque nombre de clusters.
    
    # Boucle sur la plage de valeurs de clusters spécifiée.
    for i in range(k_range[0], k_range[1]):
        pipeline['clusterer'].n_clusters = i  # Modifie le nombre de clusters du modèle.
        (fig, ax) = plt.subplots()  # Crée une nouvelle figure et des axes pour la visualisation.
        
        # Crée un visualiseur de silhouette en utilisant le modèle et les données transformées.
        visu = silhouette_visualizer(
            clone(pipeline['clusterer']),
            pipeline['preprocessor'].fit_transform(data),
            colors='yellowbrick',
            ax=ax
        )
        
        # Enregistre le score de silhouette moyen pour le nombre actuel de clusters dans le dictionnaire.
        scores_dict[i] = visu.silhouette_score_  
    
    # Affiche les scores de silhouette pour chaque nombre de clusters testé.
    for item in scores_dict:
        print('{} clusters - Average silhouette score : {}'.format(item, scores_dict[item]))
    
    return scores_dict  # Retourne le dictionnaire des scores de silhouette.




## II-15 Fonction pour la visualisation 2D des Clusters
def tsne_visualization(data, predicted_labels, tsne_params):
    """
    Visualisation 2D des clusters en utilisant l'algorithme t-SNE.
    
    Args :
        data : Les données à projeter en 2D.
        predicted_labels : Les étiquettes des clusters prédites.
        tsne_params : Les paramètres pour l'algorithme t-SNE (sous forme de dictionnaire).
    """
    
    # Applique l'algorithme t-SNE sur les données avec les paramètres spécifiés.
    tsne = manifold.TSNE(**tsne_params)
    tsne_result = tsne.fit_transform(data)
    
    # Crée un DataFrame contenant les résultats de la projection t-SNE et les étiquettes des clusters.
    df_tsne = pd.DataFrame(tsne_result, columns=['tsne_1', 'tsne_2'])
    df_tsne['predicted_labels'] = predicted_labels
    
    # Initialise une figure pour la visualisation.
    plt.figure(figsize=(15, 12))
    
    # Trace un scatterplot avec les clusters colorés en fonction des étiquettes prédites.
    sns.scatterplot(
        x='tsne_1', y='tsne_2',  # Colonnes t-SNE pour les axes.
        hue='predicted_labels',  # Variable pour la couleur des points (clusters).
        palette=sns.color_palette("hls", df_tsne['predicted_labels'].nunique()),  # Palette de couleurs adaptée au nombre de clusters.
        data=df_tsne,  # Source des données.
        legend="full",  # Affiche la légende complète.
        alpha=0.4  # Définit la transparence des points pour une meilleure visibilité.
    )

## II-16 Fonction pour l'interprétation des clusters

def clustering_interpretations(df, min_max=(0, 100)):
    """
    Interprétation des clusters en utilisant les moyennes des variables numériques et une échelle normalisée.
    
    Args :
        df : Le DataFrame contenant les données avec les étiquettes de clusters.
        min_max : La plage de normalisation pour les valeurs des moyennes (tuple).
    
    Returns :
        Un DataFrame contenant les scores normalisés pour chaque cluster.
    """
    
    # Calcule la moyenne des variables numériques pour chaque cluster.
    grouped_df = df.select_dtypes(include=['number']).groupby('predicted_label').mean()
    
    # Applique une normalisation MinMax aux moyennes des clusters.
    scaler = MinMaxScaler(min_max)
    scaled_df = pd.DataFrame(scaler.fit_transform(grouped_df), columns=grouped_df.columns)
    
    scores = []  # Initialise une liste pour stocker les scores normalisés pour chaque cluster.
    
    # Boucle sur chaque cluster pour afficher ses scores et visualisations.
    for cluster, row in scaled_df.iterrows():
        print(row)  # Affiche les scores pour le cluster actuel.
        scores.append(row)  # Ajoute les scores à la liste.
        
        # Crée une visualisation en radar pour les variables du cluster actuel.
        fig = px.line_polar(
            row, 
            r=row.values,  # Valeurs des variables (rayon).
            theta=row.index,  # Noms des variables (angle).
            line_close=True,  # Ferme le polygone formé par les points.
            title='Cluster {}'.format(cluster)  # Titre de la figure avec le numéro du cluster.
        )
        fig.update_traces(fill='toself')  # Remplit l'intérieur du polygone.
        fig.show()  # Affiche la figure.
    
    # Retourne un DataFrame avec les scores normalisés pour tous les clusters.
    return pd.DataFrame(scores)
