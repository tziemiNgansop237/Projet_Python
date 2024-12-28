 # I- Importations de mmodules

## I-1 Modules pour l'importation des données api
import requests
import openpyxl


## I-2 Modules pour l'analyse de données et manipulations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile


## I-3 Modules pour la visualisation des données
import geopandas as gpd
import dash
from dash import dcc, html, dash_table, Dash
from dash.dependencies import Input, Output
import plotly.express as px

## I-4. Imports liés à la modélisation et à la statistique
import statsmodels.api as sm
import statsmodels.tsa.filters.hp_filter as smf
import statsmodels.tsa.ardl as sma
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

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

def calculer_taux_croissance_moyen_par_pays(df, year_col, country_col, population_col):
    """
    Calcule le taux de croissance démographique moyen (n) pour chaque pays entre 1990 et 2023.
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
    
    # Calculer la population décalée (année précédente) pour chaque pays
    df['Population_lag'] = df.groupby(country_col)[population_col].shift(1)
    
    # Calculer le taux de croissance démographique pour chaque année
    df['Taux_croissance_annuel'] = (df[population_col] - df['Population_lag']) / df['Population_lag']
    
    # Supprimer les lignes où le taux ne peut pas être calculé (première année pour chaque pays)
    df = df.dropna(subset=['Taux_croissance_annuel']).copy()
    
    # Calculer le taux de croissance moyen par pays
    taux_croissance_moyen = df.groupby(country_col)['Taux_croissance_annuel'].mean().reset_index()
    taux_croissance_moyen.rename(columns={'Taux_croissance_annuel': 'Taux_croissance_moyen'}, inplace=True)
    
    return taux_croissance_moyen


## II-9- Calcul du taux d'épargne moyen par pays 
def calculer_taux_epargne_moyen_par_pays(df, year_col, country_col, capital_travailleur, pib_travailleur):
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
    df['Taux_epargne'] = df[capital_travailleur] / df[pib_travailleur]
    
    # Calculer la moyenne du taux d'épargne par pays
    taux_epargne_moyen = df.groupby(country_col)['Taux_epargne'].mean().reset_index()
    taux_epargne_moyen.rename(columns={'Taux_epargne': 'Taux_epargne_moyen'}, inplace=True)
    
    return taux_epargne_moyen


## II-10  Former échantillons des organisations économiques OCDE, UA
def former_echantillons(df, country_col, liste_ocde, liste_ua):
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
    df_ocde = df[df[country_col].isin(liste_ocde)].copy()
    
    # Filtrer les données pour les pays de l'Union Africaine
    df_ua = df[df[country_col].isin(liste_ua)].copy()
    
    return df_ocde, df_ua
