 # I- Importations de mmodules

## I-1 Modules pour l'importation des données api
import requests


## I-2 Modules pour l'analyse de données et manipulations
import pandas as pd








# II- Déclarations de fonctions

## II-1- Fonctions pour extraire les noms des pays du monde

def get_worldbank_country_codes():
    """
    Récupère les codes ISO alpha-3 des pays dans la base de données de la Banque mondiale,
    en excluant les régions et unions économiques.
    
    Returns:
        list: Chaine de caractère des codes ISO alpha-3 des pays uniquement.
    """
    url = "http://api.worldbank.org/v2/country?format=json&per_page=300"
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



## II-2- Fonctions pour importer les données des pays

def fetch_worldbank_data(countries, indicator):
    """
    Télécharge les données de la Banque mondiale avec l'API pour un indicateur et une liste de pays.
    
    Args:
        countries (str): Liste de codes ISO des pays séparés par ';' (ex: 'USA;FRA;DEU').
        indicator (dict): Indicateur recherché.
        
    Returns:
        pd.DataFrame: Base de données avec l'indicateur et les codes ISO des pays.
    """

    # URL de l'API World Bank avec les codes de pays et l'indicateur
    url = f"http://api.worldbank.org/v2/country/{countries}/indicator/{indicator}?format=json"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Erreur lors de la requête API : {response.status_code}")
    
    data = response.json()
    if len(data) < 2 or not data[1]:
        print(f"Aucune donnée disponible pour {indicator}")
    else:
        print("Téléchargement de données réussi")
    
    # Extraire et normaliser les données
    records = data[1]
    df = pd.json_normalize(records)  # Aplatir les données imbriquées

    # Vérifier si les données contiennent des informations sur les pays et extraire les codes ISO
    if 'country.id' in df.columns:
        df['Country_Code'] = df['country.id']  # Ajouter la colonne avec les codes ISO des pays
    
    # Sélectionner les colonnes pertinentes et les renommer
    df = df[['country.value', 'Country_Code', 'date', 'value']].rename(columns={
        'country.value': 'Country',
        'date': 'Year',
        'value': indicator  # Utilise le nom de l'indicateur comme colonne
    })
    
    return df
