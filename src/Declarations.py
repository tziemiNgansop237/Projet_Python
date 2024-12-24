 # I- Importations de mmodules

## I-1 Modules pour l'importation des données api
import requests


## I-2 Modules pour l'analyse de données et manipulations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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