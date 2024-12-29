# Projet Python pour la data science : Vérification du Modèle de Solow Swan 
 

## Introduction

L'objectif de ce travail est de mettre en pratique les différents élements parcourus en cours sur le language 'Python' et ce par le choix d'une problématique que nous sommes amenés à traiter. La problématique choisie est à dimension économique, il s'agit de vérifier si les PIB d'un ensemble de pays satisfont les conclusions du modèle de Solow. Ce dernier, définit une  convergence conditionnelle  du PIB par tete, postulant que des pays ayants des niveaux de production initialement différents, mais ayants les memes caractéristiques économiques(taux de progrès technique, taux d'épargne, taux de dépréciation...) convergent vers un meme etat stationnaire.

Nous disposons pour cela de bases de données **Open source** fournies par la Banque mondiale, ces dernières comportent les PIB réels , la formation brute de capital fixe(utilisé pour determiner le taux d'épargne) ainsi que d'autres indicateurs socio-économiques pour différents pays du monde.<br> 
   -Les données de la base de données sur le PIB sont annuelles et s'étalent sur plusieurs années, allant de **1990** à **2023**.<br>
   -Les données de la base de données sur la formation brute de capital fixe sont annuelles et s'étalent sur plusieurs années, allant aussi de **1990** à **2023**.<br>
   -Les données de la base de données sur la population(utilisés pour determiner le taux de croissance démographique) sont elles aussi, annuelles et s'étalent sur plusieurs années, allant aussi de **1990** à **2023**.<br>

**Précision :** La base de données sur le taux de croissance du PIB est celle du PIB déflaté (PIB réel), l'effet de l'inflation étant alors éliminé.


## Revue de littérature : 

**Modèle de Solow Swan :** 

Pour mener à bien ce projet, nous nous sommes inspirés des travaux suivants : 

1- Mankiw, N. G., David Romer, et David N. Weil. "A Contribution to the Empirics of Economic Growth." The Quarterly Journal of Economics, vol. 107, no. 2, 1992, pp. 407–437.
Disponible à : https://academic.oup.com/qje/article-abstract/107/2/407/1838296?login=false
2- Solow, R. M. (1956).
"A Contribution to the Theory of Economic Growth."
The Quarterly Journal of Economics, 70(1), 65-94.
Disponible en ligne : https://piketty.pse.ens.fr/les/Solow1956.pdf 

3- Mankiw, N. G., Romer, D., & Weil, D. N. (1992).
"A Contribution to the Empirics of Economic Growth."
The Quarterly Journal of Economics, 107(2), 407-437.
Disponible en ligne : https://eml.berkeley.edu/~dromer/papers/MRW_QJE1992.pdf

4- Barro, R. J., & Sala-i-Martin, X. (2004).
Economic Growth (2ᵉ édition).
The MIT Press.
Aperçu disponible sur Google Books : https://books.google.com/books/about/Economic_Growth_second_edition.html?id=jD3ASoSQJ-AC

## Objectifs

1. Collecter des données sur le taux de croissance démographique, le taux d'épargne, le PIB déflaté ainsi que d'autres indicateurs socio-économiques de plusieurs pays du monde.
2. Parvenir à la construction d'une base de données adaptée aux besoins du projet.
3. Visualiser et analyser le jeu de données résultant.
4. Modéliser pour répondre à la problématique.

## Bases de données

Les données utilisées dans ce projet proviennent des sources suivantes :


- **PIB :** [Global Economic Monitor]( https://databank.worldbank.org/source/world-development-indicators)


**Précision :** 

   -Nous avons recupérés les données grace aux API fournis par la banque mondiale en exploitant l'url suivante : "http://api.worldbank.org/v2/country?format=json&per_page=300"
## Fonctionnalités utilisées

- Jupyter Notebook pour la création de notebook simplifiée et légère.
- Fonctionnalités offertes par Pandas pour la manipulation des données.
- Matplotlib et Seaborn pour la visualisation.
- Numpy pour le calcul scientifique en Python. 
- Geopandas pour faciliter l'utilisation des données géospatiales.
- Plotly particulièrement le module plotly.express.
- Dash pour la création de data apps en Python.
- Statsmodels pour la modélisation.

<u> La liste ci-dessus n'est pas exhaustive. <u>

## Brainstorm

- [Jupyter Notebook](https://docs.jupyter.org/en/latest/) est un environnement interactif de développement et d'exécution de code qui permet de créer et de partager des documents contenant du code, des visualisations et du texte explicatif.
- [Pandas](https://pandas.pydata.org/docs/index.html) est une bibliothèque open source fournissant des structures de données et des outils d'analyse de données hautes performances. Elle offre des structures de données flexibles et performantes, notamment le **DataFrame**, l'objet le plus utilisé de Pandas et qui permet de stocker et de manipuler des données tabulaires de manière efficace. 
- [Matplotlib](https://matplotlib.org/stable/index.html) est une bibliothèque complète permettant de créer des visualisations statiques, animées et interactives.
- [Seaborn](https://seaborn.pydata.org/) est une bibliothèque de visualisation de données Python basée sur matplotlib . Il fournit une interface de haut niveau pour dessiner des graphiques statistiques attrayants et informatifs. Elle facilite la création de graphiques basés sur des données DataFrame de pandas.
- [NumPy](https://numpy.org/doc/) est un package pour le calcul scientifique en Python. 
- [Geopandas](https://geopandas.org/en/stable/) permet des opérations spatiales sur les types géométriques.
- [Plotly](https://plotly.com/python/) crée des graphiques interactifs de qualité publication. 
- [Dash](https://dash.plotly.com/)est un framework web interactif pour la création d'applications web analytiques en Python. C'est une extension de Flask, un framework web pour Python, et utilise également Plotly pour les visualisations interactives.
- [Statsmodels](https://www.statsmodels.org/stable/index.html) est une bibliothèque qui permet l'estimation de différents modèles statistiques, la réalisation de tests statistiques et l'exploration de données statistiques. 


## [Structure du Projet](https://pythonds.linogaliana.fr/content/getting-started/04_python_practice.html)

- `src` : Contient les notebooks Jupyter utilisés, ainsi que :

   -`src/bases`: Emplacement pour stocker les données collectées.

- `LICENSE/` pour protéger la propriété intellectuelle
- `README.md/` pour mieux comprendre la vocation du projet,
- `requirements.txt/` pour contrôler les dépendances du projet. Ce dernier est généré grâce à [pipreqs](https://pypi.org/project/pipreqs/#description).

Le projet est composé de trois notebooks jupyter, soit trois grandes parties : 

1. De prime a bord, le fichier ` Preparation_donnees.ipynb`, montre comment nous avons procedé pour pouvoir recuper les données et les traiter;
2. ensuite dans le fichier `analyse_des_donnees.ipynb` nous faisons l'analyse descriptive et temporelle de nos données;
3. enfin dans le fichier `modélisation.ipynb` nous essayons de trouver des liens entre les données grace a differents modeles afin de monter les points forts du modele de Solow Swan, puis de donner ses limites avant de chuter sur une conlusion générale.
## Comment exécuter le projet ?

1. Clonez ce dépôt sur votre machine locale.

   ```bash
   git clone https://github.com/tziemiNgansop237/Projet_Python.git

&hearts;
