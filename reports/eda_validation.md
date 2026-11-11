# Vérification de l'approfondissement EDA

Le notebook `03_exploratory_data_analysis.ipynb` conserve les 19 questions initiales et ajoute
six axes conditionnels, géographiques et temporels, pour un total de 25. L'audit critique est
consigné dans `docs/eda_audit.md` et la synthèse scientifique A–F termine le notebook.

## Contrôles exécutés

- `python -m pytest -q` dans `.venv` : **55 tests réussis**.
- Ruff : vérification du code et du formatage réussie.
- Notebook validé avec nbformat, puis toutes les cellules exécutées dans l'ordre avec le Python du projet.
- **26 figures Plotly générées et sérialisées** avec validation du schéma ; ce contrôle ne remplace
  pas l'inspection visuelle interactive dans Jupyter/VS Code.
- Recalculs indépendants sur les colonnes brutes pour support × paiements, tarif × usage et âge × région :
  lignes, clients distincts, snapshots positifs, clients positifs et taux conformes.
- Bornes des IC dans [0, 100], ordre des bornes et absence d'IC pour les petits groupes vérifiés.
- Contrôle de couverture : axes 1 à 25, et présence des six parties obligatoires de synthèse.
- Empreintes des sources et des modules existants de features/modélisation inchangées.
- Source en mémoire identique à une nouvelle lecture du CSV ; aucune vue pilote chargée.
- Aucun pipeline de features, preprocessing d'entraînement, feature selection, modèle ou tuning créé.

## Dénominateurs de référence de cette exécution

| Vue | Lignes à cible connue | Snapshots positifs |
|---|---:|---:|
| Source principale / support × retards | 38 689 | 1 164 |
| Abonnement × usage éligible | 36 679 | 1 100 |
| Satisfaction × support, NPS manquant conservé | 38 689 | 1 164 |
| Géographie et âges dans le domaine proposé | 36 197 | 1 079 |

La vue géographique exclut 2 492 lignes ; la vue tarif/usage en exclut 2 010. Les comptes de clients
peuvent se recouvrir entre groupes. Les textes chiffrés constituent une photographie de ces fichiers
et doivent être revus après une modification des sources ou du YAML.

## Provenance des CSV

| Fichier | SHA-256 après vérification, identique à l'empreinte avant modification |
|---|---|
| `churn_clients.csv` | `a5b5ae8f69e22aa4506df0a427fe5d3810872c3c8a6331eda6299576bceb6732` |
| `pilote_campagne_2023.csv` | `f5e6abdcbf6cfd6555a119cfa973d53778358dd696fd324943e74699410bc8eb` |

## Limites non résolues par les tests

L'horizon de `churn`, la règle de sortie du panel et les dates de disponibilité des variables ne sont
pas établis dans les documents métier du dépôt, qui sont vides. Les IC exploratoires par bootstrap
supposent des clients approximativement indépendants ; ils ne corrigent ni sélection ni comparaisons
multiples et ne prouvent pas un effet causal ou une interaction. Les priorités FE sont des hypothèses
d'expérimentation, pas des gains prédictifs mesurés.
