# Telco Churn Optimization

Analyse de données télécom et prédiction du churn avec validation temporelle.
Le code implémenté se trouve dans un seul package : `src/telco_churn/`.

## Installation

Python 3.12 et Poetry sont nécessaires.

```bash
poetry install --with notebook,dev
poetry run python -m ipykernel install --user --name telco-churn --display-name "Python (Telco Churn)"
```

Ouvrir les notebooks avec le kernel **Python (Telco Churn)**. Les chemins des données
et des sorties sont définis dans `configs/base.yaml` et les surcharges de `configs/environments/`.
Le package résout les chemins depuis la racine du projet.

## Parcours des notebooks

Exécuter les notebooks 01 à 06 dans l'ordre : découverte, qualité, analyse exploratoire,
construction des features, validation temporelle, puis baselines de modélisation.
Les notebooks suivants restent des étapes à implémenter.

- [Qualité et exploration](docs/eda_analysis_details.md)
- [Validation temporelle et limites](docs/temporal_validation.md)
- [Baselines et artefacts](docs/baseline_modeling.md)

## Organisation

| Chemin | Rôle |
|---|---|
| `src/telco_churn/data/` | Chargement, validation, qualité et exploration |
| `src/telco_churn/features/` | Construction déterministe et export des features |
| `src/telco_churn/modeling/` | Découpage temporel et pipelines de baseline |
| `src/telco_churn/monitoring/` | Diagnostic du drift des données |
| `src/telco_churn/visualization/` | Figures réutilisables des notebooks |
| `configs/` | Paramètres YAML utilisés par le code |
| `tests/unit/` | Tests du code implémenté |
| `data/` | Données sources et partitions transformées |
| `reports/` | Diagnostics, graphiques et prédictions |
| `models/` | Pipelines sauvegardés et manifestes |

Les scripts, configurations et modules vides du squelette initial ont été retirés.
Les faux fichiers de verrouillage et configurations Docker/CI vides ont également été retirés.
Les extensions futures (campagnes, causalité, calibration, déploiement) seront ajoutées
lors de leur implémentation.

## Vérifications

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
poetry run pytest
```

Avec un environnement `.venv` déjà installé et Ruff disponible : `make check`.
`make format` applique le formatage ; `PYTHON` et `RUFF` peuvent être surchargés.

## Statut des résultats

Les partitions et modèles actuels sont des artefacts de développement. L'horizon de la cible,
la disponibilité temporelle des features et la maturité des labels restent à confirmer.
La période test 2024 a déjà servi à l'exploration ; elle ne constitue pas un test final aveugle.
Les scores de churn ne mesurent pas l'effet causal d'une offre de rétention.
