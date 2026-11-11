# Notebook 06 — Baselines de développement

Exécuter les notebooks 04 puis 05 avant `06_baseline_modeling.ipynb`, avec les dépendances
notebook et projet installées. Le notebook 06 vérifie que les partitions exportées correspondent
au découpage recalculé et bloque si un contrôle technique de fuite échoue.

Deux pipelines sont comparés : probabilité constante apprise sur train et régression logistique
L2 (`C=1`, `max_iter=2000`). Imputation médiane et standardisation des numériques, imputation
par mode et one-hot des catégories sont ajustées dans chaque train, y compris dans les backtests.
Les catégories inconnues sont acceptées ; aucune statistique n'est apprise sur validation ou test.

Le choix du candidat utilise l'average precision sur validation. Le seuil maximise le F1 sur
validation, avec priorité au seuil le plus élevé en cas d'égalité. Ce critère est descriptif et
ne remplace pas une fonction de coûts métier. Le pipeline reste ajusté uniquement sur train.
Le test de développement reçoit le candidat et son seuil figés, sans nouveau réglage.

Les rapports incluent AP, ROC AUC, Brier, log-loss, précision, rappel, F1, matrice de confusion,
volume signalé, backtests temporels, calibration descriptive et métriques mensuelles du test.
Les résultats portent sur des snapshots corrélés, sans intervalle supposant des lignes indépendantes.

Sorties : `reports/baseline/` pour les métriques, graphiques HTML et prédictions auditables ;
`models/baseline/pipeline.joblib` et `manifest.json` pour le pipeline et sa configuration.
Le manifeste reprend celui du split, les features, le seuil et les versions. `training_validated`
reste faux. Ne charger un joblib que depuis une source de confiance.

L'horizon des labels, les sources des features et la maturité des cibles restent à confirmer.
La période test 2024 a déjà été explorée : les scores ne constituent pas une validation finale
aveugle ni une preuve de rentabilité ou d'effet causal d'une campagne.
