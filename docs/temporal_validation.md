# Notebook 05 — Validation temporelle, fuite et drift

## Question évaluée

Le scénario principal mesure une future prédiction sur la continuation du panel. Les snapshots sont
séparés dans le temps ; certains clients sont communs, l'identifiant n'est jamais une feature. Le mode
`disjoint` traite une autre question : clients non vus dans les partitions précédentes de l'échantillon.
Ce n'est pas nécessairement une cohorte de clients nouvellement acquis par l'entreprise.

Les dates sont fixées dans `configs/temporal_validation.yaml`, sans recherche de découpage minimisant
le drift ou égalisant les taux de cible. Aucun modèle, imputation, scaler ou encodage appris n'est ajusté ici.

## Découpage principal et purge

| Partition | Fenêtre configurée | Lignes conservées | Clients | Positifs | Taux |
|---|---|---:|---:|---:|---:|
| Train | Janvier–juin 2024 | 20 103 | 3 762 | 590 | 2,93 % |
| Validation | Août–septembre 2024 | 6 309 | 3 239 | 209 | 3,31 % |
| Test de développement | Novembre–décembre 2024 | 5 910 | 3 003 | 165 | 2,79 % |

60 copies équivalentes de snapshots après transformation sont retirées et 6 307 lignes se trouvent
hors fenêtres, en juillet et octobre. Le journal rattache chaque ligne écartée à un motif ;
partitions + exclusions réconcilient les 38 689 lignes en entrée.

Les égalités client/date avec des valeurs de features ou de cible différentes entraînent une erreur.
Les équivalences portent sur le dataset transformé : elles ne prétendent pas prouver l'identité de toutes
les colonnes exclues de la source brute.

Une ligne est gardée avant une partition suivante seulement si :

`date_snapshot + horizon_label + délai_disponibilité_label + embargo < début_fenêtre_suivante`.

L'égalité à la frontière est purgée. L'horizon par défaut de 30 jours n'est qu'une hypothèse, explicitement
`label_horizon_confirmed: false`. Aucun horizon réel n'est déduit du nom du fichier pilote.
Les intervalles choisis satisfont cette hypothèse sans purge supplémentaire ; un autre horizon peut
modifier les effectifs. La disponibilité des labels du test n'est pas datée (`label_observed_through: null`).

## Contrôles de data leakage

- Features prises dans le manifeste, jamais déduites par simple suppression de la cible.
- Exclusion de cible, identifiant, date brute, âge brut, facture et refus d'offre dans X.
  `age_group` reste une feature candidate explicitement autorisée ; ce n'est pas une absence d'information d'âge.
- Contrôle des clés communes, ordre temporel, purge, conflits de snapshots et identifiants partagés.
- Recherche sur train d'une égalité numérique exacte à la cible ou son inverse : signal à auditer, pas preuve exhaustive.
- Audit des déclarations de disponibilité, horizon, maturité des labels et période déjà consultée dans l'EDA.

`PASS` désigne un contrôle technique satisfait sous les hypothèses déclarées. `FAIL` bloque l'export.
`INFO` décrit un chevauchement attendu, `WARN` une limite connue, `UNVERIFIED` une preuve manquante.
Les tests unitaires vérifient le code ; le rapport du notebook porte sur les données effectivement partitionnées.

Aucune clé client/date n'est commune entre partitions. En mode principal, 3 062 clients sont communs
au train/validation et 2 812 au train/test. Cela n'est pas automatiquement une fuite pour la continuation
d'un panel, mais demande des sources réellement connues au snapshot et un protocole adapté à l'usage.

L'horizon réel, les sources des 17 features et la maturité des labels test ne sont pas confirmés. Toute
l'année 2024 a servi à l'EDA : novembre/décembre ne constituent pas un test final aveugle. L'absence de
fuite métier ne peut pas être certifiée à partir d'un bon ordre des dates.

## Sensibilité sans clients communs

À dates constantes, le mode disjoint conserve 281 lignes / 177 clients en validation et 101 lignes /
53 clients en test, dont 30 et 10 positifs. Les taux de 10,68 % et 9,90 % montrent un changement de
population par rapport au scénario principal. Les groupes sont trop limités pour traiter cette variante
comme une évaluation équivalente simplement « plus propre ».

## Backtests internes

Les deux folds restent dans le train principal : train jusqu'en février puis validation en avril ;
train jusqu'en avril puis validation en juin. La purge est réappliquée. Le test et la validation principale
ne servent pas aux backtests. Les futurs prétraitements appris devront être refittés dans chaque train de fold,
pas sur toutes les partitions ni une fois sur tout le train avant cross-validation.

## Drift

`DataDriftMonitor.fit(train_features)` définit les bornes quantiles et niveaux de référence sur train.
`compare(validation_features)` et `compare(test_features)` ne modifient pas cette référence.
Les variables discrètes, constantes et totalement manquantes ont des traitements explicites ; un marqueur
technique de manquant ne peut pas se confondre avec une vraie chaîne identique.

- PSI : comparaison des probabilités, avec lissage défini dans le YAML.
- Variation totale : distance de distribution, notamment pour les catégories/discrètes.
- Écart de manquants en points ; modalités nouvelles parmi les valeurs renseignées.
- Proportion numérique hors plage train parmi les valeurs finies, en complément des bins ouverts.

Les seuils sont des conventions heuristiques, pas des p-values. Les lignes répétées ne sont pas supposées
indépendantes pour un test statistique. `NO_ALERT` ne prouve pas l'absence de dérive, notamment multivariée,
conditionnelle ou de performance. Le moniteur ne choisit pas les dates du split et ne remplit aucune feature.

Aucune alerte en validation aux seuils actuels. Dans le test, l'ancienneté et son logarithme ont un PSI
≈ 0,129 et une variation totale ≈ 0,126 ; ce sont deux représentations du même signal. Environ 7,20 %
des anciennetés finies du test sont hors plage du train. La progression du temps peut expliquer une partie
du vieillissement du panel ; elle doit être examinée, pas cachée en remélangeant les périodes.

## Sorties et limites d'usage

`data/processed/temporal_splits/` contient les trois partitions, le journal des exclusions, les rapports de
fuite et de drift, ainsi que `split_manifest.json`. `source_row`, les identifiants, dates et cible restent
hors de la liste `feature_columns`. Le manifeste conserve la configuration et les points non confirmés.
`training_validated` reste faux : il s'agit d'un scénario de développement, pas d'une certification production.
Les fichiers sources et les notebooks précédents restent inchangés.

Avant une évaluation finale : confirmer horizon, délai des labels, disponibilité as-of des sources et règle de
sortie du panel ; prévoir une période non consultée si possible. Un drift mesuré peut être réel et acceptable,
mais son effet sur les scores et les décisions métier devra être évalué pendant la modélisation.
