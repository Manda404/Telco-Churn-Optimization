# Audit critique du notebook 03 — dataset principal

## Périmètre et diagnostic

Audit du notebook réel, des composants `Settings`, `TelcoDataLoader`, `DataQualityReport`,
des règles YAML et des documents du dépôt. Les documents métier présents sont vides :
l'horizon du label et les dates de disponibilité des variables ne sont pas établis.
Aucune analyse du pilote, modification des CSV, construction de features de production ou modélisation.

Les résultats bruts restent consultables. Les corrections de casse/espaces et les regroupements
ajoutés sont temporaires, préfixés `eda_`, et audités. Aucune correction de guillemets n'est appliquée :
le `repr` utilisé dans certains anciens graphiques ne prouve pas la présence de guillemets dans la source.

## Décision sur chacun des 19 axes existants

Les colonnes ci-dessous couvrent pertinence métier/statistique, qualité des KPI et des dénominateurs,
visuels, limites, hypothèses et contribution au futur FE. Aucun axe n'est supprimé ; les représentations
complémentaires d'une même question restent fusionnées sous un axe (heatmap/courbes, histogramme/boxplot).

| Axe | Décision | Intérêt et KPI / dénominateur | Diagnostic critique et amélioration | Hypothèse / piste FE future |
|---|---|---|---|---|
| 1 Population | Conserver | Lignes, identifiants, répétitions : unité essentielle. | Afficher toujours clients et lignes ; clients présents plusieurs fois. | Protocole temporel et gestion des identifiants, pas feature ID. |
| 2 Prévalence | Conserver | Positifs / cibles connues, pas / clients. | Classe rare ; le taux n'est pas une incidence annuelle. | Évaluation adaptée à la classe positive ultérieurement. |
| 3 Doublons | Améliorer | Sensibilité brute vs copies exactes retirées. | 60 copies après première occurrence ; 120 lignes dans des clés répétées. Conserver la distinction. | Politique explicite de doublons, non déduite de leur faible impact global. |
| 4 Temps | Améliorer | Effectifs et taux par date. | Un an ne démontre pas une saisonnalité ; compléter par l'axe 25 et la composition du panel. | Validation hors temps, horizon et censure à clarifier. |
| 5 Contrat | Améliorer | Taux conditionnels par libellé brut. | 1 189 formats non canoniques ; sensibilité sur une copie normalisée. | Encodage de catégories validées. |
| 6 Région | Conserver / approfondir | Taux et taille des populations. | Confusion par âge, contrat, ancienneté possible ; axes 23–24. | Variable contextuelle candidate, pas d'effet géographique causal établi. |
| 7 Canal | Améliorer | Taux et support par libellé. | 1 139 formats non canoniques ; ne pas classer les variantes rares comme effets de canal. | Normalisation à confirmer. |
| 8 Facturation | Améliorer | Quantiles par cible, nombres finis disponibles. | Moyennes sensibles aux montants invalides ; facture potentiellement postérieure au label. | Facture bloquée jusqu'à preuve de disponibilité. |
| 9 Usage | Améliorer | Quantiles, extrêmes, effectifs. | Valeurs jusqu'à 32 154 ; unité inconnue. Axe 21 pour le contexte tarifaire. | Transformation robuste à tester, pas suppression automatique. |
| 10 Corrélations | Conserver | Pearson et effectifs par paire. | Dépendance et non-linéarités ; aucune sélection de features fondée sur r seul. | Redondance des montants et relations conditionnelles à étudier. |
| 11 Ancienneté | Conserver | Taux par durée approchée ; qualité séparée. | 173 durées négatives ; biais de survie possible. | Durée connue au snapshot, après validation des dates. |
| 12 Support | Approfondir | Taux, événements et clients distincts par niveau. | Fenêtre 3 mois non garantie par le seul nom ; axe 20 et axe 22. | Non-linéarité et interactions candidates. |
| 13 Paiements | Approfondir | Taux et couverture des positifs. | Fenêtre inconnue ; forte proportion locale ≠ couverture globale. | Fenêtres de mesure et interactions à confirmer. |
| 14 Satisfaction | Approfondir | Notes disponibles, taux par niveau. | 67,84 % manquants ; population répondante sélectionnée. Axe 22 conserve un groupe manquant. | Imputation/indicateur à tester après audit de collecte. |
| 15 Contrat × ancienneté | Améliorer | Taux, volumes, clients positifs, IC. | Conserver version brute ; ajouter version normalisée et seuil sur clients distincts. | Interaction éventuelle, stabilité et disponibilité préalables. |
| 16 Manquants | Conserver | Taux selon disponibilité, pas un nouvel inventaire de nulls. | Les écarts ne démontrent pas MCAR/MAR/MNAR ni un effet causal. | Indicateur seulement après confirmation de disponibilité. |
| 17 Distribution âge | Conserver | Histogramme/KDE et boxplot fusionnés sous une question. | Domaine proposé vs toutes valeurs ; pas de normalité postulée. | Contrôles qualité, pas ciblage par âge. |
| 18 Âge × cible | Améliorer | Taux, volume positif et clients distincts séparés. | ≥ 80 : 120 lignes mais 12 clients ; nouvelle lecture régionale et masquage renforcé. | Diagnostic descriptif, pas règle de ciblage ni verdict d'équité. |
| 19 Trajectoires | Conserver | Dates positives distinctes et suivi post-positif. | Aucun suivi après premier positif ; ne pas inférer la définition du label de ce seul constat. | Clarification du label et du panel avant tout découpage. |

## Plan implémenté : six ajouts, total 25 axes

20. Support × retards : co-occurrence et volume, sans score de friction définitif.
21. Abonnement × consommation : contexte tarifaire et usage, sans ratio financier construit.
22. Satisfaction × support : réponses disponibles et strates NPS manquant.
23. Âge × région, composition : parts de snapshots et effectifs clients ; pas de taux de churn.
24. Âge × région, résultat : taux, snapshots positifs, clients positifs et incertitude.
25. Stabilité : profils par date et standardisation descriptive sur une composition contractuelle fixe.

## Dénominateurs et précision

- `n_observed` : lignes du groupe à cible connue ; `n_events` : lignes positives.
- `n_customers` : identifiants distincts du groupe ; `n_positive_customers` : identifiants avec au moins
  une ligne positive dans ce groupe. Les comptes clients ne s'additionnent pas entre groupes mobiles.
- Les parts géographiques ont deux dénominateurs explicites : tous les snapshots d'une tranche d'âge,
  ou les snapshots éligibles d'une région. Elles ne décrivent pas la part de clients uniques.
- Les vues restreintes affichent leurs exclusions ; les ratios descriptifs se réfèrent à la vue éligible,
  pas systématiquement à la source entière. Aucun ratio n'est présenté comme un effet causal.
- Petits groupes : moins de 100 lignes **ou** moins de 30 clients distincts pour les croisements approfondis.
  Les taux sont masqués, mais les compteurs restent consultables dans le menu des heatmaps.
- IC exploratoires à 95 %, bootstrap percentile avec 300 tirages de clients et de leurs historiques
  entiers, graine 42. Les IC sont omis si le support est insuffisant (dont moins de 5 clients positifs).
  Pas de borne artificielle [0, 0] pour une cellule sans positif.
- Ces IC supposent des clients approximativement indépendants et un panel pertinent. Ils ne corrigent
  ni la sélection, ni les comparaisons multiples, ni l'incertitude métier ; leur chevauchement n'est pas
  un test d'interaction. Le notebook n'affirme aucune significativité ajustée.
- La standardisation utilise les poids contractuels de la période EDA, documentés. Ce n'est ni une
  feature, ni un ajustement causal, ni un prétraitement à appliquer au train/test. Une strate non
  représentée ou trop petite rend le taux standardisé du mois non disponible.

## Vigilance temporelle / questions sans preuve disponible

| Variable | Question bloquante | Décision actuelle |
|---|---|---|
| `churn` | Événement, horizon, date de calcul, sortie du panel ? | Taux descriptifs par snapshot uniquement. |
| `retention_offer_declined` | Offre et réponse avant ou après le moment de prédiction ? | Exclue des relations explicatives du notebook. |
| `last_invoice_amount` | Facture complète, prorata, facture de clôture ? Date réelle de disponibilité ? | Analyse de sensibilité, pas recommandation de feature immédiate. |
| `nps` | Date et déclencheur de collecte ? Accessible avant le label ? | EDA répondants/non-répondants, pas disponibilité présumée. |
| `support_calls_3m` | Fenêtre strictement antérieure au snapshot ? | Nom de colonne insuffisant comme preuve. |
| `late_payments` | Fenêtre, cumul, date d'observation des incidents ? | Chronologie à documenter. |

Aucune règle métier absente n'est inventée. La normalisation validée pour une copie EDA devra être
réexaminée séparément lors de la future phase FE. Les sorties du notebook et leurs textes constituent
une photographie des fichiers actuels et doivent être revus lorsque les sources changent.
