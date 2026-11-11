# Notebook 04 — Première version des features candidates

## Objectif

Transformer un snapshot brut en une matrice reproductible sans utiliser le label ni apprendre de
statistique sur la population. Le résultat est un prototype documenté, pas une matrice déclarée prête
pour l'entraînement. Les preuves de disponibilité temporelle manquent encore dans les documents métier.

## Contrat

- `FeatureConfig` valide `configs/features.yaml`, dont le chemin vient de `Settings`.
- `FeatureEngineer.transform()` préserve lignes, ordre et index. Il fonctionne sans colonne cible.
- Les règles numériques proviennent de `configs/validation.yaml` ; les bornes métier restent provisoires.
- Identifiant, dates brutes et cible sont conservés dans des métadonnées séparées dans le notebook.
- Facture, refus d'offre, âge brut et `dept` sont exclus des sorties ; `age_group` est autorisée comme candidate dérivée de l'âge.
- Aucune suppression des copies exactes : la future politique d'échantillonnage doit être explicite.

## Sorties par défaut : 17 candidats

| Famille | Colonnes | Règle |
|---|---|---|
| Numériques | `monthly_fee`, `data_gb`, `support_calls_3m`, `late_payments`, `nps` | Conversion numérique ; hors domaine ou non fini → NA. |
| Qualité | `monthly_fee_missing`, `monthly_fee_invalid`, `nps_missing`, `nps_invalid` | Absence brute séparée d'une valeur renseignée invalide. |
| Catégories | `contract`, `region`, `channel` | Espaces/casse vers référentiel ; manquant conservé et inconnu réservé. |
| Âge regroupé | `age_group` | Tranches fixes du YAML ; brut exclu, manquants/hors domaine → NA. |
| Durée | `tenure_months`, `tenure_unavailable` | Différence signup/snapshot ; durée négative ou inutilisable → NA et indicateur. |
| Représentation non linéaire | `log1p_tenure_months` | Logarithme de la durée valide, sans seuil appris. |
| Interaction candidate | `support_payment_interaction` | Produit des compteurs valides ; toute entrée inconnue propage NA. |

Le produit des compteurs est un prototype de représentation de H02, pas le taux ni les bandes de
l'EDA. Son apport incrémental reste à comparer aux variables séparées. La présence d'un candidat
ne constitue pas une feature selection ni une preuve de gain prédictif.

`log_usage` est désactivé : l'unité et les extrêmes de consommation restent à confirmer. Aucun ratio
financier, règle de campagne ou multiplication systématique des interactions n'est ajouté.

## Disponibilité et fuite

Chaque feature a un catalogue : sources, formule, hypothèse EDA et disponibilités déclarées. Par défaut,
toutes les sources sont `unconfirmed`, car l'EDA n'a fourni aucune preuve as-of.
`assert_temporal_availability()` refuse les confirmations manquantes. Même une déclaration complète
ne prouve ni l'horizon du label, ni la validité du découpage, ni l'existence d'une vue historique correcte.

Une invariance à la permutation du label démontre que le transformateur n'utilise pas ce label ; elle
ne démontre pas que les données sources étaient disponibles avant l'événement. Les deux contrôles
sont séparés. Les features de facture et d'offre restent suspendues.

Le fichier 2024 a déjà été exploré : un futur découpage de ces mêmes dates ne recrée pas un test
entièrement vierge. Documenter ce biais de développement et prévoir, si possible, une période nouvelle
pour l'évaluation finale.

## Ce qui attend le notebook 05 et la préparation ultérieure

Le découpage temporel, l'horizon et le traitement des clients répétés précèdent tout apprentissage de
médianes d'imputation, catégories apprises, scalers ou seuils de clipping. Les paramètres futurs sont
ajustés sur le train seulement. Aucune de ces opérations et aucun modèle ne sont entraînés ici.

## Sauvegarde du dataset transformé

Le notebook termine par un enregistrement effectif dans
`data/processed/churn_clients_processed.csv`. Ce fichier contient les 17 features candidates,
`customer_id`, `snapshot_date` et la cible. La cible et les colonnes de contexte ne sont pas des features :
le manifeste enregistre ces deux listes séparément.

Le catalogue est conservé dans `churn_clients_processed.catalog.csv` et le statut de disponibilité
et les dtypes dans `churn_clients_processed.manifest.json`, sans affichage détaillé dans le notebook.
Une réexécution actualise ces fichiers par remplacement atomique de chaque fichier.
Les sources brutes sont conservées. Les CSV relus devront respecter les types du manifeste.

Les contrôles de reproductibilité et de disponibilité restent dans le package et les tests.
Le notebook se concentre sur les transformations et se termine après la sauvegarde.

## Vérification

Tests sur valeurs invalides/manquantes, catégories inconnues, dates négatives, alignement, cible absente
ou permutée, invariance à la population, propagation des manquants dans le produit et export sans écrasement.
Sur les fichiers actuels : 38 689 lignes, 17 candidats, 173 anciennetés indisponibles et 113 frais mensuels
invalidés en plus des 1 897 absents. Les 26 245 NPS absents ne sont pas imputés.

## Regroupement de l'âge

L'usage de `age_group` comme feature candidate a été explicitement demandé. `age` reste exclue
de la matrice, mais sa dérivée est bien une information d'âge. Les bornes sont configurables et
indépendantes de la cible ; les valeurs invalides et absentes restent NA. La source `age` est tracée
dans le catalogue et sa disponibilité est déclarée `unconfirmed`. Cette représentation ne démontre
ni gain prédictif ni absence de biais. Elle doit être évaluée dans la validation ultérieure.
