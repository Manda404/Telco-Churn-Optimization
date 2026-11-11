# EDA — Analyses et recommandations détaillées

Version détaillée conservée lors de l’allègement du notebook 03. Les chiffres décrivent les fichiers analysés ; ils sont à réviser si les sources changent.

# 03 — Comprendre le churn : questions, observations et décisions

**Périmètre : `churn_clients.csv` uniquement.** L'étude de la campagne pilote constitue une phase séparée.
Les notebooks 01 et 02 présentent les sources et les contrôles qualité. Ici, nous cherchons à comprendre
les associations qui guideront la préparation des données et la définition du futur protocole de modélisation.

Chaque axe suit la même lecture : **question → indicateurs/visuels → observation → implication et limite**.
Le parcours couvre **25 questions distinctes** ; certaines partagent une cellule de calcul ou de visualisation.
La carte des colonnes et la synthèse finale distinguent les analyses réalisées des investigations restantes.
Une EDA n'épuise pas tous les croisements possibles : la couverture est documentée, pas déclarée exhaustive.

**Repère de lecture.** Les observations écrites ci-dessous correspondent aux fichiers actuels :
38 689 lignes, 4 000 clients et des snapshots de 2024. Si les données ou le YAML changent,
les tableaux et graphiques recalculés font référence ; les commentaires chiffrés devront être révisés.

**Exécution :** kernel **Python (Telco Churn)**. Les paramètres viennent de `configs/data.yaml`,
les chemins et les cibles de `Settings`, et les domaines proposés du contrat qualité YAML.

**Audit critique :** [décisions détaillées sur les 19 axes](../docs/eda_audit.md).
Les six ajouts répondent à des questions conditionnelles, géographiques et temporelles précises.
Le parcours reste strictement : EDA → analyses → hypothèses → recommandations pour le futur FE.
Aucune variable de modèle, pipeline d'entraînement, sélection de features ou analyse du pilote n'est implémentée.


## Préparer une lecture cohérente des résultats

Les données restent brutes : copies exactes, variantes de catégories et valeurs extrêmes sont conservées,
sauf dans les vues analytiques explicitement annoncées. Les cibles manquantes sont exclues des taux ;
une cible renseignée autre que 0/1 interrompt les calculs.

Dans ce notebook, un **taux de churn** désigne la proportion de lignes positives parmi les lignes à cible
observée. Il ne s'agit pas automatiquement d'un taux annuel ni d'un risque individuel futur.
Les observations répétées d'un même client ne sont pas indépendantes. Le seuil de petit groupe porte
sur les **lignes**, pas sur les clients distincts, et ne constitue pas un test de significativité.

Les associations observées ne démontrent pas une causalité. L'âge est étudié à titre descriptif, sans
proposition de ciblage par âge. `retention_offer_declined` est réservé à un audit de disponibilité temporelle.

**Précision des approfondissements.** Les taux des nouvelles heatmaps sont masqués sous 100 lignes ou
30 clients distincts (seuils YAML). Les IC à 95 % sont des percentiles de 300 rééchantillonnages de
clients avec leurs historiques entiers, pas de lignes indépendantes. Ils sont omis en cas de support
insuffisant, notamment de clients positifs. Ils ne corrigent ni sélection du panel, ni confusion, ni
comparaisons multiples et ne constituent pas des tests de causalité ou d'interaction.


### Recharger les composants pendant le développement

La cellule suivante actualise les modules EDA et les visualisations avant leur import.
Après une modification du fichier, rouvrir le notebook pour actualiser les cellules affichées,
puis relancer depuis le début. Cette étape est technique ; elle ne constitue pas un axe d'analyse.

Le rechargement respecte aussi la dépendance schémas → exploration → diagnostics → visualisations.


## Axes 1–2 — Population observée et prévalence du churn

### Axe 1 — Quelle est l'unité réelle d'observation ?

**Question d’analyse.** Avons-nous une ligne par client ou plusieurs photographies d'un même client ?
**Méthodologie.** Comparer les lignes, les identifiants distincts et le nombre d'observations par client.
Cela détermine comment pondérer les résultats et organiser une validation sans chevauchements inappropriés.

### Axe 2 — Quelle est la fréquence de la classe positive ?

**Question d’analyse.** Le churn est-il suffisamment fréquent pour interpréter correctement les performances futures ?
**Méthodologie.** Examiner les lignes positives, le dénominateur à cible observée et le taux global.
Ce taux sert de référence descriptive aux analyses segmentaires suivantes.

**KPI étudiés — axe 1.** Lignes, clients distincts, distribution du nombre de snapshots par client.

**KPI étudiés — axe 2.** Cibles connues, snapshots positifs et taux positif par snapshot.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axes 1–2

**Axe 1.** Les **38 689 lignes concernent 4 000 clients**, soit 9,67 observations par client en moyenne.
La médiane est de 12 lignes ; le maximum de 13 ne signifie pas 13 dates distinctes, car des copies existent.
**Implication :** les taux calculés sur les lignes donnent davantage de poids aux clients observés longtemps.
Il faudra tenir compte des dates et des identifiants dans le protocole de validation.

**Axe 2.** La cible est renseignée sur toutes les lignes : **1 164 positives**, soit **3,01 %**.
La classe négative est donc très majoritaire.
**Implication :** une bonne accuracy seule ne suffira pas à juger le futur modèle ; examiner les performances
sur la classe positive et les contraintes de campagne. Aucun choix de métrique ou de seuil n'est validé ici.

**Hypothèse à approfondir — axe 1.** La longueur des historiques pondère les taux et peut modifier la population représentée.

**Implications pour le futur Feature Engineering — axe 1.** Définir le protocole temporel et la gestion des identifiants ; pas de feature ID.

**Hypothèse à approfondir — axe 2.** La rareté de la cible nécessite une évaluation centrée sur les positifs.

**Implications pour le futur Feature Engineering — axe 2.** Pas de feature dérivée de la prévalence ; définir les métriques ultérieurement.


## Axe 3 — Les copies exactes changent-elles la lecture globale ?

**Question d’analyse.** Les doublons détectés dans le notebook 02 modifient-ils sensiblement le taux de churn ?
**Méthodologie.** Comparer le taux brut à celui d'une vue sans copies exactes, sans modifier le fichier source.
Cette comparaison ne résout pas les conflits éventuels sur la clé client/date.

**KPI étudiés — axe 3.** Lignes et positifs avant/après retrait temporaire des copies, écart du taux.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axe 3

Le retrait temporaire des **60 copies exactes** ramène les lignes positives de 1 164 à 1 162.
Le taux passe de **3,0086 % à 3,0081 %** : la lecture globale change très peu.
**Implication :** les copies n'expliquent pas la prévalence globale, mais elles restent à traiter dans une
politique explicite de qualité. Leur faible effet global ne garantit pas un faible effet sur chaque segment.

**Hypothèse à approfondir — axe 3.** Un faible effet global peut masquer des différences locales.

**Implications pour le futur Feature Engineering — axe 3.** Valider la politique de doublons avant la construction des features.


## Axe 4 — Le churn et la population observée sont-ils stables dans le temps ?

**Question d’analyse.** Les écarts de taux entre snapshots accompagnent-ils un changement de volume observé ?
**Méthodologie.** Lire ensemble les effectifs de lignes, les clients distincts et le taux à chaque date.
Les dates sont des dates de photographie ; leur lien avec la période effective de résiliation reste à préciser.

**KPI étudiés — axe 4.** Effectifs, clients distincts et taux par date de photographie.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axe 4

Entre janvier et décembre 2024, les effectifs passent de **3 494 à 2 915 lignes**.
Le taux observé varie de **2,40 % environ en juin à 3,55 % en août**, sans baisse régulière sur l'année.
**Implication :** vérifier la stabilité sur des périodes futures et expliquer la diminution de la population.
Ces variations ne suffisent pas à établir une saisonnalité sur une seule année ; les trajectoires de l'axe 19
complètent cette lecture du panel.

**Hypothèse à approfondir — axe 4.** Une partie des variations reflète la composition et la sortie du panel.

**Implications pour le futur Feature Engineering — axe 4.** Tester des informations temporelles seulement sur des fenêtres disponibles avant le label.


## Axes 5–7 — Quels segments présentent des associations différentes avec le churn ?

### Axe 5 — Le type de contrat distingue-t-il les taux observés ?

**Question d’analyse.** Les contrats mensuels, annuels et biannuels ont-ils les mêmes profils de churn ?
**Méthodologie.** Comparer les taux et les effectifs par libellé de contrat, puis approfondir par ancienneté à l'axe 15.

### Axe 6 — Les différences géographiques sont-elles importantes ?

**Question d’analyse.** Certaines régions présentent-elles des taux différents avec des effectifs suffisants ?
**Méthodologie.** Comparer les taux régionaux sans les attribuer directement à la géographie : la composition
contractuelle ou l'ancienneté des clients peut différer entre régions.

### Axe 7 — Les canaux d'acquisition présentent-ils des profils différents ?

**Question d’analyse.** Le canal révèle-t-il une différence de population associée au churn ?
**Méthodologie.** Examiner les modalités et leur support, en distinguant les variantes de format des libellés principaux.

Les trois graphiques sont produits par la même cellule. Les catégories restent brutes : les guillemets
rendent les espaces visibles et ne prouvent pas la présence de guillemets dans les données.
Les petits groupes sont grisés ; le seuil est un repère de lecture, pas une garantie statistique.

**KPI étudiés — axe 5.** Taux et effectifs par libellé brut de contrat ; sensibilité normalisée à l’axe 15.

**KPI étudiés — axe 6.** Taux régionaux et effectifs ; compléter avec composition et âge aux axes 23–24.

**KPI étudiés — axe 7.** Taux par canal et effectifs de chaque libellé brut.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axes 5–7

**Axe 5 — Contrat.** Pour les libellés exacts en minuscules, les taux sont **4,81 % (`mensuel`),
1,24 % (`annuel`) et 0,72 % (`biannuel`)**. Les variantes de casse et d'espaces restent séparées.
**Implication :** le contrat est une variable candidate à examiner, après validation de la normalisation.
Ces écarts ne prouvent pas qu'un changement de contrat réduirait le churn.

**Axe 6 — Région.** Les taux vont de **2,48 % en ARA à 4,00 % en IDF**, avec plus de 5 000 lignes par région.
**Implication :** approfondir la composition des populations avant toute interprétation géographique.
Des effectifs de lignes élevés n'éliminent pas la dépendance entre snapshots d'un même client.

**Axe 7 — Canal.** Les libellés principaux sont proches : **2,96 % pour `web`, 2,96 % pour `telephone`,
3,02 % pour `boutique` et 3,22 % pour `partenaire`**. Certaines variantes ont des taux plus élevés mais
portent sur moins de lignes ; elles ne doivent pas être confondues avec un effet du canal.
**Implication :** normaliser les libellés selon une politique validée et réévaluer les écarts avant de les classer.

**Hypothèse à approfondir — axe 5.** Le contrat apporte une information conditionnelle à l’ancienneté.

**Implications pour le futur Feature Engineering — axe 5.** Encodage du contrat normalisé, sous réserve de validation des libellés.

**Hypothèse à approfondir — axe 6.** Les écarts régionaux reflètent en partie les profils de clients.

**Implications pour le futur Feature Engineering — axe 6.** Variable contextuelle candidate, à vérifier hors période et après contrôle de composition.

**Hypothèse à approfondir — axe 7.** Les différences du canal sont sensibles aux formats et aux segments représentés.

**Implications pour le futur Feature Engineering — axe 7.** Normalisation de catégories à tester dans la phase suivante.


## Axes 8–9 — Les montants et les usages diffèrent-ils selon la cible ?

### Axe 8 — La tarification et la dernière facture racontent-elles la même chose ?

**Question d’analyse.** Les observations positives se distinguent-elles par leur abonnement ou leur dernière facture ?
**Méthodologie.** Comparer les médianes et dispersions de `monthly_fee` et `last_invoice_amount`, pas seulement les moyennes.
La disponibilité de la dernière facture avant le moment de prédiction doit être confirmée.

### Axe 9 — Le volume de données consommées est-il associé au churn ?

**Question d’analyse.** Les observations positives présentent-elles un usage plus faible ou plus élevé ?
**Méthodologie.** Comparer médianes, quartiles et maxima de `data_gb`, pour repérer l'influence des valeurs extrêmes.

Les visuels existants présentent aussi support, retards et NPS ; leurs relations avec le taux de churn seront
interprétées aux axes 12–14. Les valeurs numériques absentes, non interprétables ou infinies sont exclues
de ces statistiques et comptées dans `n_excluded`. Les valeurs extrêmes finies restent incluses.
Les points hors moustaches peuvent être masqués à l'affichage, mais pas dans les statistiques.

**KPI étudiés — axe 8.** Quantiles des montants par cible et effectifs numériques disponibles.

**KPI étudiés — axe 9.** Médianes, quartiles, moyennes et maxima de consommation par cible.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axes 8–9

**Axe 8 — Montants.** La médiane de `monthly_fee` est de **63,18 pour les lignes positives contre 60,74
pour les négatives**. Pour `last_invoice_amount`, elle est au contraire de **28,90 contre 60,79**.
**Implication :** la dernière facture mérite un audit temporel prioritaire : l'écart peut refléter un ajustement,
une facture partielle ou une information postérieure à la décision de résiliation. L'EDA ne permet pas de trancher.
Les frais négatifs signalés au notebook 02 limitent aussi l'interprétation des montants bruts.

**Axe 9 — Usage.** La médiane de `data_gb` est de **2,8 pour les lignes positives contre 3,7 pour les négatives**.
Les moyennes sont beaucoup plus élevées (36,91 et 72,89), et le maximum global atteint **32 154**.
**Implication :** l'asymétrie et les valeurs extrêmes doivent être expliquées ; confirmer l'unité et les domaines
plausibles avant une transformation. Un faible usage observé n'est pas une cause démontrée du churn.

**Hypothèse à approfondir — axe 8.** La dernière facture peut contenir une information postérieure au moment de prédiction.

**Implications pour le futur Feature Engineering — axe 8.** Aucune feature de facture avant preuve de disponibilité ; ne pas créer de ratio financier ici.

**Hypothèse à approfondir — axe 9.** L’usage entretient une relation non linéaire influencée par les valeurs extrêmes.

**Implications pour le futur Feature Engineering — axe 9.** Étudier une transformation robuste après confirmation des unités, sans la construire ici.


## Axe 10 — Quelles associations linéaires et redondances apparaissent ?

**Question d’analyse.** Quelles variables évoluent ensemble, et lesquelles sont linéairement associées à la cible ?
**Méthodologie.** Lire la matrice de Pearson avec les effectifs disponibles par paire. Une corrélation faible
n'exclut pas une relation non linéaire ; une corrélation élevée n'établit ni causalité ni utilité prédictive future.
Les valeurs extrêmes et les manquants peuvent modifier fortement cette lecture.

**KPI étudiés — axe 10.** Pearson et nombres de paires disponibles pour chaque coefficient.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axe 10

`monthly_fee` et `last_invoice_amount` sont fortement corrélés (**r ≈ 0,88**). La dernière facture est
négativement associée à la cible (**r ≈ −0,265**), alors que les appels au support lui sont positivement
associés (**r ≈ 0,118**). Les corrélations avec `nps` reposent sur moins de lignes disponibles.
**Implication :** examiner la redondance des montants et auditer la disponibilité temporelle de la facture.
Ne pas supprimer ou sélectionner une variable sur la seule base de cette matrice ; elle ne teste pas
les non-linéarités et ne corrige pas la dépendance entre observations d'un même client.

**Hypothèse à approfondir — axe 10.** La redondance des montants et des relations non linéaires limitent la lecture de Pearson.

**Implications pour le futur Feature Engineering — axe 10.** Comparer des représentations dans la phase suivante, sans feature selection maintenant.


## Axe 11 — Le début de la relation client concentre-t-il le churn ?

**Question d’analyse.** Le taux observé est-il plus élevé pendant les premiers mois d'abonnement ?
**Méthodologie.** Comparer taux, effectifs, part des lignes positives et écart au taux global par tranche d'ancienneté.
Les KPI détaillés sont au survol. Le ratio de taux est descriptif, pas un uplift causal ni un lift de modèle.

L'ancienneté provient d'une copie analytique : `(snapshot_date − signup_date) / 30,4375 jours`.
C'est une durée approchée, pas un nombre de mois calendaires complets. Les intervalles sont `[a, b)`.
Les durées négatives et dates inutilisables restent identifiables comme statuts de qualité.
Un même client peut contribuer à plusieurs bandes au cours du temps.

**KPI étudiés — axe 11.** Taux, lignes positives, clients distincts, part des positifs et audit des dates.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axe 11

Le taux passe de **7,40 % avant 3 mois à 2,30 % à partir de 48 mois**.
La bande 24–48 mois représente toutefois **32,47 % des lignes positives**, malgré un taux inférieur au début
de relation, car elle contient davantage d'observations. **173 durées négatives** sont signalées séparément.
**Implication :** examiner une relation non linéaire avec l'ancienneté, en distinguant niveau de taux et volume.
Ce profil peut aussi refléter la sélection progressive des clients qui restent ; il ne démontre pas que
l'ancienneté protège causalement contre le churn.

**Hypothèse à approfondir — axe 11.** La relation à la durée est non linéaire et peut refléter un effet de sélection.

**Implications pour le futur Feature Engineering — axe 11.** Ancienneté au snapshot candidate après validation de signup_date et snapshot_date.


## Axes 12–14 — Quels signaux comportementaux sont associés au churn ?

### Axe 12 — Les appels au support révèlent-ils un niveau de difficulté particulier ?

**Question d’analyse.** Le taux augmente-t-il lorsque le nombre d'appels récents devient élevé ?
**Méthodologie.** Lire les taux par bande d'appels avec les observations, clients distincts et lignes positives au survol.

### Axe 13 — Les incidents de paiement s'accompagnent-ils d'un churn plus élevé ?

**Question d’analyse.** Les observations avec plusieurs retards présentent-elles une association différente de celles sans retard ?
**Méthodologie.** Comparer le gradient de taux et les effectifs ; un groupe à taux élevé peut contenir peu de positifs au total.

### Axe 14 — La satisfaction renseignée distingue-t-elle les taux observés ?

**Question d’analyse.** Les notes de satisfaction faibles sont-elles associées à davantage de churn ?
**Méthodologie.** Comparer les bandes de `nps` sur les valeurs disponibles. L'interprétation provisoire reste une
note individuelle de 0 à 10, à confirmer ; ce n'est pas nécessairement un NPS agrégé.

Ces graphiques complètent les boxplots : ils présentent le **taux de churn conditionnel à un niveau**,
plutôt que la distribution d'une variable selon la cible. Les bornes sont exploratoires, pas des seuils
opérationnels validés. Les domaines viennent du contrat qualité. Les manquants sont analysés à l'axe 16 ;
les valeurs hors domaine restent traçables dans les tables exportables.

**KPI étudiés — axe 12.** Taux, effectifs, clients et volume positif par niveau d’appels.

**KPI étudiés — axe 13.** Taux, effectifs et part des positifs par niveau de retards.

**KPI étudiés — axe 14.** Taux et effectifs par niveau de satisfaction, sur les notes disponibles.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axes 12–14

**Axe 12 — Support.** Le taux passe de **1,49 % sans appel à 14,43 % avec au moins 5 appels**.
Cette dernière bande contient **499 lignes, 413 clients distincts et 72 lignes positives**.
**Implication :** examiner une relation non linéaire et vérifier que les appels sont connus avant la prédiction.
La différence ne mesure pas l'effet causal des appels ou du service support.

**Axe 13 — Paiements.** Le taux passe de **2,27 % sans retard à 11,90 % avec au moins 3 retards**.
La dernière bande contient **378 lignes et 45 positives**, soit seulement 3,87 % des lignes positives globales.
**Implication :** un taux élevé ne garantit pas de couvrir une grande part du churn ; examiner simultanément
la fréquence du signal et sa stabilité temporelle, sans fixer ici un seuil de campagne.

**Axe 14 — Satisfaction.** Le taux est de **5,09 % pour une note inférieure à 5**, contre **1,90 % pour une note
au moins égale à 9**. Mais seules **12 444 lignes** ont une note disponible.
**Implication :** l'association est descriptive sur la population répondante ; elle ne peut pas être extrapolée
sans examiner le mécanisme de collecte et les **26 245 notes manquantes**.

**Hypothèse à approfondir — axe 12.** Le cumul d’incidents apporte un signal au-delà du nombre d’appels seul.

**Implications pour le futur Feature Engineering — axe 12.** Piste d’interaction, à vérifier à l’axe 20 et avec la fenêtre de mesure.

**Hypothèse à approfondir — axe 13.** Le cumul de difficultés de paiement et de support distingue certains profils.

**Implications pour le futur Feature Engineering — axe 13.** Aucune feature de friction avant confirmation des fenêtres et validation future.

**Hypothèse à approfondir — axe 14.** La satisfaction et le recours au support sont liés à la population répondante.

**Implications pour le futur Feature Engineering — axe 14.** Tester ultérieurement imputation/indicateur seulement après audit de collecte.


## Axe 15 — L'association entre ancienneté et churn varie-t-elle selon le contrat ?

**Question d’analyse.** La diminution du taux avec l'ancienneté est-elle comparable entre contrats mensuels,
annuels et biannuels ?
**Méthodologie.** Croiser contrat et ancienneté, puis lire les mêmes cellules sous forme de courbes.
Ce croisement complète les axes à une variable ; il ne démontre pas formellement une interaction statistique.

La vue utilise les anciennetés valides : **173 lignes sont exclues de ce croisement seulement**.
Les catégories restent brutes. La heatmap masque les cellules sous le seuil de lignes ; leurs données
restent disponibles dans `interaction_table`. Les taux de cette vue se rapportent à cette population restreinte.

**KPI étudiés — axe 15.** Taux, volumes, clients positifs, IC exploratoires et sensibilité des libellés.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Complément de lecture du même axe : courbes par contrat

Les courbes ci-dessous reprennent les cellules de la heatmap pour les libellés exacts `mensuel`,
`annuel` et `biannuel`. Les variantes de casse ou d'espaces ne sont pas fusionnées.
Les cellules sous le seuil interrompent les lignes. Le survol donne les effectifs et les lignes positives.
Cette seconde représentation sert à comparer les profils ; elle ne constitue pas un nouvel axe indépendant.


### Analyse des résultats — Axe 15

Pour le libellé exact `mensuel`, le taux passe de **12,65 % avant 3 mois à 3,68 % à partir de 48 mois**.
Pour `annuel`, il passe de **2,23 % à 0,88 %**. Le profil de `biannuel` fluctue davantage : il ne présente
pas une baisse régulière sur toutes les bandes.
**Implication :** le contexte contractuel mérite d'être examiné avec l'ancienneté. Avant de confirmer une
interaction candidate, valider la normalisation des libellés, les effectifs et la stabilité sur une autre période.
Aucun effet d'un changement de contrat sur le churn n'est estimé ici.

**Hypothèse à approfondir — axe 15.** L’association de l’ancienneté dépend du contexte contractuel.

**Implications pour le futur Feature Engineering — axe 15.** Interaction contrat × durée candidate, sans création de colonne de modèle.


### Sensibilité de l'axe 15 aux catégories normalisées

**Question.** Les profils observés sur les seuls libellés canoniques persistent-ils lorsque les variantes
casse/espaces sont regroupées dans une copie EDA ? Les libellés inconnus restent signalés.
**KPI et méthode.** Même population à ancienneté valide ; regroupement documenté, clients positifs et IC
par clients. Le menu sépare taux et volumes ; les bornes d'IC au survol concernent chaque taux marginal.


**Analyse des résultats.** Après regroupement de **1 189 formats de contrat**, le taux mensuel
passe de **12,44 % avant 3 mois (892 lignes, 376 clients, 111 positifs)** à **3,70 % à partir de 48 mois**.
Pour les annuels, les valeurs correspondantes sont **2,34 % et 0,88 %**. Le profil principal persiste,
mais les chiffres changent par rapport aux seuls libellés exacts. Les biannuels de 3–6 mois ne comptent
qu'un client positif : aucun IC précis n'est fabriqué pour cette cellule.
**Hypothèse.** Le contraste contractuel n'est pas uniquement un artefact de casse, mais peut rester confondu.
**Piste FE future.** Tester contrat normalisé et durée après confirmation métier, sans en déduire un effet causal.


## Axe 16 — L'absence d'une valeur est-elle elle-même associée au churn ?

**Question d’analyse.** Les observations avec et sans âge, frais mensuels ou note de satisfaction renseignés
présentent-elles des taux différents ?
**Méthodologie.** Comparer la disponibilité de la valeur et le taux de churn, sans reproduire le simple inventaire
des manquants du notebook 02. Les chaînes vides et les espaces seuls sont considérés comme manquants.
Les valeurs renseignées ne sont pas nécessairement valides dans le domaine métier.

L'âge est examiné ici sous l'angle de sa disponibilité uniquement. Une association avec un manquant peut
refléter la collecte ou la population répondante : elle ne suffit pas à valider un indicateur pour le modèle.

**KPI étudiés — axe 16.** Taux par disponibilité de chaque valeur, effectifs et écart en points.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axe 16

Pour `nps`, le taux est de **2,87 % sans note contre 3,29 % avec note**. Pour l'âge, il est de **3,62 % sans
valeur contre 2,97 % avec valeur**. L'écart est plus faible pour `monthly_fee` : **3,16 % contre 3,00 %**.
**Implication :** étudier la collecte et les différences de population avant d'imputer ou de créer un indicateur
de manquant. Ces écarts ne démontrent pas le mécanisme statistique des manquants et ne justifient pas
une règle de ciblage. Toute préparation future devra être ajustée sur les données d'entraînement seulement.

**Hypothèse à approfondir — axe 16.** L’absence de valeur reflète une différence de collecte ou de population.

**Implications pour le futur Feature Engineering — axe 16.** Comparer une stratégie d’imputation et d’indicateur dans le futur train uniquement.


## Axes 17–18 — Âge : distribution et association avec la cible

### Axe 17 — Quelle est la distribution des âges, et quels points demandent une vérification ?

**Question d’analyse.** Les âges se concentrent-ils sur certaines plages, et comment distinguer valeurs atypiques
statistiques et anomalies de domaine ?
**Méthodologie.** Le boxplot horizontal est placé au-dessus de l'histogramme et de la courbe lissée KDE,
sur le même axe. Le boxplot montre les points au-delà de 1,5 IQR ; la KDE ne suppose pas une loi normale
et ne constitue pas un test de normalité.

Le menu compare le domaine proposé **18–120 ans** et toutes les valeurs finies, dont les anomalies telles
que −1 ou 999. Le boxplot et la courbe utilisent toujours la même population. Les valeurs manquantes,
non interprétables ou infinies ne sont pas représentées ; chaque snapshot contribue à la distribution.

**KPI étudiés — axe 17.** Distribution, quantiles, nombre de points IQR et effectifs hors domaine/manquants.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axe 17

Dans le domaine proposé, **36 197 âges sont représentés**, avec une médiane de **43 ans** et une moitié
centrale comprise entre **33 et 52 ans**. Le boxplot repère **105 points atypiques statistiques** dans cette vue.
À part, la source contient **364 valeurs hors domaine** et **2 128 valeurs manquantes**.
**Implication :** distinguer les âges rares mais plausibles des anomalies métier ; ne pas supprimer
systématiquement les points signalés par l'IQR. Aucune normalité de la distribution n'est établie.

**Hypothèse à approfondir — axe 17.** Les points atypiques statistiques et les erreurs métier demandent des traitements différents.

**Implications pour le futur Feature Engineering — axe 17.** Règles de qualité à confirmer ; aucune règle de ciblage par âge.


### Axe 18 — Le taux de churn diffère-t-il selon la tranche d'âge ?

**Question d’analyse.** Quelles différences descriptives observe-t-on selon l'âge, avec combien de clients distincts ?
**Méthodologie.** Comparer taux, effectifs et écarts au taux global au survol. Les tranches viennent du YAML ;
la dernière bande (≥ 80 ans) reste limitée au domaine proposé. Les manquants sont étudiés à l'axe 16.

Un client peut contribuer à plusieurs observations ou changer de tranche. L'ancienneté et le contrat
peuvent expliquer une partie des écarts. Cette analyse ne prouve ni un effet causal ni un biais du futur
modèle ; elle ne construit pas de règle de ciblage par âge.

**KPI étudiés — axe 18.** Taux ET volume positif par tranche, clients distincts et support effectif.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axe 18

Le taux est de **3,62 % entre 18 et 25 ans exclus**, contre **2,65 % entre 55 et 65 ans exclus**.
Il n'est pas strictement décroissant sur toutes les bandes. La tranche ≥ 80 ans affiche **0,83 %**,
mais ne contient que **120 lignes, 12 clients distincts et une seule ligne positive**.
**Implication :** le seuil de 100 lignes ne suffit pas à garantir une base solide au niveau client.
Interpréter cette dernière tranche avec prudence et examiner les effets de composition, sans proposer
un ciblage par âge ni conclure sur l'équité d'un modèle qui n'est pas encore entraîné.

**Hypothèse à approfondir — axe 18.** Une partie des écarts d’âge provient de la région, du contrat ou de l’ancienneté.

**Implications pour le futur Feature Engineering — axe 18.** Audit descriptif et biais ultérieurs ; aucune feature de ciblage par âge recommandée.

La tranche à faible nombre de clients est désormais grisée, même si elle dépasse le seuil de lignes.


## Axe 19 — Comment les clients positifs apparaissent-ils dans les trajectoires ?

**Question d’analyse.** Un client peut-il être positif à plusieurs dates et rester observé après ce premier signal ?
**Méthodologie.** Compter les dates positives distinctes, les observations postérieures au premier positif et
les contradictions de cible à une même clé client/date. Les copies exactes ne créent pas de faux événements répétés.

La moyenne des taux individuels par snapshot donne le même poids à chaque client. Elle doit être
distinguée du taux pondéré par les lignes et d'une proportion de clients ayant résilié sur une année.

**KPI étudiés — axe 19.** Clients positifs, dates positives distinctes, conflits de labels et suivi post-positif.

**Visualisation.** Les cellules suivantes présentent les tableaux et graphiques associés ; les axes regroupés partagent les mêmes calculs sans reproduire les visuels.


### Analyse des résultats — Axe 19

**1 162 clients** ont au moins une date positive. Aucun n'est positif à plusieurs dates distinctes,
aucun n'est observé à une date ultérieure au premier positif, et aucune clé client/date ne porte de cibles
contradictoires. La moyenne des taux individuels par snapshot est de **9,45 %**, contre **3,01 %** pour le taux par ligne.
**Implication :** clarifier la définition du label et la règle de sortie du panel avant de construire la validation.
Ces résultats sont compatibles avec une fin de suivi au premier signal, mais l'EDA ne démontre pas le mécanisme
de constitution des données. L'écart de pondération n'est ni un taux annuel ni une mesure de risque individuel.

**Hypothèse à approfondir — axe 19.** Le processus d’observation cesse possiblement au premier label positif.

**Implications pour le futur Feature Engineering — axe 19.** Clarifier horizon, censure et disponibilité temporelle avant toute construction de features.


## Axe 20 — Support × retards : les difficultés cumulées distinguent-elles le churn ?

**Question d'analyse.** Les observations cumulant appels et retards présentent-elles un taux et un volume
positif plus importants que les observations sans ces incidents ?

**KPI étudiés.** Taux par snapshot, lignes, clients distincts, snapshots positifs, clients positifs, IC à 95 %,
écart et ratio au taux de la population éligible. Aucun compte de snapshots n'est nommé « résiliations ».

**Méthodologie.** Bandes configurées, copie EDA uniquement, aucune suppression de la source.
Le bootstrap rééchantillonne les clients entiers ; les domaines et les exclusions sont audités.
La lecture est conditionnelle et descriptive, sans test de synergie ni effet causal.

**Visualisation.** Heatmap avec menu distinct pour les taux et les volumes, support et IC au survol.


### Analyse des résultats — Axe 20

Avec **au moins 3 appels et 2 retards**, le taux est de **13,98 %** sur **415 lignes, 360 clients et
58 snapshots positifs** (58 clients positifs), contre **1,07 %** sans appel ni retard. L'IC exploratoire
du premier taux est d'environ **11,26–17,32 %**. Ce segment ne représente que **4,98 % des snapshots positifs** :
une forte proportion locale n'est pas une forte couverture globale.

**Hypothèses à approfondir.** Des difficultés simultanées pourraient apporter une information conditionnelle
au-delà de chaque variable seule. Confusion par contrat, ancienneté et chronologie des incidents restent possibles.
Les IC marginaux ne constituent pas un test de différence ni d'interaction corrigé pour comparaisons multiples.

**Implications pour le futur Feature Engineering.** Interaction support/paiements candidate après confirmation
des fenêtres antérieures au snapshot ; aucun score de friction définitif n'est construit ici.


## Axe 21 — Abonnement × consommation : un coût élevé et un faible usage vont-ils ensemble avec le churn ?

**Question d'analyse.** Les profils combinant abonnement élevé et faible consommation se distinguent-ils ?

**KPI étudiés.** Taux, effectifs, clients positifs, IC et volumes ; dénominateur limité aux montants et usages éligibles.

**Méthodologie.** Bandes exploratoires du YAML (unités des montants à confirmer), sans ratio coût/Go.
Les montants manquants ou négatifs sont exclus de cette vue seulement ; les grandes consommations finies
restent dans la dernière bande, faute de plafond métier validé.

**Visualisation.** Heatmap multi-KPI ; ne pas lire les couleurs de volume comme des taux.


### Analyse des résultats — Axe 21

La vue conserve **36 679 lignes** et en exclut **2 010** pour frais inutilisables. Pour un abonnement
**≥ 80** et un usage **< 2 Go**, le taux est de **5,37 %** : **1 452 lignes, 549 clients, 78 positifs**,
IC exploratoire environ **4,05–6,47 %**. À abonnement ≥ 80 et usage ≥ 10 Go, il est de **2,64 %**.
Le taux global de cette vue est d'environ **3,00 %**, différent par définition de celui de la source brute.

**Hypothèses à approfondir.** Une adéquation perçue entre tarif et usage pourrait intervenir ; niveau de service,
contrat ou ancienneté peuvent aussi expliquer l'association. Les unités et extrêmes d'usage restent incertains.

**Implications pour le futur Feature Engineering.** Tester une représentation conjointe tarif/usage seulement
après validation des unités et disponibilité. Aucun ratio financier ni feature tarifaire définitive n'est créé.


## Axe 22 — Satisfaction × support : le signal des appels dépend-il de la note disponible ?

**Question d'analyse.** Le recours au support accompagne-t-il un taux élevé pour toutes les notes,
y compris lorsque le NPS est manquant ?

**KPI étudiés.** Taux par croisement, clients positifs, volumes, IC et effectifs répondants/non-répondants.

**Méthodologie.** Le groupe NPS « Non renseigné » est conservé. Les domaines invalides sont exclus,
les fenêtres et la date de collecte ne sont pas présumées disponibles avant le label.

**Visualisation.** Même échelle de taux pour les bandes de satisfaction et les niveaux de support.


### Analyse des résultats — Axe 22

Pour une note **< 5** et **≥ 3 appels**, le taux est de **8,77 %** sur **422 lignes et 323 clients**
(37 positifs), contre **1,27 %** sans appel dans la même bande de note. Ce dernier taux ne repose que sur
**4 clients positifs** : aucun IC n'est présenté. Même avec une note ≥ 9, le groupe ≥ 3 appels atteint
**7,48 %**, mais sur **147 lignes** et avec un IC large (environ **3,47–11,42 %**).
Sans NPS, les taux passent de **1,41 % sans appel à 7,63 % avec ≥ 3 appels**.

**Hypothèses à approfondir.** Le recours au support pourrait apporter un signal qui ne se réduit pas à la
satisfaction renseignée. La sélection des répondants et le déclenchement des enquêtes peuvent confondre ces liens.

**Implications pour le futur Feature Engineering.** Comparer ultérieurement satisfaction, disponibilité et
support conjointement, après audit de collecte. Aucune imputation ni feature d'interaction n'est implémentée.


## Axe 23 — Âge × région : où se concentrent les populations ?

**Question d'analyse.** Quelles régions regroupent les snapshots de chaque tranche d'âge, et quelle place
ces tranches occupent-elles dans la population observée de chaque région ?

**KPI étudiés.** Lignes et clients distincts ; part des snapshots d'une tranche par région, et part d'une région
par tranche, dans la population aux âges éligibles. Le menu explicite ces deux dénominateurs. Aucun taux de churn dans cette figure.

**Méthodologie.** Âges du domaine proposé et régions renseignées, dans une copie EDA. Les clients distincts
ne s'additionnent pas entre groupes si une personne change de tranche/région. Le poids d'un client suivi
longtemps reste plus élevé dans les pourcentages par snapshot.

**Visualisation.** Heatmap de composition, séparée de l'axe suivant qui porte sur les résultats de churn.


### Analyse des résultats — Axe 23

La vue géographique contient **36 197 lignes** ; **2 492** sont exclues pour âge manquant ou hors domaine.
Parmi les snapshots de 18–24 ans, **17,46 % sont en OCC** (637 lignes, 72 clients) et **16,01 % en IDF**
(584 lignes, 70 clients). Cela décrit leur répartition, pas un niveau de churn.
Pour les ≥ 80 ans, GES et ARA regroupent respectivement **36,67 % et 34,17 %** des lignes, mais seulement
**4 clients distincts dans chaque cellule** : les pourcentages sont fragiles.

**Hypothèses à approfondir.** La composition d'âge peut contribuer aux écarts régionaux globaux.
Un dénombrement de clients à une date commune aiderait à séparer composition et durée de suivi.

**Implications pour le futur Feature Engineering.** Cette analyse sert surtout à contrôler la représentativité
et les biais ; elle ne recommande pas d'interaction âge/région pour cibler des personnes.


## Axe 24 — Âge × région : où trouve-t-on un taux élevé, et où se trouve le volume positif ?

**Question d'analyse.** À tranche d'âge comparable, les régions présentent-elles des taux différents,
et les cellules aux taux élevés sont-elles celles qui concentrent le plus de positifs ?

**KPI étudiés.** Taux, snapshots positifs, clients positifs, lignes, clients distincts et IC par clients.

**Méthodologie.** Même vue éligible que l'axe 23 ; petits groupes masqués selon lignes ET clients.
Les IC marginaux sont descriptifs et non ajustés pour les nombreuses comparaisons. Les effets de contrat,
ancienneté, période et suivi restent des facteurs de confusion potentiels.

**Visualisation.** Heatmap multi-KPI distincte de la composition ; menu pour passer des taux aux volumes.


### Analyse des résultats — Axe 24

La cellule **18–24 ans / IDF** présente **5,48 %** de churn, sur **584 lignes, 70 clients et 32 positifs** ;
l'IC exploratoire est d'environ **3,85–7,84 %**. La cellule **35–44 ans / IDF** a un taux inférieur (**4,36 %**)
mais un volume supérieur (**66 snapshots positifs**, sur 1 513 lignes et 178 clients).
**Taux, volume et clients positifs ne répondent donc pas à la même question.** Les cellules âgées peu
représentées ne sont pas classées : les taux y sont masqués, mais les compteurs restent accessibles.

**Hypothèses à approfondir.** Certaines différences régionales persistent dans une tranche, mais cela ne
suffit pas à isoler un effet de région ou d'âge, ni à conclure à une interaction statistique.

**Implications pour le futur Feature Engineering.** Étudier la représentativité des partitions et les biais
ultérieurs ; aucune règle de ciblage par âge ni feature géographique croisée n'est déduite automatiquement.


## Axe 25 — Les associations persistent-elles selon les mois et à composition contractuelle fixe ?

**Question d'analyse.** Les profils de contrat, support et âge sont-ils stables selon les dates,
et une évolution de la composition contractuelle explique-t-elle toute la variation globale ?

**KPI étudiés.** Taux mensuels par groupe, clients, positifs, IC et effectifs ; comparaison du taux brut
avec un taux standardisé à poids contractuels fixes, explicités dans le tableau de référence.

**Méthodologie.** Les regroupements sont des copies EDA ; les IC rééchantillonnent les historiques clients.
La standardisation utilise la composition contractuelle observée sur toute la période exploratoire,
ce qui est une analyse descriptive et non un paramètre appris pour l'entraînement. Une strate manquante
ou insuffisante donne un taux standardisé non disponible. La standardisation ne contrôle qu'une variable.

**Visualisation.** Profils mensuels en heatmaps, puis courbes brute/standardisée. Les cellules masquées
ne sont ni interpolées ni remplacées par zéro.


### Analyse des résultats — Axe 25

En janvier, le taux brut est de **3,18 %**, contre **2,97 %** à composition contractuelle fixe ;
en décembre, les valeurs sont **2,40 % et 2,57 %**. La variation persiste après cette standardisation :
les taux standardisés vont d'environ **2,39 % en juin à 3,60 % en août**.
La composition contractuelle influe donc sur la lecture, mais ne suffit pas à faire disparaître la variation.
Les contrats mensuels ont le taux le plus élevé sur les **12 snapshots** ; l'ordre annuel/biannuel
s'inverse en mars. Les taux du support ≥ 3 appels dépassent ceux sans appel sur les **12 snapshots**.
Les 18–24 ans ont un taux supérieur aux 55–64 ans sur **8 snapshots sur 12**, pas sur tous.
Les profils par groupe et leur support se lisent dans les heatmaps ; les groupes peu représentés,
notamment les âges élevés, ne doivent pas servir à affirmer une stabilité.

**Hypothèses à approfondir.** D'autres changements de composition, conditions de mesure ou mécanismes
de sortie du panel contribuent aux écarts. Une seule année et des cellules répétées ne prouvent pas une saisonnalité.

**Implications pour le futur Feature Engineering.** Priorité à une validation hors temps et à des fenêtres
strictement antérieures au label. Les poids de standardisation ne sont pas des features et ne doivent pas
être réutilisés comme un prétraitement appris sur le futur jeu de test.


# Synthèse de l'EDA — Hypothèses et recommandations pour le Feature Engineering

Cette synthèse se rapporte aux fichiers actuels. Les chiffres sont des faits descriptifs ; les explications
et les gains prédictifs restent des hypothèses. Les IC sont exploratoires, par clients, et non des tests
corrigés pour toutes les comparaisons. Les transformations temporaires `eda_*` ne constituent pas un pipeline FE.

## A. Principaux enseignements métier

- **Population et cible :** 38 689 snapshots, 4 000 clients, 1 164 lignes positives (3,01 %), concernant
  1 162 clients distincts. Les copies exactes expliquent deux lignes positives supplémentaires. Un client
  peut contribuer à plusieurs segments ; les effectifs clients ne s'additionnent pas entre segments mobiles.
- **Contrats et ancienneté :** après normalisation EDA, les contrats mensuels avant 3 mois ont un taux de
  12,44 % (892 lignes, 376 clients), contre 3,70 % à partir de 48 mois. Le profil existe aussi dans la vue
  brute mais avec des chiffres différents : la sensibilité des catégories ne doit pas être ignorée.
- **Support et paiements :** le cumul ≥ 3 appels et ≥ 2 retards atteint 13,98 % (415 lignes, 58 clients
  positifs). Il couvre 4,98 % des lignes positives : le fort taux ne garantit pas une couverture importante.
- **Tarif et usage :** pour les abonnements ≥ 80, le taux est de 5,37 % avec moins de 2 Go et de 2,64 %
  avec au moins 10 Go. Cela motive une hypothèse sur leur relation conjointe, pas un ratio financier validé.
  L'unité du montant, la durée de consommation et les usages extrêmes restent à confirmer.
- **Satisfaction :** la note faible est associée au churn sur les répondants, mais 26 245 notes sont
  manquantes. Avec ≥ 3 appels, les taux sont élevés même pour NPS ≥ 9 (7,48 %, 147 lignes, IC large).
  On ne peut pas conclure que faible satisfaction et appels ont un effet spécifique de renforcement.
- **Facturation :** la facture médiane est de 28,90 sur les positifs contre 60,79 sur les négatifs ;
  sa corrélation à la cible est d'environ −0,265. Ce résultat impose un audit de disponibilité avant
  toute utilisation prédictive ; une facture de clôture ou un prorata peut être postérieur à la décision.
- **Âge et géographie :** les 18–24 ans en IDF ont un taux de 5,48 % (584 lignes, 70 clients, 32 positifs),
  contre 4,36 % pour les 35–44 ans en IDF, qui regroupent pourtant 66 positifs. Volume et taux racontent
  deux choses différentes. Les ≥ 80 ans n'ont que 12 clients au total ; leurs cellules régionales sont masquées.
- **Temps :** l'association des appels au support est visible sur les 12 snapshots ; les contrats mensuels
  ont chaque mois le taux le plus élevé. Les écarts d'âge sont moins stables. La standardisation contractuelle
  ne supprime pas la variation temporelle (environ 2,39–3,60 %), sans démontrer une saisonnalité.
- **Trajectoires :** aucun client n'est observé à une date ultérieure à son premier positif. La règle de sortie
  du panel et l'horizon du label doivent être expliqués avant de définir l'échantillonnage et la validation.

## B. Hypothèses issues de l'EDA

| ID | Observation calculée | Hypothèse à tester, non démontrée | Variables concernées | Vérification future |
|---|---|---|---|---|
| H01 | Mensuel normalisé : 12,44 % avant 3 mois, 3,70 % après 48 mois. | L'ancienneté et le contexte contractuel apportent une information conjointe. | `contract`, `signup_date`, `snapshot_date` | Confirmer les dates et comparer les représentations hors temps. |
| H02 | ≥ 3 appels et ≥ 2 retards : 13,98 %, 58 clients positifs. | La co-occurrence des incidents peut apporter une information conditionnelle. | `support_calls_3m`, `late_payments` | Confirmer les fenêtres, examiner les strates, puis validation future sur train et période séparée. |
| H03 | Abonnement ≥ 80 : 5,37 % avec usage < 2, contre 2,64 % avec usage ≥ 10. | Tarif et usage pourraient refléter une inadéquation perçue, ou une population différente. | `monthly_fee`, `data_gb`, `contract` | Vérifier unités, anomalies et confusion par contrat/ancienneté ; comparaison prédictive ultérieure. |
| H04 | Même avec NPS ≥ 9, ≥ 3 appels : 7,48 % avec un IC large. | Le signal de support ne se réduit peut-être pas à la satisfaction déclarée. | `nps`, `support_calls_3m` | Auditer le déclenchement de l'enquête et comparer les interactions sur répondants/non-répondants. |
| H05 | NPS manquant : 2,87 %, renseigné : 3,29 %. | La disponibilité reflète un processus de collecte ou une population sélectionnée. | `nps`, `snapshot_date` | Contrôler la collecte et sa disponibilité, puis tester des stratégies de manquants dans le train seulement. |
| H06 | Dernière facture médiane beaucoup plus basse sur les positifs. | Une partie du signal peut venir d'une information postérieure à la prédiction. | `last_invoice_amount`, `monthly_fee`, `churn` | Obtenir les horodatages, distinguer facture normale/clôture et reconstituer une vue as-of. |
| H07 | En IDF, les 35–44 ans concentrent plus de positifs mais ont un taux inférieur aux 18–24 ans. | La composition des populations explique une partie des écarts régionaux. | `age`, `region`, `contract`, ancienneté analytique | Standardisations descriptives additionnelles et audit de représentativité, sans règle de ciblage par âge. |
| H08 | Standardisation contractuelle : variation toujours présente ; profils d'âge non constants. | Les écarts temporels ne proviennent pas uniquement du mix contractuel. | `snapshot_date`, `contract`, support, âge | Vérifier collecte, panel et stabilité sur de nouvelles périodes. |
| H09 | Pas de suivi après le premier positif ; 9,45 % en moyenne des taux clients contre 3,01 % par ligne. | La fin de suivi et les durées d'exposition contribuent aux associations. | `customer_id`, `snapshot_date`, `churn` | Documenter construction du panel, horizon et censure ; définir ensuite la validation. |
| H10 | `data_gb` : médianes 2,8/3,7, moyennes 36,91/72,89 et maximum 32 154. | Une queue extrême ou une unité incohérente masque des relations d'usage. | `data_gb` | Confirmer la mesure avant toute transformation ; analyse de sensibilité après validation métier. |

## C. Recommandations pour le futur Feature Engineering — non implémentées

| Feature / représentation candidate | Hypothèse EDA | Justification | Risque / condition préalable | Priorité |
|---|---|---|---|---|
| Ancienneté continue au snapshot, avec représentation non linéaire à comparer | H01 | Gradient observé sur plusieurs bandes. | Deux dates disponibles, durées négatives et sélection du panel. | Haute après validation des dates |
| Contrat normalisé et interaction éventuelle avec la durée | H01 | Profil conservé après regroupement des formats. | Mapping validé, valeurs nouvelles et effet de composition. | Haute après validation métier |
| Représentation conjointe support/paiements | H02 | Association du cumul et volume de 58 clients positifs. | Fenêtres strictement antérieures, absence de preuve de gain incrémental. | Moyenne, conditionnelle |
| Représentation conjointe montant/usage | H03, H10 | Contrastes conditionnels et usage asymétrique. | Unités, anomalies, influence du contrat ; aucun ratio proposé automatiquement. | Moyenne après clarification |
| Stratégie de manquants NPS et interaction éventuelle avec support | H04, H05 | Signal sur répondants et non-répondants. | Collecte non aléatoire possible, notes indisponibles au snapshot. | Moyenne après audit de collecte |
| Transformation robuste de l'usage, à comparer au brut | H10 | Forte asymétrie, extrêmes influents. | Ne pas cacher des erreurs de mesure ; paramètres futurs ajustés sur train uniquement. | Moyenne après validation des unités |
| Variable de facture ou écart facture/abonnement | H06 | Association observée mais temporellement suspecte. | **Bloquée avant preuve de disponibilité** ; pas de feature `invoice_fee_gap` créée. | En attente, non recommandée immédiatement |

Ni `customer_id` ni les regroupements âge/région ne sont recommandés comme règles de ciblage.
Les seuils des graphiques sont exploratoires et ne sont pas automatiquement des seuils de features ou de campagne.
Aucun tableau de cette section n'est un résultat de feature selection ni une preuve de gain de performance.

## D. Variables et conditions nécessitant une vigilance particulière

| Sujet | Éléments observés / inconnus | Conséquence |
|---|---|---|
| Label et panel | Horizon non documenté, fin du suivi après premier positif. | Définir événement, as-of, censure et période avant modélisation. |
| Offre de rétention | Instant de l'offre et du refus absent des documents. | `retention_offer_declined` reste exclue des relations explicatives. |
| Dernière facture | Fort contraste de distribution, date de disponibilité inconnue. | Analyse descriptive autorisée, usage futur suspendu. |
| NPS | 67,84 % manquants, date et déclencheur de collecte inconnus. | Limiter l'extrapolation et confirmer l'accès en production. |
| Support / paiements | Fenêtres et horodatages réels inconnus. | Le nom d'une colonne ne prouve pas son antériorité. |
| Catégories | 1 189 formats de contrat et 1 139 de canal regroupés dans la copie. | Mapping à valider et prévoir les modalités non vues. |
| Montants et usage | 113 frais négatifs ; unité et extrêmes de consommation à confirmer. | Pas de correction ou suppression automatique. |
| Redondance | Corrélation abonnement/facture ≈ 0,88. | Comparaison future, pas de suppression sur ce seul critère. |
| Ancienneté | 173 durées négatives. | Exclusions par vue explicites, règle métier à définir. |
| Âge / région | Très petits nombres de clients dans certaines cellules. | Pas de classement robuste des cellules masquées, pas de conclusion de biais du futur modèle. |

## E. Priorisation des prochaines expérimentations

1. **Prérequis bloquants :** définir le label, son horizon et les instants de disponibilité ; expliquer la
   sortie du panel. Sans cela, la validation ultérieure pourrait mesurer un signal indisponible en production.
2. **Travaux simples, plausibles et traçables :** valider le contrat et les dates pour étudier l'ancienneté
   et les catégories. Les observations sont nettes, la construction est simple, mais le gain reste à tester.
3. **Interactions conditionnelles :** support/paiements, puis montant/usage ; leur pertinence métier et
   les effectifs motivent des expériences futures, après clarification des fenêtres et unités.
4. **Collecte et manquants :** NPS/support et disponibilité ; les limites de collecte rendent leur priorité
   dépendante de la qualité des informations accessibles au moment de la prédiction.
5. **Variables temporellement suspectes :** facture et refus d'offre restent suspendus. Un signal exploratoire
   fort ne compense pas un risque de fuite non résolu.

L'ordre combine force descriptive, pertinence métier, disponibilité, fuite, complexité et possibilité de
validation. Il ne prédit pas l'amélioration d'un score de modèle. Les expériences auront lieu dans un
notebook distinct, avec une période de validation tenue à l'écart et des paramètres appris sur train seulement.

## F. Questions métier encore ouvertes

- `churn` désigne-t-il une décision, une résiliation effective ou un événement dans une fenêtre future ?
- Quel est son horizon ? Comment le label est-il calculé à chaque snapshot ?
- Pourquoi les clients disparaissent-ils après le premier positif ? Peuvent-ils revenir ?
- Les dates de `signup_date` et `snapshot_date` sont-elles historisées et disponibles as-of ?
- Quand l'offre de rétention et son refus sont-ils enregistrés par rapport au moment de prédiction ?
- La dernière facture est-elle une facture de clôture/prorata, et quand devient-elle accessible ?
- La fenêtre de `late_payments` est-elle fixe, cumulative ou glissante ?
- Les appels « 3m » sont-ils strictement antérieurs au snapshot, avec quel délai de consolidation ?
- Quand et pourquoi le NPS est-il collecté ? La disponibilité actuelle correspond-elle à celle en production ?
- Quelles unités et périodes couvrent montant et consommation ? Comment expliquer les extrêmes ?
- Les mappings contrat/canal et les bornes d'âge sont-ils validés ? Que représentent les codes `dept` ?

## Couverture finale et limites

Les **19 axes initiaux sont conservés**, avec approfondissement des axes 3–15 et 18 lorsque nécessaire ;
les six ajouts portent le total à **25 questions**, sans compter deux fois heatmap/courbes ou histogramme/boxplot.
L'[audit détaillé](../docs/eda_audit.md) classe chaque axe et documente les dénominateurs et limites.
`dept` reste sans étude géographique fine faute de dictionnaire ; l'offre de rétention reste à auditer,
et la phase pilote est différée. Cette couverture est large, pas exhaustive de tous les croisements possibles.

**Ce qui a été produit :** statistiques, vues temporaires, graphiques, hypothèses et recommandations.
**Ce qui n'a pas été commencé :** pipeline de features, preprocessing d'entraînement, sélection de features,
modèle, tuning, optimisation de campagne ou analyse causale du pilote.


### Export optionnel des résultats EDA

Les exports sont des rapports statistiques agrégés, pas un dataset de features. Les données sources,
le pilote et les modules de préparation de features ou de modélisation restent inchangés.

