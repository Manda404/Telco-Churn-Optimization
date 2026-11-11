"""Vues temporaires et diagnostics statistiques EDA ; aucune feature persistante."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from telco_churn.data.exploration import ChurnExplorer, ExplorationConfig, numeric_bands
from telco_churn.data.schema import DatasetRules, EDAOptions

QUALITY_GROUPS = {"Non renseigné", "Hors domaine / invalide"}


def analytical_view(
    df: pd.DataFrame,
    config: ExplorationConfig,
    rules: DatasetRules,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copier la source et documenter chaque normalisation ou segmentation EDA.

    Les noms eda_* ne sont ni exportés comme dataset de features ni utilisés pour
    entraîner un modèle. Les catégories inconnues restent visibles, sans rapprochement flou.
    """
    if any(str(column).startswith("eda_") for column in df):
        raise ValueError("La source passée doit être brute, sans colonnes eda_* préexistantes.")
    view = df.copy(deep=True)
    audit = []
    for column, allowed in rules.categories.items():
        if column not in df:
            raise ValueError(f"Colonne catégorielle absente : {column}")
        raw = df[column].astype("string")
        folded = raw.str.strip().str.casefold().replace("", pd.NA)
        mapping = {value.strip().casefold(): value for value in allowed}
        normalized = folded.map(mapping).fillna(folded)
        view[f"eda_{column}"] = normalized
        audit.append(
            {
                "column": column,
                "operation": "espaces externes et casse",
                "n_changed": int((raw.notna() & raw.ne(normalized).fillna(True)).sum()),
                "n_missing": int(normalized.isna().sum()),
                "n_invalid": int((normalized.notna() & ~normalized.isin(allowed)).sum()),
            }
        )
    bands = {**config.deep_eda.interaction_bins, config.age_column: config.age_bins}
    for column, edges in bands.items():
        if column not in df or column not in rules.numeric:
            raise ValueError(f"Variable ou domaine numérique absent : {column}")
        result = numeric_bands(df[column], edges, rules.numeric[column])
        view[f"eda_{column}_band"] = result
        audit.append(
            {
                "column": column,
                "operation": "bandes temporaires du YAML",
                "n_changed": pd.NA,
                "n_missing": int(result.eq("Non renseigné").sum()),
                "n_invalid": int(result.eq("Hors domaine / invalide").sum()),
            }
        )
    logger.bind(pipeline="eda.diagnostics").info(
        "EDA copy: rows={} analytical_columns={}", len(view), len(view.columns) - len(df.columns)
    )
    return view, pd.DataFrame(audit)


def eligible_view(
    view: pd.DataFrame,
    columns: list[str],
    *,
    keep_missing: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Restreindre un croisement, en conservant explicitement les manquants demandés.

    Les domaines doivent avoir été vérifiés/encodés par analytical_view. L'audit
    compte les lignes exclues sans les additionner par colonne (motifs recouvrants).
    """
    keep_missing = keep_missing or []
    mask = pd.Series(True, index=view.index)
    for column in columns:
        if column not in view:
            raise ValueError(f"Colonne absente : {column}")
        if column in keep_missing:
            mask &= ~view[column].astype("string").eq("Hors domaine / invalide").fillna(True)
        else:
            mask &= view[column].notna() & ~view[column].astype("string").isin(QUALITY_GROUPS)
    return view.loc[mask].copy(), {
        "n_input": len(view),
        "n_eligible": int(mask.sum()),
        "n_excluded": int((~mask).sum()),
    }


def clustered_rates(
    explorer: ChurnExplorer,
    df: pd.DataFrame,
    columns: list[str],
    options: EDAOptions,
) -> pd.DataFrame:
    """Taux par snapshot et IC percentile par bootstrap de clients entiers.

    Chaque tirage reprend les historiques des clients sélectionnés, avec remise.
    Le taux reste pondéré par les lignes ; la dépendance intra-client est préservée.
    L'IC suppose des clients approximativement indépendants et ne corrige pas
    sélection du panel, confusion, multiplicité des comparaisons ou biais de collecte.
    """
    observed = explorer._observed(df)
    explorer._require_columns(observed, [explorer.id_column, *columns])
    if observed[explorer.id_column].isna().any():
        raise ValueError("Les identifiants manquants empêchent le bootstrap par client.")
    table = explorer.interaction_rates(observed, columns)
    table["small_group"] |= table.n_customers.lt(options.min_clients)
    table["ci_low_percent"] = np.nan
    table["ci_high_percent"] = np.nan
    table["ci_status"] = "support insuffisant"
    if observed.empty:
        return table
    group_index = pd.MultiIndex.from_frame(table[columns])
    row_group = group_index.get_indexer(pd.MultiIndex.from_frame(observed[columns]))
    if (row_group < 0).any():
        raise ValueError("Alignement des groupes impossible pour le bootstrap.")
    customer_code, customers = pd.factorize(observed[explorer.id_column], sort=True)
    denominators = np.zeros((len(customers), len(table)))
    numerators = np.zeros_like(denominators)
    np.add.at(denominators, (customer_code, row_group), 1)
    np.add.at(
        numerators,
        (customer_code, row_group),
        observed[explorer.target_column].to_numpy(dtype=float),
    )
    rng = np.random.default_rng(options.random_seed)
    weights = rng.multinomial(
        len(customers),
        np.full(len(customers), 1 / len(customers)),
        size=options.bootstrap_repetitions,
    ).astype(float)
    boot_den = weights @ denominators
    boot_num = weights @ numerators
    rates = np.divide(
        100 * boot_num, boot_den, out=np.full_like(boot_den, np.nan), where=boot_den > 0
    )
    alpha = (1 - options.confidence_level) / 2
    eligible = (
        ~table.small_group
        & table.n_events.ge(options.min_events_ci)
        & table.n_positive_customers.ge(options.min_events_ci)
    )
    for column in table.index[eligible]:
        finite = rates[:, column][np.isfinite(rates[:, column])]
        if len(finite) >= 0.9 * options.bootstrap_repetitions:
            table.loc[column, ["ci_low_percent", "ci_high_percent"]] = np.quantile(
                finite, [alpha, 1 - alpha]
            )
            table.loc[column, "ci_status"] = "bootstrap clients"
    return table


def regional_composition(
    view: pd.DataFrame, age_column: str, region_column: str, id_column: str
) -> pd.DataFrame:
    """Compter toutes les lignes éligibles et les clients, indépendamment du label.

    Les pourcentages se rapportent aux snapshots, pas aux clients distincts. Les
    comptes clients par cellule ne s'additionnent pas si une personne change de groupe.
    """
    table = (
        view.groupby([age_column, region_column], observed=True, dropna=False)
        .agg(
            n_rows=(id_column, "size"),
            n_customers=(id_column, "nunique"),
        )
        .reset_index()
    )
    table["share_within_age_percent"] = (
        table.n_rows / table.groupby(age_column, observed=True).n_rows.transform("sum") * 100
    )
    table["share_within_region_percent"] = (
        table.n_rows / table.groupby(region_column, observed=True).n_rows.transform("sum") * 100
    )
    return table


def standardized_monthly_rates(
    explorer: ChurnExplorer,
    df: pd.DataFrame,
    date_column: str,
    group_column: str,
    date_format: str,
    min_clients: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Comparer les taux mensuels bruts et à composition contractuelle fixe.

    Les poids décrivent toute la période EDA, pas un estimateur appris pour un
    modèle. Un mois sans support pour une strate garde un taux standardisé NA.
    Cette standardisation à une variable ne corrige pas tous les facteurs de confusion.
    """
    view = explorer._observed(df)
    view[date_column] = pd.to_datetime(view[date_column], format=date_format, errors="coerce")
    if view[date_column].isna().any():
        raise ValueError("Dates invalides : standardisation temporelle impossible.")
    strata = explorer.interaction_rates(view, [date_column, group_column])
    strata["small_group"] |= strata.n_customers.lt(min_clients)
    weights = (
        view.groupby(group_column, observed=True).size() / len(view)
        if len(view)
        else pd.Series(dtype=float)
    )
    rates = strata.pivot(
        index=date_column, columns=group_column, values="churn_rate_percent"
    ).reindex(columns=weights.index)
    small = strata.pivot(
        index=date_column, columns=group_column, values="small_group"
    ).reindex_like(rates)
    rates = rates.mask(small.astype("boolean").fillna(True))
    standardized = (
        rates.mul(weights, axis=1).sum(axis=1, min_count=len(weights))
        if len(weights)
        else pd.Series(dtype=float)
    )
    monthly = explorer.temporal_churn(view, date_column, date_format)
    monthly["standardized_rate_percent"] = monthly[date_column].map(standardized)
    return monthly, weights.rename("reference_weight").reset_index()
