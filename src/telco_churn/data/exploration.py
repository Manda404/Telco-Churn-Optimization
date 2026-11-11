"""Statistiques descriptives du churn, sans transformation des sources."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from telco_churn.data.schema import EDAOptions, NumericRule
from telco_churn.settings import Settings, _read_yaml


def _validate_edges(edges: list[float]) -> None:
    """Les bornes définissent des intervalles gauches fermés, le dernier est ouvert à +inf."""
    if (
        len(edges) < 2
        or not np.isfinite(edges).all()
        or any(a >= b for a, b in zip(edges, edges[1:], strict=False))
    ):
        raise ValueError(
            "Les bornes doivent être finies, strictement croissantes et au moins deux."
        )


def numeric_bands(series: pd.Series, edges: list[float], rule: NumericRule) -> pd.Series:
    """Segmenter une vue avec deux modalités explicites pour les manquants et invalides.

    Les domaines proviennent du contrat qualité. Les intervalles sont [a, b),
    le dernier est [a, +inf). Les valeurs hors domaine ne disparaissent pas.
    """
    _validate_edges(edges)
    numeric = pd.to_numeric(series, errors="coerce")
    missing = series.isna() | series.astype("string").str.strip().eq("").fillna(False)
    valid = numeric.notna() & np.isfinite(numeric) & numeric.ge(edges[0])
    if rule.minimum is not None:
        valid &= numeric.ge(rule.minimum)
    if rule.maximum is not None:
        valid &= numeric.le(rule.maximum)
    if rule.integer:
        valid &= numeric.where(np.isfinite(numeric)).mod(1).eq(0)
    labels = [f"[{a:g}, {b:g})" for a, b in zip(edges, edges[1:], strict=False)] + [
        f"≥ {edges[-1]:g}"
    ]
    bands = pd.cut(numeric.where(valid), [*edges, np.inf], labels=labels, right=False).astype(
        "string"
    )
    bands = bands.mask(~valid & ~missing, "Hors domaine / invalide").mask(missing, "Non renseigné")
    return pd.Series(
        pd.Categorical(
            bands, categories=[*labels, "Non renseigné", "Hors domaine / invalide"], ordered=True
        ),
        index=series.index,
    )


class ExplorationConfig(BaseModel):
    """Paramètres de l'EDA ; les noms des cibles restent dans Settings."""

    model_config = ConfigDict(extra="forbid")
    id_column: str = Field(min_length=1)
    snapshot_column: str = Field(min_length=1)
    snapshot_format: str
    categorical_columns: list[str] = Field(min_length=1)
    numeric_columns: list[str] = Field(min_length=1)
    comparison_columns: list[str] = Field(min_length=1)
    min_group_size: int = Field(ge=1)
    correlation_min_pairs: int = Field(ge=2)
    boxplot_showfliers: bool = False
    signup_column: str = Field(min_length=1)
    tenure_column: str = Field(min_length=1)
    tenure_band_column: str = Field(min_length=1)
    days_per_month: float = Field(gt=0, allow_inf_nan=False)
    tenure_bins: list[float] = Field(min_length=2)
    diagnostic_bands: dict[str, list[float]]
    interaction_column: str = Field(min_length=1)
    age_column: str = Field(min_length=1)
    age_bins: list[float] = Field(min_length=2)
    deep_eda: EDAOptions

    @model_validator(mode="after")
    def check_columns(self) -> ExplorationConfig:
        if self.id_column in self.numeric_columns:
            raise ValueError("L'identifiant ne doit pas figurer dans les variables numériques.")
        if set(self.categorical_columns) & set(self.numeric_columns):
            raise ValueError("Les listes de variables numériques et catégorielles se recouvrent.")
        for columns in (self.categorical_columns, self.numeric_columns, self.comparison_columns):
            if len(columns) != len(set(columns)) or any(not column.strip() for column in columns):
                raise ValueError("Les noms de colonnes doivent être uniques et non vides.")
        if self.interaction_column not in self.categorical_columns:
            raise ValueError(
                "La variable d'interaction doit figurer parmi les variables catégorielles."
            )
        if not set(self.diagnostic_bands).issubset(self.numeric_columns):
            raise ValueError(
                "Les variables de seuils doivent figurer parmi les variables numériques."
            )
        required_pairs = {"support_payment", "fee_usage", "satisfaction_support"}
        if not required_pairs.issubset(self.deep_eda.interaction_pairs):
            raise ValueError("Les trois interactions EDA attendues doivent être configurées.")
        for pair in self.deep_eda.interaction_pairs.values():
            if len(set(pair)) != 2 or not set(pair).issubset(self.deep_eda.interaction_bins):
                raise ValueError("Chaque interaction doit référencer deux variables avec des bornes EDA.")
        derived = [self.tenure_column, self.tenure_band_column]
        if len(set(derived)) != 2 or set(derived) & {
            self.id_column,
            self.signup_column,
            self.snapshot_column,
        }:
            raise ValueError(
                "Les noms des variables analytiques doivent être distincts des colonnes sources."
            )
        for edges in [
            self.tenure_bins,
            self.age_bins,
            *self.diagnostic_bands.values(),
            *self.deep_eda.interaction_bins.values(),
        ]:
            _validate_edges(edges)
        return self

    @classmethod
    def load(cls, settings: Settings) -> ExplorationConfig:
        """Charger le YAML d'exploration depuis son chemin configuré."""
        try:
            return cls.model_validate(_read_yaml(settings.eda_config_path))
        except ValueError as exc:
            raise ValueError(
                f"Configuration EDA invalide ({settings.eda_config_path}) : {exc}"
            ) from exc


class ChurnExplorer:
    """Calculer des associations descriptives au niveau des lignes observées.

    Les cibles absentes sont exclues des taux et comptées dans population_summary.
    Une cible renseignée différente de 0/1 déclenche une erreur explicite.
    Aucune méthode ne modifie le dataframe reçu.
    """

    def __init__(self, target_column: str, id_column: str, min_group_size: int = 100) -> None:
        if not target_column.strip() or not id_column.strip() or min_group_size < 1:
            raise ValueError("Cible et identifiant requis ; min_group_size doit être positif.")
        self.target_column = target_column
        self.id_column = id_column
        self.min_group_size = min_group_size

    @staticmethod
    def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
        if not df.columns.is_unique:
            raise ValueError("Les noms de colonnes doivent être uniques.")
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes absentes pour l'EDA : {sorted(missing)}")

    def _observed(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df, [self.target_column])
        target = df[self.target_column]
        if (target.notna() & ~target.isin([0, 1])).any():
            raise ValueError(f"La cible {self.target_column} doit contenir uniquement 0, 1 ou NA.")
        return df.loc[target.notna()].copy()

    def population_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Distinguer lignes, clients uniques, événements et disponibilité de la cible."""
        self._require_columns(df, [self.id_column])
        observed = self._observed(df)
        events = int(observed[self.target_column].sum())
        return {
            "n_rows": len(df),
            "n_customers": int(df[self.id_column].nunique()),
            "n_missing_id": int(df[self.id_column].isna().sum()),
            "n_target_observed": len(observed),
            "n_target_missing": len(df) - len(observed),
            "n_events": events,
            "churn_rate_percent": 100 * events / len(observed) if len(observed) else np.nan,
        }

    def churn_by_group(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Retourner le taux par modalité brute et signaler les petits groupes."""
        self._require_columns(df, [column])
        observed = self._observed(df)
        table = (
            observed.groupby(column, dropna=False, observed=True)[self.target_column]
            .agg(n_observed="size", n_events="sum")
            .reset_index()
        )
        table["churn_rate_percent"] = table["n_events"] / table["n_observed"] * 100
        table["small_group"] = table["n_observed"] < self.min_group_size
        return table.sort_values(
            "churn_rate_percent", ascending=False, kind="stable", ignore_index=True
        )

    def temporal_churn(self, df: pd.DataFrame, column: str, date_format: str) -> pd.DataFrame:
        """Regrouper par date observée, sans compléter des périodes artificiellement."""
        self._require_columns(df, [column, self.id_column])
        observed = self._observed(df)
        dates = pd.to_datetime(observed[column], format=date_format, errors="coerce")
        if dates.isna().any():
            raise ValueError(
                f"{int(dates.isna().sum())} dates absentes/invalides dans {column} : voir notebook 02."
            )
        observed[column] = dates
        table = (
            observed.groupby(column, observed=True)
            .agg(
                n_observed=(self.target_column, "size"),
                n_customers=(self.id_column, "nunique"),
                n_events=(self.target_column, "sum"),
            )
            .reset_index()
            .sort_values(column, ignore_index=True)
        )
        table["churn_rate_percent"] = table["n_events"] / table["n_observed"] * 100
        return table

    def numeric_values(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Copie analytique : nombres interprétables et finis, sans seuil métier ajouté.

        Les autres valeurs sont mises à NA dans cette vue seulement. Les valeurs
        négatives et extrêmes finies sont conservées ; les règles du notebook 02
        restent nécessaires pour les interpréter.
        """
        self._require_columns(df, columns)
        if self.target_column in columns or self.id_column in columns:
            raise ValueError("Cible et identifiant doivent être exclus de la liste numérique.")
        observed = self._observed(df)
        values = (
            observed[columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        values[self.target_column] = observed[self.target_column]
        return values

    def numeric_summary(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Quantiles et effectifs numériques par classe, avec exclusions explicites."""
        values = self.numeric_values(df, columns)
        rows = []
        for outcome, group in values.groupby(self.target_column, observed=True):
            for column in columns:
                finite = group[column].dropna()
                rows.append(
                    {
                        "column": column,
                        "target_value": outcome,
                        "n_observed": len(group),
                        "n_finite": len(finite),
                        "n_excluded": len(group) - len(finite),
                        "min": finite.min(),
                        "q25": finite.quantile(0.25),
                        "median": finite.median(),
                        "mean": finite.mean(),
                        "q75": finite.quantile(0.75),
                        "max": finite.max(),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "column",
                "target_value",
                "n_observed",
                "n_finite",
                "n_excluded",
                "min",
                "q25",
                "median",
                "mean",
                "q75",
                "max",
            ],
        )

    def correlations(
        self,
        df: pd.DataFrame,
        columns: list[str],
        min_pairs: int = 100,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Corrélations de Pearson par paires et effectifs associés, sans inférence."""
        if min_pairs < 2:
            raise ValueError("min_pairs doit être au moins égal à 2.")
        values = self.numeric_values(df, columns)
        present = values.notna().astype("int64")
        return values.corr(method="pearson", min_periods=min_pairs), present.T.dot(present)

    def category_composition(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Composition de toutes les lignes, indépendamment de la disponibilité de la cible."""
        self._require_columns(df, [column])
        counts = (
            df.groupby(column, dropna=False, observed=True).size().rename("n_rows").reset_index()
        )
        counts["share_percent"] = counts["n_rows"] / len(df) * 100 if len(df) else np.nan
        return counts

    def analytical_tenure_view(
        self,
        df: pd.DataFrame,
        config: ExplorationConfig,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Calculer une ancienneté approchée dans une copie, pas dans le pipeline de features."""
        self._require_columns(df, [config.signup_column, config.snapshot_column])
        self._observed(df)  # Vérification de la cible, sans exclure de lignes de cette vue.
        if {config.tenure_column, config.tenure_band_column} & set(df.columns):
            raise ValueError(
                "Une colonne analytique existe déjà ; choisir un autre nom dans le YAML."
            )
        view = df.copy(deep=True)
        signup = pd.to_datetime(
            view[config.signup_column], format=config.snapshot_format, errors="coerce"
        )
        snapshot = pd.to_datetime(
            view[config.snapshot_column], format=config.snapshot_format, errors="coerce"
        )
        tenure = (snapshot - signup).dt.total_seconds() / (86400 * config.days_per_month)
        view[config.tenure_column] = tenure
        view[config.tenure_band_column] = numeric_bands(
            tenure,
            config.tenure_bins,
            NumericRule(minimum=0, note="Ancienneté non négative attendue."),
        )
        audit = {
            "n_rows": len(view),
            "n_dates_absent_or_invalid": int(tenure.isna().sum()),
            "n_negative_tenure": int(tenure.lt(0).sum()),
            "n_valid_tenure": int(tenure.ge(0).sum()),
        }
        return view, audit

    def interaction_rates(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Effectifs, taux, écart et ratio descriptif par groupe ou croisement.

        Le taux de référence utilise toutes les lignes à cible observée de la vue
        passée. Les parts d'événements s'additionnent à 100 % si des positifs existent.
        Le ratio n'est ni un uplift causal ni un lift de modèle.
        """
        if not columns or len(set(columns)) != len(columns) or self.target_column in columns:
            raise ValueError("Définir des colonnes de regroupement distinctes de la cible.")
        self._require_columns(df, [*columns, self.id_column])
        observed = self._observed(df)
        table = (
            observed.groupby(columns, dropna=False, observed=True)
            .agg(
                n_observed=(self.target_column, "size"),
                n_customers=(self.id_column, "nunique"),
                n_events=(self.target_column, "sum"),
            )
            .reset_index()
        )
        positive_clients = (
            observed.loc[observed[self.target_column].eq(1)]
            .groupby(columns, dropna=False, observed=True)[self.id_column]
            .nunique()
            .rename("n_positive_customers")
            .reset_index()
        )
        table = table.merge(positive_clients, on=columns, how="left", validate="one_to_one")
        table["n_positive_customers"] = table["n_positive_customers"].fillna(0).astype(int)
        total_events = int(observed[self.target_column].sum())
        baseline = 100 * total_events / len(observed) if len(observed) else np.nan
        table["churn_rate_percent"] = 100 * table.n_events / table.n_observed
        table["event_share_percent"] = (
            100 * table.n_events / total_events if total_events else np.nan
        )
        table["delta_pp"] = table.churn_rate_percent - baseline
        table["rate_ratio"] = table.churn_rate_percent / baseline if baseline > 0 else np.nan
        table["small_group"] = table.n_observed < self.min_group_size
        return table

    def band_churn(
        self,
        df: pd.DataFrame,
        column: str,
        edges: list[float],
        rule: NumericRule,
    ) -> pd.DataFrame:
        """Évaluer des seuils descriptifs, sans modifier ou éliminer de valeur brute."""
        self._require_columns(df, [column])
        bands = numeric_bands(df[column], edges, rule)
        view = df.copy()
        temporary = "__eda_band__"
        if temporary in df:
            raise ValueError(f"La colonne réservée {temporary} est déjà présente.")
        view[temporary] = bands
        return self.interaction_rates(view, [temporary]).rename(columns={temporary: "band"})

    def missingness_churn(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Comparer les cibles selon la disponibilité, pas la valeur, des variables."""
        self._require_columns(df, columns)
        tables = []
        for column in columns:
            missing = df[column].isna() | df[column].astype("string").str.strip().eq("").fillna(
                False
            )
            view = df[[self.id_column, self.target_column]].copy()
            view["availability"] = np.where(missing, "Non renseigné", "Renseigné")
            table = self.interaction_rates(view, ["availability"])
            table.insert(0, "variable", column)
            tables.append(table)
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    def history_diagnostics(
        self,
        df: pd.DataFrame,
        date_column: str,
        date_format: str,
    ) -> tuple[pd.DataFrame, dict[str, int | float]]:
        """Examiner les trajectoires sur des dates distinctes, sans inférer des résiliations.

        Les copies à la même date ne créent pas de faux événements répétés. Des
        snapshots après un positif demandent de clarifier la définition de la cible.
        Les cibles manquantes sont exclues, comme dans les autres taux.
        """
        self._require_columns(df, [self.id_column, date_column])
        observed = self._observed(df).reset_index(drop=True)
        dates = pd.to_datetime(observed[date_column], format=date_format, errors="coerce")
        if dates.isna().any() or observed[self.id_column].isna().any():
            raise ValueError("Dates ou identifiants absents/invalides : revoir le notebook 02.")
        observed[date_column] = dates
        history = observed.groupby(self.id_column).agg(
            n_observed=(self.target_column, "size"),
            n_dates=(date_column, "nunique"),
        )
        positives = observed.loc[observed[self.target_column].eq(1)]
        positive_dates = positives.groupby(self.id_column)[date_column]
        history["n_positive_dates"] = positive_dates.nunique().reindex(history.index, fill_value=0)
        first_positive = observed[self.id_column].map(positive_dates.min())
        after = observed.loc[observed[date_column].gt(first_positive)]
        history["n_dates_after_first_positive"] = (
            after.groupby(self.id_column)[date_column]
            .nunique()
            .reindex(history.index, fill_value=0)
        )
        conflicts = (
            observed.groupby([self.id_column, date_column])[self.target_column].nunique().gt(1)
        )
        summary = {
            "n_customers_with_observed_target": len(history),
            "n_customers_positive_at_least_once": int(history.n_positive_dates.gt(0).sum()),
            "n_customers_positive_on_multiple_dates": int(history.n_positive_dates.gt(1).sum()),
            "n_customers_observed_after_first_positive": int(
                history.n_dates_after_first_positive.gt(0).sum()
            ),
            "n_customer_date_target_conflicts": int(conflicts.sum()),
            "mean_client_snapshot_churn_percent": float(
                observed.groupby(self.id_column)[self.target_column].mean().mean() * 100
            ),
        }
        return history.reset_index(), summary
