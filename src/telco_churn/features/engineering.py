"""Features déterministes par ligne ; aucune imputation ou statistique apprise."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from telco_churn.data.schema import DatasetRules
from telco_churn.features.config import FeatureConfig


class FeatureEngineer:
    """Construire une matrice candidate, séparée de la cible et des métadonnées.

    Une disponibilité déclarée dans le YAML n'est pas une preuve métier. Le
    catalogue expose les confirmations manquantes ; aucune méthode ne choisit
    des variables selon la cible et transform() fonctionne sans colonne cible.
    """

    def __init__(self, config: FeatureConfig, rules: DatasetRules, target_column: str) -> None:
        self.config = config
        self.rules = rules
        self.target_column = target_column
        sources = set(config.numeric_columns) | set(config.categorical_columns)
        if config.age_grouping:
            sources.add(config.age_column)
            if config.age_column not in rules.numeric:
                raise ValueError(
                    "Le regroupement d'âge nécessite un domaine numérique dans validation.yaml."
                )
        if target_column in sources:
            raise ValueError("La cible ne peut pas être une source de features.")
        if not set(config.numeric_columns).issubset(rules.numeric):
            raise ValueError(
                "Chaque variable numérique doit avoir un domaine dans validation.yaml."
            )
        if not set(config.categorical_columns).issubset(rules.categories):
            raise ValueError("Chaque catégorie doit avoir un référentiel dans validation.yaml.")
        if any(config.unknown_category in values for values in rules.categories.values()):
            raise ValueError(
                "Le libellé de catégorie inconnue ne doit pas être une catégorie canonique."
            )

    @staticmethod
    def _missing(series: pd.Series) -> pd.Series:
        return series.isna() | series.astype("string").str.strip().eq("").fillna(False)

    def _numeric(self, series: pd.Series, column: str) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Parser les nombres ; séparer absence et valeur renseignée invalide."""
        missing = self._missing(series)
        values = pd.to_numeric(series, errors="coerce").astype(float)
        rule = self.rules.numeric[column]
        valid = values.notna() & np.isfinite(values)
        valid &= ~series.map(lambda value: isinstance(value, (bool, np.bool_)))
        if rule.minimum is not None:
            valid &= values.ge(rule.minimum)
        if rule.maximum is not None:
            valid &= values.le(rule.maximum)
        if rule.integer:
            valid &= values.where(np.isfinite(values)).mod(1).eq(0)
        invalid = ~missing & ~valid
        return values.where(valid), missing, invalid

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retourner les features candidates en préservant lignes, ordre et index.

        Les valeurs hors domaine deviennent NA ; les catégories inconnues utilisent
        un libellé réservé. Les copies exactes restent présentes : leur politique
        relève de la préparation des échantillons, pas de ce transformateur.
        """
        c = self.config
        required = (
            set(c.numeric_columns)
            | set(c.categorical_columns)
            | {c.snapshot_column, c.signup_column}
        )
        if c.age_grouping:
            required.add(c.age_column)
        if not df.columns.is_unique:
            raise ValueError("Les noms de colonnes doivent être uniques.")
        if not required.issubset(df.columns):
            raise ValueError(f"Sources manquantes : {sorted(required - set(df.columns))}")
        result = pd.DataFrame(index=df.index)
        for column in c.numeric_columns:
            values, missing, invalid = self._numeric(df[column], column)
            result[column] = values
            if column in c.indicator_columns:
                result[f"{column}_missing"] = missing.astype("uint8")
                result[f"{column}_invalid"] = invalid.astype("uint8")
        for column in c.categorical_columns:
            raw = df[column].astype("string")
            folded = raw.str.strip().str.casefold().mask(self._missing(raw))
            mapping = {value.strip().casefold(): value for value in self.rules.categories[column]}
            normalized = folded.map(mapping).astype("string")
            result[column] = normalized.mask(folded.notna() & normalized.isna(), c.unknown_category)
        if c.age_grouping:
            age, _, _ = self._numeric(df[c.age_column], c.age_column)
            edges = c.age_bins
            labels = [f"[{a:g}, {b:g})" for a, b in zip(edges, edges[1:], strict=False)] + [
                f"≥ {edges[-1]:g}"
            ]
            result["age_group"] = pd.cut(age, [*edges, np.inf], right=False, labels=labels).astype(
                "string"
            )
        snapshot = pd.to_datetime(df[c.snapshot_column], format=c.date_format, errors="coerce")
        signup = pd.to_datetime(df[c.signup_column], format=c.date_format, errors="coerce")
        tenure = (snapshot - signup).dt.total_seconds() / (86400 * c.days_per_month)
        result["tenure_months"] = tenure.where(tenure.ge(0))
        result["tenure_unavailable"] = result.tenure_months.isna().astype("uint8")
        if c.log_tenure:
            result["log1p_tenure_months"] = np.log1p(result.tenure_months)
        if c.log_usage:
            if result[c.usage_column].dropna().lt(0).any():
                raise ValueError("log1p de l'usage nécessite un domaine non négatif.")
            result[f"log1p_{c.usage_column}"] = np.log1p(result[c.usage_column])
        if c.support_payment_interaction:
            result["support_payment_interaction"] = (
                result[c.support_column] * result[c.payments_column]
            )
        if np.isinf(result.select_dtypes(include="number").to_numpy()).any():
            raise ValueError(
                "Une dérivation produit des valeurs infinies : vérifier les domaines numériques."
            )
        blocked = set(c.excluded_columns) | {
            self.target_column,
            c.id_column,
            c.snapshot_column,
            c.signup_column,
        }
        if blocked & set(result.columns):
            raise ValueError(
                f"Colonnes exclues présentes dans la matrice : {sorted(blocked & set(result.columns))}"
            )
        if list(result.columns) != self.catalog().feature.tolist():
            # Le catalogue est l'ordre de référence, stable au train comme au scoring.
            result = result.loc[:, self.catalog().feature.tolist()]
        result.attrs["status"] = "candidate_features_not_training_validated"
        logger.bind(pipeline="features.engineering").info(
            "Candidate features: rows={} columns={} missing_cells={}",
            len(result),
            len(result.columns),
            int(result.isna().sum().sum()),
        )
        return result

    def catalog(self) -> pd.DataFrame:
        """Dictionnaire de chaque feature, formule, sources et disponibilité déclarée."""
        c = self.config
        rows = []

        def add(name: str, sources: list[str], formula: str) -> None:
            unconfirmed = [
                source
                for source in sources
                if c.source_availability[source] != "available_at_snapshot"
            ]
            rows.append(
                {
                    "feature": name,
                    "sources": ", ".join(sources),
                    "formula": formula,
                    "hypotheses": "; ".join(
                        dict.fromkeys(c.hypotheses[source] for source in sources)
                    ),
                    "availability": "unconfirmed" if unconfirmed else "declared_available",
                    "unconfirmed_sources": ", ".join(unconfirmed),
                }
            )

        for column in c.numeric_columns:
            add(column, [column], "Conversion numérique ; valeurs hors domaine/infinies → NA")
            if column in c.indicator_columns:
                add(
                    f"{column}_missing",
                    [column],
                    "1 si valeur brute absente ou chaîne vide ; 0 sinon",
                )
                add(
                    f"{column}_invalid",
                    [column],
                    "1 si valeur renseignée non numérique ou hors domaine ; 0 sinon",
                )
        for column in c.categorical_columns:
            add(
                column,
                [column],
                "Espaces externes/casse → libellé canonique ; inconnu réservé ; manquant conservé",
            )
        if c.age_grouping:
            add(
                "age_group",
                [c.age_column],
                f"Tranches [a, b), dernière ouverte ; bornes {c.age_bins} ; hors domaine/manquant → NA",
            )
        dates = [c.snapshot_column, c.signup_column]
        add(
            "tenure_months",
            dates,
            f"Durée en jours / {c.days_per_month} ; chronologie invalide → NA",
        )
        add("tenure_unavailable", dates, "1 si dates absentes/invalides ou ancienneté négative")
        if c.log_tenure:
            add("log1p_tenure_months", dates, "log(1 + tenure_months), sans apprentissage")
        if c.log_usage:
            add(
                f"log1p_{c.usage_column}",
                [c.usage_column],
                "log(1 + usage valide), sans apprentissage",
            )
        if c.support_payment_interaction:
            add(
                "support_payment_interaction",
                [c.support_column, c.payments_column],
                "Produit des deux compteurs valides ; NA propagés",
            )
        return pd.DataFrame(rows)

    def quality_summary(self, features: pd.DataFrame) -> pd.DataFrame:
        """Vérifier disponibilité des valeurs, types, cardinalité et catégories inconnues."""
        expected = self.catalog().feature.tolist()
        if list(features.columns) != expected:
            raise ValueError("Le schéma reçu diffère de celui du catalogue des features.")
        return pd.DataFrame(
            {
                "dtype": features.dtypes.astype(str),
                "n_missing": features.isna().sum(),
                "missing_percent": features.isna().mean() * 100,
                "n_unique": features.nunique(dropna=True),
                "n_unknown_category": [
                    int(features[column].eq(self.config.unknown_category).sum())
                    if column in self.config.categorical_columns
                    else 0
                    for column in expected
                ],
            }
        ).rename_axis("feature")

    def assert_temporal_availability(self) -> None:
        """Refuser une promotion silencieuse des candidats aux sources non confirmées.

        Ce contrôle ne valide ni la définition du label ni le futur découpage.
        La confirmation métier et le protocole temporel restent distincts.
        """
        pending = self.catalog().loc[lambda table: table.availability.eq("unconfirmed"), "feature"]
        if len(pending):
            raise ValueError(f"Disponibilité temporelle non confirmée pour : {', '.join(pending)}")
