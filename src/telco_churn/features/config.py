"""Contrat explicite du premier jeu de features candidates."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from telco_churn.settings import Settings, _read_yaml


class FeatureConfig(BaseModel):
    """Paramètres déterministes ; aucune statistique apprise sur la population."""

    model_config = ConfigDict(extra="forbid")
    id_column: str
    snapshot_column: str
    signup_column: str
    date_format: str
    days_per_month: float = Field(gt=0, allow_inf_nan=False)
    numeric_columns: list[str] = Field(min_length=1)
    categorical_columns: list[str] = Field(min_length=1)
    indicator_columns: list[str]
    excluded_columns: list[str]
    unknown_category: str = Field(min_length=1)
    source_availability: dict[str, Literal["unconfirmed", "available_at_snapshot"]]
    log_tenure: bool = True
    log_usage: bool = False
    usage_column: str
    support_payment_interaction: bool = True
    support_column: str
    payments_column: str
    hypotheses: dict[str, str]
    age_grouping: bool = True
    age_column: str = Field(default="age", min_length=1)
    age_bins: list[float] = Field(min_length=2)

    @model_validator(mode="after")
    def check_references(self) -> FeatureConfig:
        selected = [*self.numeric_columns, *self.categorical_columns]
        if len(selected) != len(set(selected)) or any(not value.strip() for value in selected):
            raise ValueError("Les colonnes sélectionnées doivent être uniques et non vides.")
        protected = {"age", "dept", "last_invoice_amount", "retention_offer_declined"}
        if set(selected) & protected:
            raise ValueError(
                "Une variable protégée ou temporellement suspecte ne peut pas être activée dans cette version."
            )
        generated = {
            "tenure_months",
            "tenure_unavailable",
            "log1p_tenure_months",
            "support_payment_interaction",
            "age_group",
        }
        generated |= {
            f"{column}_{suffix}"
            for column in self.indicator_columns
            for suffix in ("missing", "invalid")
        }
        if self.log_usage:
            generated.add(f"log1p_{self.usage_column}")
        if generated & set(selected):
            raise ValueError("Une source porte le nom réservé d'une feature dérivée.")
        if set(selected) & set(self.excluded_columns):
            raise ValueError("Une colonne exclue apparaît dans les sources de features.")
        if set(selected) & {self.id_column, self.snapshot_column, self.signup_column}:
            raise ValueError(
                "Identifiant et dates brutes ne sont pas des features numériques/catégorielles."
            )
        if not set(self.indicator_columns).issubset(self.numeric_columns):
            raise ValueError("Les indicateurs doivent référencer des variables numériques.")
        required = set(selected) | {self.snapshot_column, self.signup_column}
        if self.age_grouping:
            if self.age_column not in self.excluded_columns:
                raise ValueError(
                    "L'âge brut doit rester exclu des sorties lorsque son regroupement est actif."
                )
            if not np.isfinite(self.age_bins).all() or any(
                a >= b for a, b in zip(self.age_bins, self.age_bins[1:], strict=False)
            ):
                raise ValueError("Les bornes d'âge doivent être finies et strictement croissantes.")
            required.add(self.age_column)
        if not required.issubset(self.source_availability):
            raise ValueError("La disponibilité de chaque source doit être déclarée explicitement.")
        if self.support_payment_interaction and not {
            self.support_column,
            self.payments_column,
        }.issubset(self.numeric_columns):
            raise ValueError(
                "L'interaction support/paiements nécessite ses deux sources numériques."
            )
        if self.log_usage and self.usage_column not in self.numeric_columns:
            raise ValueError("La transformation d'usage nécessite sa source numérique.")
        if not required.issubset(self.hypotheses):
            raise ValueError(
                "Chaque source doit être rattachée à une hypothèse ou règle de qualité."
            )
        return self

    @classmethod
    def load(cls, settings: Settings) -> FeatureConfig:
        """Charger le YAML depuis le chemin résolu par Settings."""
        try:
            config = cls.model_validate(_read_yaml(settings.features_config_path))
        except ValueError as exc:
            raise ValueError(
                f"Configuration des features invalide ({settings.features_config_path}) : {exc}"
            ) from exc
        if settings.churn_target_column in [*config.numeric_columns, *config.categorical_columns]:
            raise ValueError("La cible ne peut pas figurer parmi les sources de features.")
        return config
