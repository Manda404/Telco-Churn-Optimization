"""Règles de qualité typées, chargées depuis le YAML du projet."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from telco_churn.settings import Settings, _read_yaml


class NumericRule(BaseModel):
    """Domaine numérique attendu ; bornes inclusives, valeurs manquantes séparées."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    minimum: float | None = None
    maximum: float | None = None
    integer: bool = False
    severity: Literal["error", "warning"] = "error"
    note: str

    @model_validator(mode="after")
    def check_bounds(self) -> NumericRule:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum doit être inférieur ou égal à maximum")
        return self


class DatasetRules(BaseModel):
    """Contrat d'une source ; les contraintes métier restent documentées."""

    model_config = ConfigDict(extra="forbid")
    required_columns: list[str] = Field(min_length=1)
    id_column: str
    unique_key: list[str] = Field(min_length=1)
    key_severity: Literal["error", "warning"] = "error"
    key_note: str
    numeric: dict[str, NumericRule] = Field(default_factory=dict)
    categories: dict[str, list[str]] = Field(default_factory=dict)
    dates: dict[str, str] = Field(default_factory=dict)
    date_order: list[tuple[str, str]] = Field(default_factory=list)
    missingness_columns: list[str] = Field(default_factory=list)
    group_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_references(self) -> DatasetRules:
        referenced = {
            self.id_column,
            *self.unique_key,
            *self.numeric,
            *self.categories,
            *self.dates,
            *self.missingness_columns,
            *self.group_columns,
        }
        for earlier, later in self.date_order:
            referenced.update((earlier, later))
            if earlier not in self.dates or later not in self.dates:
                raise ValueError("Les dates comparées doivent avoir un format déclaré.")
        if not referenced.issubset(self.required_columns):
            raise ValueError(
                f"Colonnes non déclarées : {sorted(referenced - set(self.required_columns))}"
            )
        if any(not values for values in self.categories.values()):
            raise ValueError("Les listes de catégories attendues ne doivent pas être vides.")
        return self


class ValidationConfig(BaseModel):
    """Règles des deux sources et taille maximale des exemples d'anomalies."""

    model_config = ConfigDict(extra="forbid")
    example_limit: int = Field(default=5, ge=1, le=50)
    datasets: dict[Literal["churn", "pilot"], DatasetRules]

    @model_validator(mode="after")
    def require_datasets(self) -> ValidationConfig:
        if set(self.datasets) != {"churn", "pilot"}:
            raise ValueError("Les règles churn et pilot sont toutes deux nécessaires.")
        return self

    @classmethod
    def load(cls, settings: Settings) -> ValidationConfig:
        """Lire et valider le fichier configuré, indépendamment du dossier courant."""
        path = settings.validation_config_path
        try:
            return cls.model_validate(_read_yaml(path))
        except ValueError as exc:
            raise ValueError(f"Règles de qualité invalides dans {path} : {exc}") from exc
