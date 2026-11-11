"""Découpage temporel purgé, traçabilité des exclusions et contrôles de fuite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from telco_churn.settings import Settings, _read_yaml


def _shift_days(dates: pd.Series, days: int) -> pd.Series:
    """Ajouter des jours avec une unité NumPy explicite, sans conversion implicite."""
    shifted = dates.to_numpy(dtype="datetime64[ns]") + np.timedelta64(days, "D")
    return pd.Series(shifted, index=dates.index)


class BacktestFold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train_end: date
    validation_start: date
    validation_end: date


class DriftConfig(BaseModel):
    """Seuils descriptifs heuristiques, pas des tests de significativité."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    quantile_bins: int = Field(ge=2, le=50)
    smoothing: float = Field(gt=0, lt=0.01)
    psi_warning: float = Field(gt=0)
    psi_high: float = Field(gt=0)
    tv_warning: float = Field(gt=0, le=1)
    tv_high: float = Field(gt=0, le=1)
    missing_warning_pp: float = Field(gt=0, le=100)
    missing_high_pp: float = Field(gt=0, le=100)
    unseen_warning_percent: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def check_thresholds(self) -> DriftConfig:
        for low, high in [
            (self.psi_warning, self.psi_high),
            (self.tv_warning, self.tv_high),
            (self.missing_warning_pp, self.missing_high_pp),
        ]:
            if low >= high:
                raise ValueError("Le seuil d'alerte doit précéder le seuil élevé.")
        return self


class TemporalConfig(BaseModel):
    """Fenêtres définies a priori et hypothèses à confirmer avant évaluation finale."""

    model_config = ConfigDict(extra="forbid")
    id_column: str
    date_column: str
    date_format: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date
    label_horizon_days: int = Field(ge=1)
    label_horizon_confirmed: bool
    label_availability_lag_days: int = Field(ge=0)
    embargo_days: int = Field(ge=0)
    label_observed_through: date | None
    customer_policy: Literal["recurring", "disjoint"]
    duplicate_policy: Literal["drop_equivalent", "error"]
    prior_eda_start: date
    prior_eda_end: date
    backtest_folds: list[BacktestFold] = Field(default_factory=list)
    drift: DriftConfig

    @model_validator(mode="after")
    def check_dates(self) -> TemporalConfig:
        if (
            not self.train_start
            <= self.train_end
            < self.validation_start
            <= self.validation_end
            < self.test_start
            <= self.test_end
        ):
            raise ValueError(
                "Les fenêtres train, validation et test doivent être ordonnées sans chevauchement."
            )
        if self.prior_eda_start > self.prior_eda_end:
            raise ValueError("Les dates de l'EDA doivent être ordonnées.")
        for fold in self.backtest_folds:
            if (
                not self.train_start
                <= fold.train_end
                < fold.validation_start
                <= fold.validation_end
                <= self.train_end
            ):
                raise ValueError(
                    "Chaque backtest doit rester entièrement dans la période du train principal."
                )
        return self

    @property
    def purge_days(self) -> int:
        return self.label_horizon_days + self.label_availability_lag_days + self.embargo_days

    @classmethod
    def load(cls, settings: Settings) -> TemporalConfig:
        try:
            return cls.model_validate(_read_yaml(settings.temporal_config_path))
        except ValueError as exc:
            raise ValueError(
                f"Configuration temporelle invalide ({settings.temporal_config_path}) : {exc}"
            ) from exc


@dataclass
class TemporalPartitions:
    """Partitions et source des lignes, sans perte silencieuse."""

    partitions: dict[str, pd.DataFrame]
    excluded: pd.DataFrame
    input_rows: int

    def summary(self, config: TemporalConfig, target: str) -> pd.DataFrame:
        rows = []
        for name, frame in self.partitions.items():
            rows.append(
                {
                    "partition": name,
                    "n_rows": len(frame),
                    "n_clients": frame[config.id_column].nunique(),
                    "first_date": frame[config.date_column].min(),
                    "last_date": frame[config.date_column].max(),
                    "n_positive_snapshots": int(frame[target].sum()),
                    "n_positive_clients": frame.loc[
                        frame[target].eq(1), config.id_column
                    ].nunique(),
                    "positive_percent": frame[target].mean() * 100 if len(frame) else np.nan,
                }
            )
        return pd.DataFrame(rows)


class TemporalSplitter:
    """Découper sans rééquilibrage, avec labels purgés avant la fenêtre suivante."""

    def __init__(self, config: TemporalConfig, target_column: str) -> None:
        self.config = config
        self.target_column = target_column

    def split(self, df: pd.DataFrame, *, require_nonempty: bool = True) -> TemporalPartitions:
        c = self.config
        required = {c.id_column, c.date_column, self.target_column}
        if not df.columns.is_unique or not required.issubset(df.columns):
            raise ValueError("Colonnes uniques et identifiant/date/cible obligatoires.")
        if "source_row" in df or "exclusion_reason" in df:
            raise ValueError("Noms réservés source_row/exclusion_reason présents dans la source.")
        work = df.copy(deep=True).reset_index(drop=True)
        work.index.name = "source_row"
        work[c.date_column] = pd.to_datetime(
            work[c.date_column], format=c.date_format, errors="coerce"
        )
        if work[c.date_column].isna().any() or work[c.id_column].isna().any():
            raise ValueError(
                "Dates ou identifiants absents/invalides : impossible de garantir le découpage."
            )
        if (work[self.target_column].notna() & ~work[self.target_column].isin([0, 1])).any():
            raise ValueError("La cible doit être binaire 0/1 lorsqu'elle est renseignée.")
        reasons = pd.Series("outside_windows", index=work.index, dtype="string")
        duplicates = work.duplicated(keep="first")
        if duplicates.any() and c.duplicate_policy == "error":
            raise ValueError("Snapshots équivalents dupliqués : politique error.")
        reasons.loc[duplicates] = "duplicate_equivalent_snapshot"
        eligible = work.loc[~duplicates]
        if eligible.duplicated([c.id_column, c.date_column], keep=False).any():
            raise ValueError(
                "Clés client/date non équivalentes : résoudre les conflits avant le split."
            )
        missing_target = work[self.target_column].isna() & ~duplicates
        reasons.loc[missing_target] = "missing_target"
        eligible = work.loc[~duplicates & ~missing_target]
        windows = {
            "train": (c.train_start, c.train_end, c.validation_start),
            "validation": (c.validation_start, c.validation_end, c.test_start),
            "test": (c.test_start, c.test_end, None),
        }
        parts = {}
        seen: set = set()
        for name, (start, end, next_start) in windows.items():
            selected = eligible.loc[
                eligible[c.date_column].between(pd.Timestamp(start), pd.Timestamp(end))
            ].copy()
            if next_start is not None:
                mature = _shift_days(selected[c.date_column], c.purge_days) < pd.Timestamp(
                    next_start
                )
                reasons.loc[selected.index[~mature]] = "purged_label_overlap"
                selected = selected.loc[mature]
            if c.customer_policy == "disjoint":
                overlap = selected[c.id_column].isin(seen)
                reasons.loc[selected.index[overlap]] = "customer_already_seen"
                selected = selected.loc[~overlap]
            selected = selected.sort_values([c.date_column, c.id_column], kind="stable")
            if require_nonempty and selected.empty:
                raise ValueError(
                    f"Partition {name} vide après filtres : revoir le scénario sans mélanger les dates."
                )
            seen.update(selected[c.id_column].unique())
            reasons.loc[selected.index] = pd.NA
            parts[name] = selected
        excluded = work.loc[reasons.notna()].copy()
        excluded["exclusion_reason"] = reasons.loc[excluded.index]
        if sum(len(part) for part in parts.values()) + len(excluded) != len(df):
            raise AssertionError(
                "Le compte des lignes affectées/exclues ne correspond pas à l'entrée."
            )
        return TemporalPartitions(parts, excluded, len(df))

    def backtests(self, result: TemporalPartitions) -> dict[str, dict[str, pd.DataFrame]]:
        """Fenêtres croissantes limitées au train principal ; aucun test final consulté."""
        c = self.config
        training = result.partitions["train"]
        folds = {}
        for index, fold in enumerate(c.backtest_folds, 1):
            train = training.loc[training[c.date_column].le(pd.Timestamp(fold.train_end))].copy()
            train = train.loc[
                _shift_days(train[c.date_column], c.purge_days)
                < pd.Timestamp(fold.validation_start)
            ]
            valid = training.loc[
                training[c.date_column].between(
                    pd.Timestamp(fold.validation_start), pd.Timestamp(fold.validation_end)
                )
            ].copy()
            if c.customer_policy == "disjoint":
                valid = valid.loc[~valid[c.id_column].isin(train[c.id_column])]
            if train.empty or valid.empty:
                raise ValueError(f"Backtest {index} vide avec la stratégie choisie.")
            folds[f"fold_{index}"] = {"train": train, "validation": valid}
        return folds

    def leakage_report(self, result: TemporalPartitions, manifest: dict) -> pd.DataFrame:
        """Distinguer contrôles techniques et points impossibles à certifier automatiquement."""
        c = self.config
        rows = []
        features = manifest["feature_columns"]

        def add(check: str, status: str, detail: str) -> None:
            rows.append({"check": check, "status": status, "detail": detail})

        forbidden = {
            self.target_column,
            c.id_column,
            c.date_column,
            "age",
            "last_invoice_amount",
            "retention_offer_declined",
        }
        found = sorted(set(features) & forbidden)
        add(
            "feature_exclusions",
            "FAIL" if found else "PASS",
            f"Colonnes interdites dans X : {found}",
        )
        for left, right in combinations(result.partitions, 2):
            a, b = result.partitions[left], result.partitions[right]
            shared_keys = len(
                pd.MultiIndex.from_frame(a[[c.id_column, c.date_column]]).intersection(
                    pd.MultiIndex.from_frame(b[[c.id_column, c.date_column]])
                )
            )
            shared_clients = len(set(a[c.id_column]) & set(b[c.id_column]))
            add(
                f"snapshot_overlap_{left}_{right}",
                "FAIL" if shared_keys else "PASS",
                f"{shared_keys} clés communes",
            )
            add(
                f"client_overlap_{left}_{right}",
                "INFO"
                if c.customer_policy == "recurring"
                else ("FAIL" if shared_clients else "PASS"),
                f"{shared_clients} clients communs ; politique {c.customer_policy}",
            )
        names = list(result.partitions)
        for left, right in zip(names, names[1:], strict=False):
            a, b = result.partitions[left], result.partitions[right]
            valid = bool(
                len(a)
                and len(b)
                and _shift_days(a[c.date_column], c.purge_days).max() < b[c.date_column].min()
            )
            add(
                f"chronology_and_purge_{left}_{right}",
                "PASS" if valid else "FAIL",
                f"Purge de {c.purge_days} jours, selon horizon déclaré",
            )
        add(
            "label_horizon",
            "PASS" if c.label_horizon_confirmed else "UNVERIFIED",
            f"{c.label_horizon_days} jours : hypothèse"
            if not c.label_horizon_confirmed
            else "Horizon déclaré confirmé",
        )
        pending = manifest.get("unconfirmed_features")
        add(
            "feature_availability",
            "UNVERIFIED" if pending is None or len(pending) else "PASS",
            f"Sources de {len(pending or [])} features non confirmées ; voir catalogue",
        )
        add(
            "learned_preprocessing",
            "PASS" if manifest.get("learned_statistics") is False else "UNVERIFIED",
            "Aucune statistique apprise avant le split déclarée dans le manifeste",
        )
        test = result.partitions["test"]
        seen_eda = (
            test[c.date_column]
            .between(pd.Timestamp(c.prior_eda_start), pd.Timestamp(c.prior_eda_end))
            .any()
        )
        add(
            "untouched_test",
            "WARN" if seen_eda else "UNVERIFIED",
            "Période déjà explorée dans l'EDA : test de développement"
            if seen_eda
            else "Hors période EDA déclarée ; provenance à confirmer",
        )
        if c.label_observed_through is None:
            add(
                "test_label_maturity",
                "UNVERIFIED",
                "Fin de disponibilité des labels non documentée",
            )
        else:
            mature = not test.empty and _shift_days(
                test[c.date_column], c.label_horizon_days + c.label_availability_lag_days
            ).max() <= pd.Timestamp(c.label_observed_through)
            add(
                "test_label_maturity",
                "PASS" if mature else "FAIL",
                f"Labels observés jusqu'au {c.label_observed_through}",
            )
        train = result.partitions["train"]
        for column in features:
            if column in forbidden:
                continue  # Déjà signalé comme fuite ; ne pas dupliquer la cible dans la sélection.
            if column not in train:
                add(f"missing_feature_{column}", "FAIL", "Feature déclarée absente des partitions")
                continue
            if (
                pd.api.types.is_numeric_dtype(train[column])
                and train[self.target_column].nunique() == 2
            ):
                observed = train[[column, self.target_column]].dropna()
                if len(observed) and (
                    observed[column].eq(observed[self.target_column]).all()
                    or observed[column].eq(1 - observed[self.target_column]).all()
                ):
                    add(
                        f"target_equivalence_{column}",
                        "WARN",
                        "Égalité exacte à la cible/inverse sur train : signal suspect à auditer, pas preuve de fuite",
                    )
        return pd.DataFrame(rows)


def load_processed(settings: Settings) -> tuple[pd.DataFrame, dict]:
    """Relire le dataset et son schéma ; ne jamais deviner les features par drop de la cible."""
    path = settings.feature_output_dir / "churn_clients_processed.csv"
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    features = manifest["feature_columns"]
    metadata = manifest["metadata_columns"]
    if len(set(features + metadata)) != len(features) + len(metadata):
        raise ValueError("Features et métadonnées doivent être distinctes et uniques.")
    frame = pd.read_csv(path, dtype=manifest.get("feature_dtypes", {}))
    if frame.columns.tolist() != [*metadata, *features] or len(frame) != manifest["n_rows"]:
        raise ValueError("Dataset transformé et manifeste incohérents : relancer le notebook 04.")
    return frame, manifest


def save_temporal_partitions(
    result: TemporalPartitions,
    manifest: dict,
    config: TemporalConfig,
    output_dir: Path,
    leakage: pd.DataFrame,
    drift: pd.DataFrame,
) -> Path:
    """Exporter des partitions de développement et leur statut sans les certifier prêtes au modèle."""
    if leakage.status.eq("FAIL").any():
        raise ValueError("Export refusé : un contrôle technique de fuite a échoué.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in result.partitions.items():
        frame.to_csv(output_dir / f"{name}.csv", index=True, index_label="source_row")
    result.excluded.to_csv(output_dir / "excluded_rows.csv", index=True, index_label="source_row")
    leakage.to_csv(output_dir / "leakage_report.csv", index=False)
    drift.to_csv(output_dir / "drift_report.csv", index=False)
    payload = {
        "status": "provisional_development_split",
        "training_validated": False,
        "feature_columns": manifest["feature_columns"],
        "feature_dtypes": manifest.get("feature_dtypes", {}),
        "metadata_columns": ["source_row", *manifest["metadata_columns"]],
        "source_row_is_not_a_feature": True,
        "config": config.model_dump(mode="json"),
        "n_rows": {name: len(frame) for name, frame in result.partitions.items()},
        "excluded_rows": len(result.excluded),
        "unverified_checks": leakage.loc[leakage.status.eq("UNVERIFIED"), "check"].tolist(),
        "failed_checks": leakage.loc[leakage.status.eq("FAIL"), "check"].tolist(),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir
