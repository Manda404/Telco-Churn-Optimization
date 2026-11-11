"""Contrôles non destructifs et diagnostics ciblés pour le notebook qualité."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from telco_churn.data.schema import DatasetRules, NumericRule

RESULT_COLUMNS = [
    "rule",
    "column",
    "severity",
    "status",
    "n_checked",
    "n_failed",
    "failed_percent",
    "examples",
    "action",
]


def _missing(series: pd.Series) -> pd.Series:
    """Inclure les chaînes vides ou composées d'espaces dans les manquants."""
    return series.isna() | series.astype("string").str.strip().eq("").fillna(False)


class DataValidator:
    """Évaluer un contrat sans modifier la source ni interrompre l'exploration.

    Les taux portent sur les lignes éligibles à chaque règle. Une ligne peut
    échouer à plusieurs règles : ne pas additionner les effectifs d'anomalies.
    Les exemples contiennent des positions de lignes (base zéro), pas des labels d'index.
    """

    def __init__(self, rules: DatasetRules, target_column: str, example_limit: int = 5) -> None:
        if not target_column.strip() or example_limit < 1:
            raise ValueError("La cible doit être renseignée et example_limit positif.")
        self.rules = rules
        self.target_column = target_column
        self.example_limit = example_limit

    def assess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retourner les contrôles PASS, FAIL ou SKIP et des exemples d'anomalies."""
        if not df.columns.is_unique:
            raise ValueError(
                "Les noms de colonnes dupliqués doivent être résolus avant validation."
            )
        # Un index de travail unique évite les alignements ambigus ; la source reste intacte.
        frame = df.reset_index(drop=True)
        results: list[dict[str, Any]] = []

        def record(
            rule: str,
            columns: list[str],
            bad: pd.Series | None,
            *,
            eligible: pd.Series | None = None,
            severity: str = "error",
            action: str,
        ) -> None:
            if bad is None:
                results.append(
                    dict(
                        zip(
                            RESULT_COLUMNS,
                            [
                                rule,
                                ", ".join(columns),
                                severity,
                                "SKIP",
                                0,
                                0,
                                None,
                                [],
                                action,
                            ],
                            strict=True,
                        )
                    )
                )
                return
            eligible = pd.Series(True, index=frame.index) if eligible is None else eligible
            selected = bad.fillna(False) & eligible.fillna(False)
            checked, failed = int(eligible.sum()), int(selected.sum())
            examples = [
                {"row_position": int(pos), **frame.loc[pos, columns].to_dict()}
                for pos in frame.index[selected][: self.example_limit]
            ]
            results.append(
                dict(
                    zip(
                        RESULT_COLUMNS,
                        [
                            rule,
                            ", ".join(columns),
                            severity,
                            "FAIL" if failed else ("PASS" if checked else "SKIP"),
                            checked,
                            failed,
                            100 * failed / checked if checked else None,
                            examples,
                            action,
                        ],
                        strict=True,
                    )
                )
            )

        # Les contrôles de structure comptent les colonnes, pas les lignes.
        required = list(dict.fromkeys([*self.rules.required_columns, self.target_column]))
        missing_columns = [column for column in required if column not in frame]
        results.append(
            dict(
                zip(
                    RESULT_COLUMNS,
                    [
                        "required_columns",
                        "schema",
                        "error",
                        "FAIL" if missing_columns else "PASS",
                        len(required),
                        len(missing_columns),
                        100 * len(missing_columns) / len(required),
                        missing_columns,
                        "Vérifier le fichier source et le contrat de colonnes.",
                    ],
                    strict=True,
                )
            )
        )
        results.append(
            dict(
                zip(
                    RESULT_COLUMNS,
                    [
                        "non_empty_dataset",
                        "dataset",
                        "error",
                        "PASS" if len(frame) else "FAIL",
                        1,
                        int(frame.empty),
                        100.0 if frame.empty else 0.0,
                        [],
                        "Un fichier sans observations ne permet pas de conclure sur la qualité.",
                    ],
                    strict=True,
                )
            )
        )
        record(
            "duplicate_rows",
            list(frame.columns),
            frame.duplicated(keep="first"),
            action="Examiner les copies après la première occurrence avant toute suppression.",
        )

        critical = {
            self.rules.id_column,
            self.target_column,
            *self.rules.unique_key,
            *self.rules.dates,
        }
        for column in required:
            if column not in frame:
                record(
                    "missing_values",
                    [column],
                    None,
                    action="Contrôle impossible : colonne absente.",
                )
                continue
            record(
                "missing_values",
                [column],
                _missing(frame[column]),
                severity="error" if column in critical else "warning",
                action="Identifier l'origine des manquants ; aucune imputation à cette étape.",
            )

        key = self.rules.unique_key
        if all(column in frame for column in key):
            eligible = ~pd.concat([_missing(frame[column]) for column in key], axis=1).any(axis=1)
            bad = pd.Series(False, index=frame.index)
            bad.loc[eligible] = frame.loc[eligible].duplicated(key, keep=False)
            record(
                "unique_key",
                key,
                bad,
                eligible=eligible,
                severity=self.rules.key_severity,
                action=self.rules.key_note + " Toutes les lignes des clés répétées sont comptées.",
            )
        else:
            record("unique_key", key, None, action="Contrôle impossible : clé incomplète.")

        numeric_rules = dict(self.rules.numeric)
        # La cible vient de Settings ; son nom n'est jamais dupliqué dans validation.yaml.
        numeric_rules[self.target_column] = NumericRule(
            minimum=0, maximum=1, integer=True, note="Cible binaire 0/1 attendue."
        )
        for column, rule in numeric_rules.items():
            if column not in frame:
                record("numeric_domain", [column], None, action="Colonne absente.")
                continue
            present = ~_missing(frame[column])
            numeric = pd.to_numeric(frame[column], errors="coerce")
            finite = numeric.notna() & np.isfinite(numeric)
            # Les nombres encodés en chaînes sont signalés, sans conversion de la source.
            record(
                "numeric_type",
                [column],
                present
                & ~frame[column].map(
                    lambda value: isinstance(value, (int, float, np.number))
                    and not isinstance(value, (bool, np.bool_))
                ),
                eligible=present,
                severity=rule.severity,
                action="Vérifier le type numérique inféré et les chaînes de caractères.",
            )
            bad = ~finite
            if rule.minimum is not None:
                bad |= numeric < rule.minimum
            if rule.maximum is not None:
                bad |= numeric > rule.maximum
            if rule.integer:
                bad |= numeric.where(finite).mod(1).ne(0)
            record(
                "numeric_domain",
                [column],
                bad,
                eligible=present,
                severity=rule.severity,
                action=rule.note,
            )

        for column, allowed in self.rules.categories.items():
            if column not in frame:
                record("category", [column], None, action="Colonne absente.")
                continue
            series = frame[column]
            present = ~_missing(series)
            record(
                "category",
                [column],
                ~series.isin(allowed),
                eligible=present,
                severity="warning",
                action=f"Modalités canoniques proposées : {allowed}. À confirmer.",
            )
            normalized = series.astype("string").str.strip().str.casefold()
            expected = normalized.map({value.strip().casefold(): value for value in allowed})
            record(
                "category_format",
                [column],
                expected.notna() & series.astype("string").ne(expected),
                eligible=present,
                severity="warning",
                action="Vérifier les différences de casse/espaces, sans normaliser la source.",
            )

        dates: dict[str, pd.Series] = {}
        for column, date_format in self.rules.dates.items():
            if column not in frame:
                record("date_parse", [column], None, action="Colonne absente.")
                continue
            dates[column] = pd.to_datetime(frame[column], format=date_format, errors="coerce")
            record(
                "date_parse",
                [column],
                dates[column].isna(),
                eligible=~_missing(frame[column]),
                action=f"Date attendue au format {date_format} ; vérifier les valeurs invalides.",
            )
        for earlier, later in self.rules.date_order:
            if earlier not in dates or later not in dates:
                record("date_order", [earlier, later], None, action="Dates nécessaires absentes.")
                continue
            eligible = dates[earlier].notna() & dates[later].notna()
            record(
                "date_order",
                [earlier, later],
                dates[earlier] > dates[later],
                eligible=eligible,
                action=f"Vérifier la chronologie : {earlier} doit précéder ou égaler {later}.",
            )
        result = pd.DataFrame(results, columns=RESULT_COLUMNS)
        logger.bind(pipeline="data.validation").info(
            "Quality checks: rows={} rules={} failed={} skipped={}",
            len(frame),
            len(result),
            int(result.status.eq("FAIL").sum()),
            int(result.status.eq("SKIP").sum()),
        )
        return result

    def missingness_by_group(self, df: pd.DataFrame, group_column: str) -> pd.DataFrame:
        """Comparer les manquants par segment ; conserver les groupes eux-mêmes manquants."""
        if group_column not in df:
            raise ValueError(f"Colonne de regroupement absente : {group_column}")
        rows = []
        for group, subset in df.groupby(group_column, dropna=False, observed=True):
            for column in self.rules.missingness_columns:
                if column not in subset:
                    continue
                count = int(_missing(subset[column]).sum())
                rows.append(
                    {
                        "group": group,
                        "column": column,
                        "n_rows": len(subset),
                        "n_missing": count,
                        "missing_percent": 100 * count / len(subset),
                    }
                )
        return pd.DataFrame(
            rows, columns=["group", "column", "n_rows", "n_missing", "missing_percent"]
        )

    def temporal_coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Afficher les bornes observées et la disponibilité des dates déclarées."""
        rows = []
        for column, date_format in self.rules.dates.items():
            if column not in df:
                continue
            parsed = pd.to_datetime(df[column], format=date_format, errors="coerce")
            rows.append(
                {
                    "column": column,
                    "first_date": parsed.min(),
                    "last_date": parsed.max(),
                    "n_valid": int(parsed.notna().sum()),
                    "n_distinct": parsed.nunique(),
                }
            )
        return pd.DataFrame(
            rows, columns=["column", "first_date", "last_date", "n_valid", "n_distinct"]
        )
