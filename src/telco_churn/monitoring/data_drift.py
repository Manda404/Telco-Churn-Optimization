"""Drift descriptif : référence et bornes définies sur le train uniquement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from telco_churn.modeling.temporal_split import DriftConfig


class DataDriftMonitor:
    """Mesurer les différences sans choisir un split qui les ferait disparaître.

    PSI/variation totale sont des tailles d'écart, pas des p-values. Les seuils
    sont heuristiques. Aucune hypothèse d'indépendance des lignes n'est utilisée
    pour un test statistique, et le drift multivarié/conceptuel n'est pas certifié.
    """

    def __init__(self, config: DriftConfig) -> None:
        self.config = config
        self.reference_: dict[str, dict] = {}

    @staticmethod
    def _numeric(series: pd.Series) -> pd.Series:
        return (
            pd.to_numeric(series, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        )

    @staticmethod
    def _category(series: pd.Series) -> pd.Series:
        # Le préfixe évite une collision entre une vraie modalité et le marqueur NA.
        return ("value:" + series.astype("string")).fillna("missing:")

    def _distribution(self, series: pd.Series, model: dict) -> tuple[pd.Series, pd.Series]:
        if model["kind"] == "numeric":
            values = self._numeric(series)
            codes = pd.cut(values, model["edges"], labels=False, right=False)
            encoded = codes.astype("Int64").astype("string").fillna("missing:")
            return encoded.value_counts(), values
        if model["kind"] == "numeric_discrete":
            values = self._numeric(series)
            return self._category(values).value_counts(), values
        return self._category(series).value_counts(), series

    def fit(self, train: pd.DataFrame) -> DataDriftMonitor:
        """Apprendre uniquement la référence de monitoring, jamais sur validation/test."""
        if train.empty or not train.columns.is_unique:
            raise ValueError("Le train de référence doit être non vide avec des colonnes uniques.")
        self.reference_ = {}
        for column in train:
            series = train[column]
            if pd.api.types.is_numeric_dtype(series):
                values = self._numeric(series)
                finite = values.dropna()
                kind = (
                    "numeric_discrete"
                    if finite.nunique() <= self.config.quantile_bins
                    else "numeric"
                )
                model = {
                    "kind": kind,
                    "min": float(finite.min()) if len(finite) else None,
                    "max": float(finite.max()) if len(finite) else None,
                }
                if kind == "numeric":
                    internal = np.unique(
                        finite.quantile(
                            np.linspace(0, 1, self.config.quantile_bins + 1)[1:-1]
                        ).to_numpy()
                    )
                    model["edges"] = np.array([-np.inf, *internal, np.inf])
            else:
                model = {"kind": "categorical"}
            counts, values = self._distribution(series, model)
            model.update(
                counts=counts, n_rows=len(series), missing_percent=float(values.isna().mean() * 100)
            )
            self.reference_[column] = model
        return self

    def compare(self, other: pd.DataFrame) -> pd.DataFrame:
        """Comparer sans modifier les catégories ou bornes apprises sur train."""
        if not self.reference_:
            raise ValueError("Appeler fit(train_features) avant compare().")
        if other.empty or set(other.columns) != set(self.reference_):
            raise ValueError(
                "La partition comparée doit être non vide avec le même schéma que le train."
            )
        rows = []
        c = self.config
        for column, model in self.reference_.items():
            counts, values = self._distribution(other[column], model)
            keys = model["counts"].index.union(counts.index)
            p = model["counts"].reindex(keys, fill_value=0).to_numpy(dtype=float) / model["n_rows"]
            q = counts.reindex(keys, fill_value=0).to_numpy(dtype=float) / len(other)
            tv = float(np.abs(p - q).sum() / 2)
            smooth_p = np.maximum(p, c.smoothing)
            smooth_p /= smooth_p.sum()
            smooth_q = np.maximum(q, c.smoothing)
            smooth_q /= smooth_q.sum()
            psi = float(((smooth_q - smooth_p) * np.log(smooth_q / smooth_p)).sum())
            missing = float(values.isna().mean() * 100)
            delta = missing - model["missing_percent"]
            unseen = 0.0
            outside = np.nan
            if model["kind"] != "numeric":
                encoded = self._category(values)
                observed = values.notna()
                unseen = (
                    float((~encoded.loc[observed].isin(model["counts"].index)).mean() * 100)
                    if observed.any()
                    else 0.0
                )
            if model["kind"].startswith("numeric") and model["min"] is not None:
                finite = values.dropna()
                outside = (
                    float((finite.lt(model["min"]) | finite.gt(model["max"])).mean() * 100)
                    if len(finite)
                    else np.nan
                )
            high = psi >= c.psi_high or abs(delta) >= c.missing_high_pp
            warn = (
                psi >= c.psi_warning
                or abs(delta) >= c.missing_warning_pp
                or unseen > c.unseen_warning_percent
            )
            if model["kind"] != "numeric":
                high |= tv >= c.tv_high
                warn |= tv >= c.tv_warning
            rows.append(
                {
                    "feature": column,
                    "kind": model["kind"],
                    "psi": psi,
                    "total_variation": tv,
                    "train_missing_percent": model["missing_percent"],
                    "other_missing_percent": missing,
                    "missing_delta_pp": delta,
                    "unseen_percent_of_nonmissing": unseen,
                    "outside_train_range_percent": outside,
                    "reference_status": "all_missing"
                    if model["missing_percent"] == 100
                    else "available",
                    "severity": "HIGH" if high else "WARNING" if warn else "NO_ALERT",
                    "n_train": model["n_rows"],
                    "n_other": len(other),
                }
            )
        return pd.DataFrame(rows).sort_values(
            "psi", ascending=False, kind="stable", ignore_index=True
        )

    def reference_summary(self) -> pd.DataFrame:
        """Rendre les bornes de référence inspectables, sans consulter le test."""
        return pd.DataFrame(
            [
                {
                    "feature": column,
                    "kind": model["kind"],
                    "n_reference": model["n_rows"],
                    "bins_or_levels": len(model.get("edges", [])) - 1
                    if "edges" in model
                    else len(model["counts"]),
                    "train_min": model.get("min"),
                    "train_max": model.get("max"),
                }
                for column, model in self.reference_.items()
            ]
        )

    def distribution_table(
        self, feature: str, comparisons: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Rendre visibles les proportions dans les mêmes bins/niveaux de référence."""
        if feature not in self.reference_:
            raise ValueError(f"Feature absente de la référence : {feature}")
        model = self.reference_[feature]
        distributions = {"train": (model["counts"], model["n_rows"])}
        for name, frame in comparisons.items():
            if frame.empty or feature not in frame:
                raise ValueError(
                    "Une comparaison de distributions nécessite des lignes et la feature."
                )
            distributions[name] = (self._distribution(frame[feature], model)[0], len(frame))
        keys = set().union(*(counts.index for counts, _ in distributions.values()))
        if model["kind"] == "numeric":
            ordered = sorted(keys, key=lambda key: float("inf") if key == "missing:" else int(key))
        else:
            ordered = sorted(keys)
        rows = []
        for key in ordered:
            if key == "missing:":
                label = "Manquant / non fini"
            elif model["kind"] == "numeric":
                index = int(key)
                label = f"[{model['edges'][index]:.2f}, {model['edges'][index + 1]:.2f})"
            else:
                label = str(key).removeprefix("value:")
            for name, (counts, size) in distributions.items():
                rows.append(
                    {"bin": label, "partition": name, "percent": counts.get(key, 0) / size * 100}
                )
        return pd.DataFrame(rows)
