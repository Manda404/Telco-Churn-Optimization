"""Contrats du rapport qualité adapté de BNPL."""

import json

import numpy as np
import pandas as pd
import pytest

from telco_churn.data.quality import DataQualityReport


def test_report_is_serializable_and_counts_missing_target():
    frame = pd.DataFrame({"target": [0, 1, None], "value": [1, None, 1]})
    report = DataQualityReport("target").build(frame)
    json.dumps(report, allow_nan=False)
    assert report["n_missing_values_total"] == 2
    assert report["target_missing_count"] == 1
    assert report["dtypes"]["value"] == "float64"
    assert report["target_distribution"]["1.0"]["rate"] == pytest.approx(1 / 3)


def test_profile_excludes_infinite_values_from_statistics():
    frame = pd.DataFrame({"value": [0, 1, 1, 1, 100, np.inf, -np.inf, np.nan]})
    original = frame.copy(deep=True)
    row = DataQualityReport().profile_dataset(frame).iloc[0]
    assert row["Infinite"] == 2
    assert row["Missing"] == 1
    assert row["Outliers"] == 2
    assert row["% Outliers"] == 40
    assert row["% Zeros"] == 20
    assert row["Mean"] == pytest.approx(20.6)
    pd.testing.assert_frame_equal(frame, original)


def test_empty_and_categorical_profiles():
    reporter = DataQualityReport("target")
    frame = pd.DataFrame({"target": pd.Series(dtype="string")})
    assert reporter.build(frame)["target_distribution"] == {}
    assert reporter.profile_dataset(frame).iloc[0]["% Missing"] == 0
    assert DataQualityReport().profile_dataset(pd.DataFrame()).empty
    frame = pd.DataFrame({"label": ["a", None, "b", "a"]})
    row = DataQualityReport().profile_dataset(frame, categorical_examples=1).iloc[0]
    assert row["Examples"] == ["a"]
    assert row["Cardinality"] == 2


def test_invalid_parameters_and_missing_target():
    with pytest.raises(ValueError):
        DataQualityReport(size=(0, 8))
    with pytest.raises(ValueError):
        DataQualityReport().profile_dataset(pd.DataFrame(), iqr_multiplier=float("nan"))
    with pytest.raises(ValueError, match="Colonne cible absente"):
        DataQualityReport("unknown").build(pd.DataFrame({"value": [1]}))


def test_dashboard_saves_without_leaving_open_figure(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = tmp_path / "figures" / "quality.png"
    frame = pd.DataFrame({"target": ["a", "b", "c", "d"], "value": [1, None, 2, 3]})
    before = plt.get_fignums()
    DataQualityReport("target").show(frame, save_path=destination)
    assert destination.stat().st_size > 0
    assert plt.get_fignums() == before
