"""Référence train uniquement et détection des changements cachés par les cas limites."""

import copy

import numpy as np
import pandas as pd
import pytest

from telco_churn.modeling.temporal_split import TemporalConfig
from telco_churn.monitoring.data_drift import DataDriftMonitor
from telco_churn.settings import Settings


@pytest.fixture
def config():
    return TemporalConfig.load(Settings.load()).drift


def test_identical_data_has_zero_drift(config):
    frame = pd.DataFrame({"number": np.arange(100, dtype=float), "category": ["a", "b"] * 50})
    result = DataDriftMonitor(config).fit(frame).compare(frame)
    assert result.psi.abs().max() < 1e-10
    assert result.total_variation.abs().max() < 1e-10
    assert result.severity.eq("NO_ALERT").all()


def test_numeric_constant_shift_is_detected(config):
    train = pd.DataFrame({"flag": [0] * 100})
    result = DataDriftMonitor(config).fit(train).compare(pd.DataFrame({"flag": [1] * 100})).iloc[0]
    assert result.severity == "HIGH"
    assert result.unseen_percent_of_nonmissing == 100
    assert result.total_variation == 1


def test_new_categories_and_missingness_are_distinct(config):
    train = pd.DataFrame({"category": ["missing:", "a"] * 50})
    other = pd.DataFrame({"category": [None, "new"] * 50})
    row = DataDriftMonitor(config).fit(train).compare(other).iloc[0]
    assert row.missing_delta_pp == 50
    assert row.unseen_percent_of_nonmissing == 100
    assert row.severity == "HIGH"


def test_all_missing_reference_does_not_hide_new_values(config):
    monitor = DataDriftMonitor(config).fit(pd.DataFrame({"number": [np.nan] * 100}))
    row = monitor.compare(pd.DataFrame({"number": [5.0] * 100})).iloc[0]
    assert row.reference_status == "all_missing"
    assert row.severity == "HIGH"


def test_reference_bins_are_not_refit_on_validation(config):
    train = pd.DataFrame({"number": np.arange(100, dtype=float)})
    monitor = DataDriftMonitor(config).fit(train)
    bins_before = monitor.reference_["number"]["edges"].copy()
    counts_before = copy.deepcopy(monitor.reference_["number"]["counts"])
    shifted = monitor.compare(pd.DataFrame({"number": np.arange(1000, 1100, dtype=float)})).iloc[0]
    assert shifted.severity == "HIGH"
    assert shifted.outside_train_range_percent == 100
    np.testing.assert_array_equal(bins_before, monitor.reference_["number"]["edges"])
    pd.testing.assert_series_equal(counts_before, monitor.reference_["number"]["counts"])
    assert monitor.compare(train).psi.iloc[0] == pytest.approx(0)


def test_monitor_requires_nonempty_matching_schema(config):
    monitor = DataDriftMonitor(config)
    with pytest.raises(ValueError, match="fit"):
        monitor.compare(pd.DataFrame({"x": [1]}))
    monitor.fit(pd.DataFrame({"x": [1, 2]}))
    with pytest.raises(ValueError, match="schéma"):
        monitor.compare(pd.DataFrame({"y": [1]}))


def test_distribution_plot_uses_same_reference_groups_and_sums_to_100(config):
    train = pd.DataFrame({"number": np.arange(100, dtype=float)})
    other = pd.DataFrame({"number": [np.nan, 1000.0, 1.0]})
    monitor = DataDriftMonitor(config).fit(train)
    table = monitor.distribution_table("number", {"validation": other})
    assert table.groupby("partition").percent.sum().tolist() == pytest.approx([100, 100])
    assert "Manquant / non fini" in table.bin.values
