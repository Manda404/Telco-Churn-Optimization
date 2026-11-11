"""Chronologie, purge, intégrité des partitions et absence de fausse certification."""

import pandas as pd
import pytest

from telco_churn.modeling.temporal_split import TemporalConfig, TemporalSplitter
from telco_churn.settings import Settings


@pytest.fixture
def config():
    return TemporalConfig.load(Settings.load())


@pytest.fixture
def frame(config):
    return pd.DataFrame(
        {
            config.id_column: [1, 2, 1, 3, 1, 4, 5],
            config.date_column: [
                "2024-01-31",
                "2024-06-30",
                "2024-08-31",
                "2024-09-30",
                "2024-11-30",
                "2024-12-31",
                "2024-07-31",
            ],
            "target": [0, 1, 0, 1, 1, 0, 0],
            "feature": [1, 2, 3, 4, 5, 6, 7],
        }
    )


def test_dates_key_disjointness_and_row_accounting(config, frame):
    original = frame.copy(deep=True)
    result = TemporalSplitter(config, "target").split(frame)
    assert [len(result.partitions[name]) for name in ("train", "validation", "test")] == [2, 2, 2]
    assert result.excluded.exclusion_reason.tolist() == ["outside_windows"]
    assert sum(map(len, result.partitions.values())) + len(result.excluded) == len(frame)
    pd.testing.assert_frame_equal(frame, original)
    train_rows = set(result.partitions["train"].index)
    assert train_rows.isdisjoint(result.partitions["test"].index)


def test_duplicate_equivalent_rows_are_recorded(config, frame):
    repeated = pd.concat([frame, frame.iloc[[0]]])
    result = TemporalSplitter(config, "target").split(repeated)
    assert result.excluded.exclusion_reason.eq("duplicate_equivalent_snapshot").sum() == 1
    assert len(result.partitions["train"]) == 2


def test_conflicting_customer_date_fails(config, frame):
    extra = frame.iloc[[0]].copy()
    extra["target"] = 1
    with pytest.raises(ValueError, match="Clés client/date non équivalentes"):
        TemporalSplitter(config, "target").split(pd.concat([frame, extra]))


def test_horizon_purge_is_strict_and_audited(config, frame):
    # 2024-06-30 + 32 jours = 2024-08-01 : égalité à la frontière donc purge.
    config = TemporalConfig.model_validate({**config.model_dump(), "label_horizon_days": 32})
    result = TemporalSplitter(config, "target").split(frame)
    assert len(result.partitions["train"]) == 1
    assert result.excluded.exclusion_reason.eq("purged_label_overlap").sum() >= 1


def test_disjoint_mode_excludes_seen_clients(config, frame):
    config = TemporalConfig.model_validate({**config.model_dump(), "customer_policy": "disjoint"})
    result = TemporalSplitter(config, "target").split(frame)
    assert result.partitions["validation"][config.id_column].tolist() == [3]
    assert result.partitions["test"][config.id_column].tolist() == [4]
    assert result.excluded.exclusion_reason.eq("customer_already_seen").sum() == 2


def test_missing_target_is_excluded_and_bad_dates_fail(config, frame):
    frame.loc[0, "target"] = None
    result = TemporalSplitter(config, "target").split(frame)
    assert result.excluded.exclusion_reason.eq("missing_target").sum() == 1
    frame.loc[0, config.date_column] = "bad-date"
    with pytest.raises(ValueError, match="Dates ou identifiants"):
        TemporalSplitter(config, "target").split(frame)


def test_unknown_horizon_and_availability_are_not_passed(config, frame):
    splitter = TemporalSplitter(config, "target")
    result = splitter.split(frame)
    report = splitter.leakage_report(
        result,
        {
            "feature_columns": ["feature"],
            "unconfirmed_features": ["feature"],
            "learned_statistics": False,
        },
    ).set_index("check")
    assert report.loc["label_horizon", "status"] == "UNVERIFIED"
    assert report.loc["feature_availability", "status"] == "UNVERIFIED"
    assert report.loc["test_label_maturity", "status"] == "UNVERIFIED"
    assert report.loc["untouched_test", "status"] == "WARN"
    assert report.loc["client_overlap_train_test", "status"] == "INFO"


def test_invalid_periods_and_folds_are_rejected(config):
    with pytest.raises(ValueError, match="fenêtres"):
        TemporalConfig.model_validate({**config.model_dump(), "validation_start": "2024-05-01"})
    with pytest.raises(ValueError, match="backtest"):
        TemporalConfig.model_validate(
            {
                **config.model_dump(),
                "backtest_folds": [
                    {
                        "train_end": "2024-06-30",
                        "validation_start": "2024-08-01",
                        "validation_end": "2024-09-30",
                    }
                ],
            }
        )


def test_backtests_never_use_main_validation_or_test(config):
    frame = pd.DataFrame(
        {
            config.id_column: range(12),
            config.date_column: pd.date_range("2024-01-31", periods=12, freq="ME"),
            "target": [0, 1] * 6,
            "feature": range(12),
        }
    )
    splitter = TemporalSplitter(config, "target")
    result = splitter.split(frame)
    folds = splitter.backtests(result)
    allowed = set(result.partitions["train"].index)
    for fold in folds.values():
        assert set(fold["train"].index).issubset(allowed)
        assert set(fold["validation"].index).issubset(allowed)
        assert (
            fold["train"][config.date_column].max() + pd.DateOffset(days=config.purge_days)
            < fold["validation"][config.date_column].min()
        )


def test_maturity_is_checked_when_label_observation_cutoff_is_provided(config, frame):
    manifest = {
        "feature_columns": ["feature"],
        "unconfirmed_features": [],
        "learned_statistics": False,
    }
    for cutoff, expected in [("2025-01-30", "PASS"), ("2024-12-31", "FAIL")]:
        scenario = TemporalConfig.model_validate(
            {**config.model_dump(), "label_observed_through": cutoff}
        )
        splitter = TemporalSplitter(scenario, "target")
        report = splitter.leakage_report(splitter.split(frame), manifest).set_index("check")
        assert report.loc["test_label_maturity", "status"] == expected


def test_export_refuses_technical_leakage_failures(config, frame, tmp_path):
    from telco_churn.modeling.temporal_split import save_temporal_partitions

    splitter = TemporalSplitter(config, "target")
    result = splitter.split(frame)
    manifest = {
        "feature_columns": ["target"],
        "metadata_columns": [config.id_column, config.date_column],
        "unconfirmed_features": [],
        "learned_statistics": False,
    }
    leakage = splitter.leakage_report(result, manifest)
    assert leakage.loc[leakage.check.eq("feature_exclusions"), "status"].iloc[0] == "FAIL"
    with pytest.raises(ValueError, match="Export refusé"):
        save_temporal_partitions(result, manifest, config, tmp_path, leakage, pd.DataFrame())


def test_split_frieze_uses_calendar_months_and_configured_windows(config, frame):
    from datetime import datetime

    from telco_churn.visualization.validation import plot_split_timeline

    summary = TemporalSplitter(config, "target").split(frame).summary(config, "target")
    figure = plot_split_timeline(summary, config)
    assert list(figure.layout.xaxis.ticktext) == [
        "Jan",
        "Fév",
        "Mar",
        "Avr",
        "Mai",
        "Juin",
        "Juil",
        "Août",
        "Sept",
        "Oct",
        "Nov",
        "Déc",
    ]
    assert list(figure.data[1].x) == [datetime(2024, 1, 1), datetime(2024, 7, 1)]
    assert list(figure.data[2].x) == [datetime(2024, 8, 1), datetime(2024, 10, 1)]
    assert list(figure.data[3].x) == [datetime(2024, 11, 1), datetime(2025, 1, 1)]
    assert all(set(trace.y) == {0} for trace in figure.data)
    assert sum(annotation.text == "Hors<br>split" for annotation in figure.layout.annotations) == 2
    figure.to_json(validate=True)


def test_partition_target_bars_show_percentages_and_class_counts(config, frame):
    from telco_churn.visualization.validation import plot_partition_target_balance

    summary = TemporalSplitter(config, "target").split(frame).summary(config, "target")
    figure = plot_partition_target_balance(summary)
    assert figure.layout.barmode == "group"
    assert all(trace.textposition == "outside" for trace in figure.data)
    assert list(figure.data[0].x) == ["Train", "Validation", "Test"]
    assert list(figure.data[0].y) == [50, 50, 50]
    assert list(figure.data[1].y) == [50, 50, 50]
    assert list(figure.data[1].customdata[0]) == [1, 2]
    figure.to_json(validate=True)
