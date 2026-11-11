"""Contrôles qualité sur anomalies synthétiques et configurations invalides."""

import pandas as pd
import pytest

from telco_churn.data.schema import DatasetRules, NumericRule, ValidationConfig
from telco_churn.data.validation import DataValidator
from telco_churn.settings import Settings


@pytest.fixture
def rules():
    return DatasetRules(
        required_columns=["id", "date", "signup", "amount", "segment"],
        id_column="id",
        unique_key=["id", "date"],
        key_note="Une ligne par date.",
        numeric={"amount": NumericRule(minimum=0, note="Montant positif.")},
        categories={"segment": ["web", "store"]},
        dates={"date": "%Y-%m-%d", "signup": "%Y-%m-%d"},
        date_order=[("signup", "date")],
        missingness_columns=["amount"],
        group_columns=["segment"],
    )


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["2024-01-31"] * 3,
            "signup": ["2023-01-01"] * 3,
            "amount": [10.0, 20.0, 30.0],
            "segment": ["web", "store", "web"],
            "outcome": [0, 1, 0],
        }
    )


def check(result, rule, column):
    return result.loc[result.rule.eq(rule) & result.column.eq(column)].iloc[0]


def test_clean_data_and_source_unchanged(rules, frame):
    original = frame.copy(deep=True)
    result = DataValidator(rules, "outcome").assess(frame)
    assert result.status.eq("PASS").all()
    pd.testing.assert_frame_equal(frame, original)


def test_domains_dates_and_formats(rules, frame):
    frame.loc[0, "amount"] = float("inf")
    frame.loc[1, "amount"] = -1
    frame.loc[0, "segment"] = " WEB "
    frame.loc[0, "signup"] = "bad-date"
    frame.loc[1, "signup"] = "2025-01-01"
    frame.loc[2, "outcome"] = 2
    result = DataValidator(rules, "outcome").assess(frame)
    assert check(result, "numeric_domain", "amount").n_failed == 2
    assert check(result, "numeric_domain", "outcome").n_failed == 1
    assert check(result, "category_format", "segment").n_failed == 1
    assert check(result, "date_parse", "signup").n_failed == 1
    order = check(result, "date_order", "signup, date")
    assert order.n_checked == 2
    assert order.n_failed == 1
    assert order.failed_percent == 50


def test_duplicates_and_non_unique_index_use_positions(rules, frame):
    frame = pd.concat([frame, frame.iloc[[0]]])  # index 0 répété
    result = DataValidator(rules, "outcome").assess(frame)
    assert check(result, "duplicate_rows", ", ".join(frame.columns)).n_failed == 1
    key = check(result, "unique_key", "id, date")
    assert key.n_failed == 2
    assert [example["row_position"] for example in key.examples] == [0, 3]


def test_missing_blanks_and_conditional_rates(rules, frame):
    frame["amount"] = [None, None, 10.0]
    frame["segment"] = [" ", None, "web"]
    frame.loc[0, "id"] = None
    result = DataValidator(rules, "outcome").assess(frame)
    assert check(result, "missing_values", "segment").n_failed == 2
    assert check(result, "numeric_domain", "amount").n_checked == 1
    assert check(result, "unique_key", "id, date").n_checked == 2
    table = DataValidator(rules, "outcome").missingness_by_group(frame, "segment")
    assert table.n_rows.sum() == 3
    assert table.n_missing.sum() == 2


def test_missing_columns_and_empty_source_are_not_success(rules, frame):
    validator = DataValidator(rules, "outcome")
    result = validator.assess(frame.drop(columns=["amount", "date"]))
    assert check(result, "required_columns", "schema").n_failed == 2
    assert check(result, "numeric_domain", "amount").status == "SKIP"
    assert check(result, "unique_key", "id, date").status == "SKIP"
    result = validator.assess(frame.iloc[:0])
    assert check(result, "non_empty_dataset", "dataset").status == "FAIL"
    assert check(result, "numeric_domain", "amount").status == "SKIP"


def test_numeric_strings_and_temporal_coverage(rules, frame):
    frame["amount"] = ["12", "bad", "5.5"]
    validator = DataValidator(rules, "outcome")
    result = validator.assess(frame)
    assert check(result, "numeric_type", "amount").n_failed == 3
    assert check(result, "numeric_domain", "amount").n_failed == 1
    coverage = validator.temporal_coverage(frame).set_index("column")
    assert coverage.loc["date", "n_distinct"] == 1
    assert coverage.loc["signup", "n_valid"] == 3


def test_config_rejects_invalid_bounds_and_unknown_columns():
    with pytest.raises(ValueError):
        NumericRule(minimum=10, maximum=1, note="Invalid")
    with pytest.raises(ValueError):
        DatasetRules(required_columns=["id"], id_column="missing", unique_key=["id"], key_note="")


def test_config_load_is_portable_and_targets_remain_in_settings(monkeypatch, tmp_path):
    settings = Settings.load()
    monkeypatch.chdir(tmp_path)
    config = ValidationConfig.load(settings)
    assert set(config.datasets) == {"churn", "pilot"}
    assert settings.churn_target_column not in config.datasets["churn"].numeric
    assert config.datasets["churn"].unique_key == ["customer_id", "snapshot_date"]


def test_duplicate_column_names_fail_explicitly(rules):
    with pytest.raises(ValueError, match="noms de colonnes dupliqués"):
        DataValidator(rules, "outcome").assess(pd.DataFrame([[1, 2]], columns=["id", "id"]))


def test_uppercase_canonical_category_is_not_an_anomaly(rules, frame):
    rules.categories["segment"] = ["WEB", "STORE"]
    frame["segment"] = ["WEB", "STORE", "web "]
    result = DataValidator(rules, "outcome").assess(frame)
    row = check(result, "category_format", "segment")
    assert row.n_failed == 1
    assert row.examples[0]["row_position"] == 2
