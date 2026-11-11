"""Contrats des features : indépendance du label, qualité et scoring reproductible."""

import numpy as np
import pandas as pd
import pytest

from telco_churn.data.schema import ValidationConfig
from telco_churn.features.config import FeatureConfig
from telco_churn.features.engineering import FeatureEngineer
from telco_churn.settings import Settings


@pytest.fixture
def config():
    return FeatureConfig.load(Settings.load())


@pytest.fixture
def engineer(config):
    settings = Settings.load()
    return FeatureEngineer(
        config, ValidationConfig.load(settings).datasets["churn"], settings.churn_target_column
    )


@pytest.fixture
def frame(config):
    frame = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in config.numeric_columns})
    frame[config.id_column] = [101, 102, 103]
    frame[config.snapshot_column] = ["2024-01-31"] * 3
    frame[config.signup_column] = ["2023-01-31", "2024-02-01", "bad"]
    frame["contract"] = [" MENSUEL ", None, "new_contract"]
    frame["region"] = ["idf", " ARA ", "UNKNOWN"]
    frame["channel"] = ["Web", " boutique ", None]
    frame["churn"] = [0, 1, 0]
    frame["age"] = [20, 30, 40]
    frame["last_invoice_amount"] = [10, 20, 30]
    frame["retention_offer_declined"] = [0, 1, 1]
    return frame


def test_rows_index_and_source_are_preserved(engineer, frame):
    frame.index = [4, 4, 8]
    original = frame.copy(deep=True)
    result = engineer.transform(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert result.index.equals(frame.index)
    assert len(result) == len(frame)
    assert set(result.columns).isdisjoint(
        {
            "churn",
            "customer_id",
            "snapshot_date",
            "signup_date",
            "age",
            "last_invoice_amount",
            "retention_offer_declined",
        }
    )


def test_target_permutation_and_absence_do_not_change_features(engineer, frame):
    expected = engineer.transform(frame)
    changed = frame.copy()
    changed["churn"] = 1 - frame.churn
    pd.testing.assert_frame_equal(engineer.transform(changed), expected)
    pd.testing.assert_frame_equal(engineer.transform(frame.drop(columns="churn")), expected)


def test_transform_is_independent_of_other_rows(engineer, frame):
    full = engineer.transform(frame)
    for index in frame.index:
        pd.testing.assert_frame_equal(engineer.transform(frame.loc[[index]]), full.loc[[index]])


def test_invalid_numeric_values_and_missingness_remain_distinct(engineer, frame):
    frame["monthly_fee"] = [None, -10, "not-a-number"]
    frame["nps"] = [None, 11, 10]
    result = engineer.transform(frame)
    assert result.monthly_fee.isna().all()
    assert result.monthly_fee_missing.tolist() == [1, 0, 0]
    assert result.monthly_fee_invalid.tolist() == [0, 1, 1]
    assert result.nps.isna().tolist() == [True, True, False]
    assert result.nps_missing.tolist() == [1, 0, 0]
    assert result.nps_invalid.tolist() == [0, 1, 0]


def test_category_normalization_unknowns_and_missing_values(engineer, frame, config):
    result = engineer.transform(frame)
    assert result.contract.iloc[0] == "mensuel"
    assert pd.isna(result.contract.iloc[1])
    assert result.contract.iloc[2] == config.unknown_category
    assert result.region.tolist() == ["IDF", "ARA", config.unknown_category]


def test_tenure_uses_snapshot_and_rejects_bad_chronology(engineer, frame, config):
    result = engineer.transform(frame)
    assert result.tenure_months.iloc[0] == pytest.approx(365 / config.days_per_month)
    assert result.tenure_months.iloc[1:].isna().all()
    assert result.tenure_unavailable.tolist() == [0, 1, 1]
    assert result.log1p_tenure_months.iloc[0] == pytest.approx(
        np.log1p(365 / config.days_per_month)
    )


def test_interaction_propagates_unknown_inputs(engineer, frame):
    frame["support_calls_3m"] = [2, None, 0]
    frame["late_payments"] = [3, 2, None]
    result = engineer.transform(frame)
    assert result.support_payment_interaction.iloc[0] == 6
    assert result.support_payment_interaction.iloc[1:].isna().all()


def test_features_keep_finite_extremes_without_learning_a_threshold(engineer, frame):
    frame["data_gb"] = [0, 32154, np.inf]
    result = engineer.transform(frame)
    assert result.data_gb.iloc[1] == 32154
    assert pd.isna(result.data_gb.iloc[2])
    assert "log1p_data_gb" not in result


def test_catalog_and_availability_gate(engineer, frame):
    result = engineer.transform(frame)
    catalog = engineer.catalog()
    assert result.columns.tolist() == catalog.feature.tolist()
    assert catalog.availability.eq("unconfirmed").all()
    assert engineer.quality_summary(result).index.tolist() == result.columns.tolist()
    with pytest.raises(ValueError, match="Disponibilité temporelle non confirmée"):
        engineer.assert_temporal_availability()


def test_missing_sources_and_duplicate_column_names_fail(engineer, frame):
    with pytest.raises(ValueError, match="Sources manquantes"):
        engineer.transform(frame.drop(columns="signup_date"))
    invalid = pd.concat([frame, frame[["contract"]]], axis=1)
    with pytest.raises(ValueError, match="noms de colonnes"):
        engineer.transform(invalid)


@pytest.mark.parametrize("column", ["age", "last_invoice_amount", "retention_offer_declined"])
def test_protected_sources_cannot_be_enabled(config, column):
    values = config.model_dump()
    values["numeric_columns"].append(column)
    with pytest.raises(ValueError, match="protégée"):
        FeatureConfig.model_validate(values)


def test_configuration_loads_from_another_directory(monkeypatch, tmp_path):
    settings = Settings.load()
    monkeypatch.chdir(tmp_path)
    assert FeatureConfig.load(settings).id_column == "customer_id"


def test_candidate_export_has_manifest_and_does_not_overwrite(engineer, frame, tmp_path):
    import json

    from telco_churn.features.artifacts import export_candidates

    features = engineer.transform(frame)
    metadata = frame[[engineer.config.id_column, engineer.config.snapshot_column, "churn"]]
    destination = export_candidates(features, metadata, engineer.catalog(), tmp_path)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["training_validated"] is False
    assert manifest["n_features"] == len(features.columns)
    assert manifest["n_rows"] == len(frame)
    assert "churn" not in manifest["feature_columns"]
    assert "churn" in manifest["metadata_columns"]
    assert len(pd.read_csv(destination / "features.csv")) == len(frame)
    with pytest.raises(FileExistsError):
        export_candidates(features, metadata, engineer.catalog(), tmp_path)


def test_export_refuses_misaligned_rows(engineer, frame, tmp_path):
    from telco_churn.features.artifacts import export_candidates

    features = engineer.transform(frame)
    with pytest.raises(ValueError, match="même index"):
        export_candidates(
            features, frame[[engineer.config.id_column]].iloc[::-1], engineer.catalog(), tmp_path
        )


def test_age_groups_are_candidates_while_raw_age_stays_excluded(engineer, frame):
    frame["age"] = [18, 25, 80]
    features = engineer.transform(frame)
    assert "age" not in features
    assert features.age_group.tolist() == ["[18, 25)", "[25, 35)", "≥ 80"]
    row = engineer.catalog().set_index("feature").loc["age_group"]
    assert row.sources == "age"
    assert row.availability == "unconfirmed"


def test_age_groups_keep_invalid_and_missing_ages_as_missing(engineer, frame):
    frame["age"] = [-1, 999, None]
    assert engineer.transform(frame).age_group.isna().all()


def test_age_grouping_can_be_disabled_and_does_not_require_age(config, frame):
    values = config.model_dump()
    values["age_grouping"] = False
    disabled = FeatureConfig.model_validate(values)
    settings = Settings.load()
    engineer = FeatureEngineer(
        disabled, ValidationConfig.load(settings).datasets["churn"], settings.churn_target_column
    )
    assert "age_group" not in engineer.transform(frame.drop(columns="age"))


def test_age_bins_must_be_ordered(config):
    values = config.model_dump()
    values["age_bins"] = [25, 18]
    with pytest.raises(ValueError, match="bornes d'âge"):
        FeatureConfig.model_validate(values)


def test_processed_dataset_is_saved_with_context_and_can_be_refreshed(engineer, frame, tmp_path):
    import json

    from telco_churn.features.artifacts import save_processed_dataset

    features = engineer.transform(frame)
    metadata = frame[[engineer.config.id_column, engineer.config.snapshot_column, "churn"]]
    path = save_processed_dataset(features, metadata, engineer.catalog(), tmp_path)
    saved = pd.read_csv(path)
    assert saved.columns.tolist() == [*metadata.columns, *features.columns]
    assert len(saved) == len(frame)
    assert "age" not in saved and "age_group" in saved
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    assert "churn" not in manifest["feature_columns"]
    assert manifest["training_validated"] is False
    assert save_processed_dataset(features, metadata, engineer.catalog(), tmp_path) == path


def test_processed_export_refuses_misaligned_rows(engineer, frame, tmp_path):
    from telco_churn.features.artifacts import save_processed_dataset

    with pytest.raises(ValueError, match="même index"):
        save_processed_dataset(
            engineer.transform(frame),
            frame[["customer_id"]].iloc[::-1],
            engineer.catalog(),
            tmp_path,
        )
