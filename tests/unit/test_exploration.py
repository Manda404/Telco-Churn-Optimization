"""Vérifier les dénominateurs et les cas limites des statistiques exploratoires."""

import numpy as np
import pandas as pd
import pytest

from telco_churn.data.exploration import ChurnExplorer, ExplorationConfig
from telco_churn.settings import Settings


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "customer": [1, 1, 2, 3],
            "date": ["2024-01-31", "2024-02-29", "2024-01-31", "2024-02-29"],
            "target": [0, 1, None, 0],
            "segment": ["a", "a", "b", None],
            "amount": [10.0, 20.0, 30.0, np.inf],
        }
    )


def test_population_distinguishes_clients_rows_and_targets(frame):
    summary = ChurnExplorer("target", "customer").population_summary(frame)
    assert summary["n_rows"] == 4
    assert summary["n_customers"] == 3
    assert summary["n_target_observed"] == 3
    assert summary["n_target_missing"] == 1
    assert summary["n_events"] == 1
    assert summary["churn_rate_percent"] == pytest.approx(100 / 3)


def test_segments_keep_missing_groups_and_use_observed_target(frame):
    table = ChurnExplorer("target", "customer", min_group_size=2).churn_by_group(frame, "segment")
    group = table.loc[table.segment.eq("a")].iloc[0]
    assert group.n_observed == 2
    assert group.churn_rate_percent == 50
    assert not group.small_group
    assert table.segment.isna().sum() == 1
    assert table.n_observed.sum() == 3


def test_temporal_counts_and_invalid_dates(frame):
    explorer = ChurnExplorer("target", "customer")
    table = explorer.temporal_churn(frame, "date", "%Y-%m-%d")
    assert table.n_observed.tolist() == [1, 2]
    assert table.churn_rate_percent.tolist() == [0.0, 50.0]
    frame.loc[0, "date"] = "bad"
    with pytest.raises(ValueError, match="dates absentes/invalides"):
        explorer.temporal_churn(frame, "date", "%Y-%m-%d")


def test_numeric_view_excludes_nonfinite_values_without_mutation(frame):
    original = frame.copy(deep=True)
    explorer = ChurnExplorer("target", "customer")
    statistics = explorer.numeric_summary(frame, ["amount"]).set_index("target_value")
    assert statistics.loc[0, "n_finite"] == 1
    assert statistics.loc[0, "n_excluded"] == 1
    assert statistics.loc[1, "median"] == 20
    pd.testing.assert_frame_equal(frame, original)


def test_correlations_report_pairwise_counts(frame):
    frame["other"] = [1, None, 3, 4]
    matrix, counts = ChurnExplorer("target", "customer").correlations(
        frame, ["amount", "other"], min_pairs=2
    )
    assert counts.loc["amount", "target"] == 2
    assert counts.loc["amount", "other"] == 1
    assert np.isnan(matrix.loc["amount", "other"])
    assert matrix.loc["amount", "target"] == pytest.approx(1)


def test_invalid_targets_and_identifier_in_numeric_list(frame):
    explorer = ChurnExplorer("target", "customer")
    with pytest.raises(ValueError, match="Cible et identifiant"):
        explorer.numeric_values(frame, ["customer"])
    frame.loc[0, "target"] = 2
    with pytest.raises(ValueError, match="uniquement 0, 1"):
        explorer.population_summary(frame)


def test_empty_sources_do_not_report_zero_churn(frame):
    empty = frame.iloc[:0]
    explorer = ChurnExplorer("target", "customer")
    assert np.isnan(explorer.population_summary(empty)["churn_rate_percent"])
    assert explorer.churn_by_group(empty, "segment").empty
    assert explorer.numeric_summary(empty, ["amount"]).empty
    assert explorer.temporal_churn(empty, "date", "%Y-%m-%d").empty


def test_composition_uses_all_rows_including_missing_target(frame):
    table = ChurnExplorer("target", "customer").category_composition(frame, "segment")
    assert table.n_rows.sum() == 4
    assert table.share_percent.sum() == pytest.approx(100)
    assert table.loc[table.segment.eq("b"), "share_percent"].iloc[0] == 25


def test_config_is_portable_and_validates_identifiers(monkeypatch, tmp_path):
    settings = Settings.load()
    monkeypatch.chdir(tmp_path)
    config = ExplorationConfig.load(settings)
    assert config.id_column not in config.numeric_columns
    with pytest.raises(ValueError):
        ExplorationConfig.model_validate(
            {**config.model_dump(), "numeric_columns": [config.id_column]}
        )


def test_numeric_bands_include_boundaries_and_keep_invalid_values():
    from telco_churn.data.exploration import numeric_bands
    from telco_churn.data.schema import NumericRule

    series = pd.Series([0, 5, 7, 9, 10, 11, -1, np.inf, None, " ", "bad"])
    bands = numeric_bands(series, [0, 5, 7, 9], NumericRule(minimum=0, maximum=10, note="NPS"))
    assert bands.iloc[:5].astype(str).tolist() == ["[0, 5)", "[5, 7)", "[7, 9)", "≥ 9", "≥ 9"]
    assert bands.iloc[5:8].eq("Hors domaine / invalide").all()
    assert bands.iloc[8:10].eq("Non renseigné").all()
    assert bands.iloc[10] == "Hors domaine / invalide"
    assert not bands.isna().any()


def test_tenure_view_reports_bad_dates_and_preserves_source():
    config = ExplorationConfig.load(Settings.load())
    frame = pd.DataFrame(
        {
            config.id_column: [1, 2, 3],
            "outcome": [1, 0, 0],
            config.snapshot_column: ["2024-01-31"] * 3,
            config.signup_column: ["2023-01-31", "2024-02-01", "invalid"],
        }
    )
    original = frame.copy(deep=True)
    view, audit = ChurnExplorer("outcome", config.id_column).analytical_tenure_view(frame, config)
    assert audit == {
        "n_rows": 3,
        "n_dates_absent_or_invalid": 1,
        "n_negative_tenure": 1,
        "n_valid_tenure": 1,
    }
    assert view.loc[0, config.tenure_column] == pytest.approx(365 / 30.4375)
    assert view.loc[1, config.tenure_band_column] == "Hors domaine / invalide"
    assert view.loc[2, config.tenure_band_column] == "Non renseigné"
    pd.testing.assert_frame_equal(frame, original)


def test_interaction_denominators_and_zero_event_behavior(frame):
    explorer = ChurnExplorer("target", "customer", min_group_size=2)
    table = explorer.interaction_rates(frame, ["segment"])
    assert table.n_observed.sum() == 3
    assert table.n_events.sum() == 1
    assert table.event_share_percent.sum() == pytest.approx(100)
    group = table.loc[table.segment.eq("a")].iloc[0]
    assert group.rate_ratio == pytest.approx(1.5)
    assert group.delta_pp == pytest.approx(50 - 100 / 3)
    frame["target"] = 0
    no_events = explorer.interaction_rates(frame, ["segment"])
    assert no_events.rate_ratio.isna().all()
    assert no_events.event_share_percent.isna().all()


def test_missingness_signal_counts_blank_strings_and_missing_targets(frame):
    frame["amount"] = [" ", 12, None, 5]
    table = ChurnExplorer("target", "customer").missingness_churn(frame, ["amount"])
    assert table.n_observed.sum() == 3
    assert table.loc[table.availability.eq("Non renseigné"), "n_observed"].iloc[0] == 1
    assert table.loc[table.availability.eq("Renseigné"), "n_events"].iloc[0] == 1


def test_history_uses_distinct_positive_dates_and_reports_conflicts():
    frame = pd.DataFrame(
        {
            "customer": [1, 1, 1, 1, 2, 2],
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-01-01",
                "2024-01-01",
            ],
            "target": [1, 1, 1, 0, 0, 1],
        }
    )
    histories, summary = ChurnExplorer("target", "customer").history_diagnostics(
        frame, "date", "%Y-%m-%d"
    )
    assert histories.loc[histories.customer.eq(1), "n_positive_dates"].iloc[0] == 2
    assert summary["n_customers_positive_on_multiple_dates"] == 1
    assert summary["n_customers_observed_after_first_positive"] == 1
    assert summary["n_customer_date_target_conflicts"] == 1
    assert summary["n_customers_positive_at_least_once"] == 2


def test_new_diagnostics_handle_empty_sources(frame):
    explorer = ChurnExplorer("target", "customer")
    assert explorer.interaction_rates(frame.iloc[:0], ["segment"]).empty
    history, summary = explorer.history_diagnostics(frame.iloc[:0], "date", "%Y-%m-%d")
    assert history.empty
    assert summary["n_customers_positive_on_multiple_dates"] == 0
    assert np.isnan(summary["mean_client_snapshot_churn_percent"])


@pytest.mark.parametrize("edges", [[0], [0, 0], [1, 0], [0, np.inf]])
def test_bad_band_configuration_is_rejected(edges):
    config = ExplorationConfig.load(Settings.load())
    with pytest.raises(ValueError, match="bornes|tenure_bins"):
        ExplorationConfig.model_validate({**config.model_dump(), "tenure_bins": edges})


def test_plotly_heatmap_masks_small_cells_and_serializes(frame):
    from telco_churn.visualization.eda_interactive import plot_interaction_churn

    table = ChurnExplorer("target", "customer", min_group_size=2).interaction_rates(
        frame, ["segment", "date"]
    )
    figure = plot_interaction_churn(table, "segment", "date", title="Test croisement")
    z = np.asarray(figure.data[0].z, dtype=float)
    assert np.isnan(z).all()  # chaque cellule a un seul élément à cible observée
    assert "Test croisement" in figure.to_json()


def test_age_distribution_combines_box_histogram_and_density_without_mutation():
    from telco_churn.visualization.eda_interactive import plot_age_distribution

    ages = pd.Series([20, 25, 30, 35, 40, 90, 999, None])
    original = ages.copy(deep=True)
    figure = plot_age_distribution(ages, minimum=18, maximum=120)
    assert [trace.type for trace in figure.data] == ["box", "histogram", "scatter"] * 2
    assert figure.data[0].boxpoints == "outliers"
    assert figure.layout.xaxis.matches == "x2"
    assert figure.layout.meta["n_outside_domain"] == 1
    assert figure.layout.meta["n_non_numeric_or_nonfinite"] == 1
    assert max(figure.data[0].x) == 90
    assert max(figure.data[3].x) == 999
    assert np.isfinite(figure.data[2].y).all()
    figure.to_json(validate=True)
    pd.testing.assert_series_equal(ages, original)


def test_age_distribution_handles_constant_values_without_fabricating_a_curve():
    from telco_churn.visualization.eda_interactive import plot_age_distribution

    figure = plot_age_distribution(pd.Series([30, 30, 30]), minimum=18, maximum=120)
    assert len(figure.data[2].x) == 0
    assert "KDE non estimable" in figure.layout.title.text


def test_age_distribution_rejects_empty_numeric_population():
    from telco_churn.visualization.eda_interactive import plot_age_distribution

    with pytest.raises(ValueError, match="Aucun âge numérique fini"):
        plot_age_distribution(pd.Series([None, float("inf"), "bad"]), minimum=18, maximum=120)


def test_contract_tenure_lines_reuse_heatmap_rates_and_leave_small_groups_as_gaps():
    from telco_churn.visualization.eda_interactive import plot_contract_tenure_lines

    table = pd.DataFrame(
        {
            "contract": ["mensuel", "mensuel", "Mensuel"],
            "tenure": pd.Categorical(
                ["[0, 3)", "[3, 6)", "[0, 3)"], categories=["[0, 3)", "[3, 6)"], ordered=True
            ),
            "churn_rate_percent": [12.65, 8.1, 99.0],
            "small_group": [False, True, False],
            "n_observed": [1000, 8, 100],
            "n_customers": [800, 7, 70],
            "n_events": [126, 1, 99],
        }
    )
    figure = plot_contract_tenure_lines(
        table, "contract", "tenure", ["mensuel", "annuel", "biannuel"]
    )
    assert figure.data[0].y[0] == 12.65  # la variante Mensuel n'est pas réagrégée
    assert np.isnan(figure.data[0].y[1])
    assert not figure.data[0].connectgaps
    assert list(figure.data[0].x) == ["0–3", "3–6"]
    assert np.isnan(figure.data[1].y).all()
    figure.to_json(validate=True)
