"""Dénominateurs, bootstrap de clients et vues analytiques réversibles."""

import numpy as np
import pandas as pd
import pytest

from telco_churn.data.eda_diagnostics import (
    analytical_view,
    clustered_rates,
    eligible_view,
    regional_composition,
    standardized_monthly_rates,
)
from telco_churn.data.exploration import ChurnExplorer, ExplorationConfig
from telco_churn.data.schema import EDAOptions, ValidationConfig
from telco_churn.settings import Settings


def options(**overrides):
    return EDAOptions(
        **{
            "min_clients": 2,
            "min_events_ci": 1,
            "bootstrap_repetitions": 100,
            "interaction_bins": {},
            "interaction_pairs": {},
            **overrides,
        }
    )


def test_bootstrap_resamples_clients_and_preserves_whole_history():
    frame = pd.DataFrame(
        {"id": np.repeat(range(8), 10), "target": np.repeat([0, 1] * 4, 10), "group": "A"}
    )
    original = frame.copy(deep=True)
    explorer = ChurnExplorer("target", "id", min_group_size=1)
    result = clustered_rates(explorer, frame, ["group"], options())
    doubled = clustered_rates(explorer, pd.concat([frame, frame]), ["group"], options())
    row = result.iloc[0]
    assert row.n_observed == 80
    assert row.n_events == 40
    assert row.n_positive_customers == 4
    assert row.churn_rate_percent == 50
    assert row.ci_low_percent < row.ci_high_percent
    assert row.ci_status == "bootstrap clients"
    assert doubled.iloc[0].ci_low_percent == row.ci_low_percent
    assert doubled.iloc[0].ci_high_percent == row.ci_high_percent
    pd.testing.assert_frame_equal(frame, original)


def test_large_row_count_does_not_replace_distinct_client_support():
    frame = pd.DataFrame({"id": [1] * 120, "target": [1] * 120, "group": "A"})
    row = clustered_rates(ChurnExplorer("target", "id", 100), frame, ["group"], options()).iloc[0]
    assert row.small_group
    assert np.isnan(row.ci_low_percent)


def test_missing_target_excluded_but_zero_event_ci_not_fabricated():
    frame = pd.DataFrame({"id": [1, 2, 3], "target": [0, 0, None], "group": "A"})
    row = clustered_rates(ChurnExplorer("target", "id", 1), frame, ["group"], options()).iloc[0]
    assert row.n_observed == 2
    assert row.churn_rate_percent == 0
    assert row.n_positive_customers == 0
    assert np.isnan(row.ci_high_percent)


def test_eligibility_retains_missing_stratum_only_when_explicit():
    frame = pd.DataFrame(
        {"nps": ["Non renseigné", "[0, 5)", "Hors domaine / invalide"], "support": ["0", "0", "0"]}
    )
    kept, audit = eligible_view(frame, ["nps", "support"], keep_missing=["nps"])
    assert len(kept) == 2
    assert audit == {"n_input": 3, "n_eligible": 2, "n_excluded": 1}


def test_normalized_view_keeps_source_and_reports_unknown_categories():
    settings = Settings.load()
    config = ExplorationConfig.load(settings)
    rules = ValidationConfig.load(settings).datasets["churn"]
    # Minimal frame includes the columns used by configured EDA operations.
    frame = pd.DataFrame({column: [1, 1] for column in config.deep_eda.interaction_bins})
    frame[config.age_column] = [30, 999]
    for column, allowed in rules.categories.items():
        frame[column] = [allowed[0], f" {allowed[0].upper()} "]
    frame.loc[0, "channel"] = "unknown"
    original = frame.copy(deep=True)
    view, audit = analytical_view(frame, config, rules)
    assert view.loc[1, "eda_contract"] == "mensuel"
    assert audit.set_index("column").loc["channel", "n_invalid"] == 1
    pd.testing.assert_frame_equal(frame, original)


def test_composition_uses_rows_while_reporting_distinct_clients():
    frame = pd.DataFrame(
        {"age_band": ["young", "young", "young"], "region": ["A", "A", "B"], "id": [1, 1, 2]}
    )
    table = regional_composition(frame, "age_band", "region", "id")
    assert table.share_within_age_percent.sum() == pytest.approx(100)
    assert table.loc[table.region.eq("A"), "n_customers"].iloc[0] == 1
    assert table.loc[table.region.eq("A"), "n_rows"].iloc[0] == 2


def test_standardization_distinguishes_rate_change_from_composition_change():
    rows = []
    customer = 0
    for date, n_a, n_b in [("2024-01-31", 90, 10), ("2024-02-29", 10, 90)]:
        for contract, count, fraction in [("A", n_a, 0.1), ("B", n_b, 0.5)]:
            for i in range(count):
                rows.append(
                    {
                        "id": customer,
                        "date": date,
                        "contract": contract,
                        "target": int(i < count * fraction),
                    }
                )
                customer += 1
    table, weights = standardized_monthly_rates(
        ChurnExplorer("target", "id", 1), pd.DataFrame(rows), "date", "contract", "%Y-%m-%d", 2
    )
    assert table.churn_rate_percent.tolist() == pytest.approx([14, 46])
    assert table.standardized_rate_percent.tolist() == pytest.approx([30, 30])
    assert weights.reference_weight.sum() == pytest.approx(1)


def test_metric_menu_masks_rates_but_keeps_counts():
    from telco_churn.visualization.eda_interactive import plot_exploratory_grid

    table = pd.DataFrame(
        {
            "row": ["A"],
            "col": ["B"],
            "churn_rate_percent": [100.0],
            "small_group": [True],
            "n_observed": [120],
            "n_customers": [1],
            "n_events": [120],
            "n_positive_customers": [1],
        }
    )
    fig = plot_exploratory_grid(table, "row", "col", title="Test")
    assert np.isnan(np.asarray(fig.data[0].z, dtype=float)).all()
    assert np.asarray(fig.data[1].z)[0, 0] == 120
    fig.to_json(validate=True)
