"""Graphiques Plotly pour les axes complémentaires de l'EDA du dataset principal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {"primary": "#2563EB", "churn": "#EF4444", "muted": "#94A3B8", "dark": "#0F172A"}
HOVER_COLUMNS = [
    "n_observed",
    "n_customers",
    "n_events",
    "delta_pp",
    "rate_ratio",
    "event_share_percent",
]


def style_figure(fig: go.Figure, title: str, height: int = 470) -> go.Figure:
    """Appliquer une identité visuelle commune sans modifier les réglages globaux."""
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.02, font=dict(size=19, color=COLORS["dark"])),
        font=dict(family="Arial", size=12, color=COLORS["dark"]),
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=60, r=35, t=95, b=75),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


def plot_diagnostic_rates(
    table: pd.DataFrame,
    column: str,
    *,
    baseline_rate: float,
    title: str,
) -> go.Figure:
    """Taux par bande avec effectifs, événements et comparaison au taux brut."""
    colors = [
        COLORS["muted"]
        if small or str(label) in ("Non renseigné", "Hors domaine / invalide")
        else COLORS["primary"]
        for label, small in zip(table[column], table.small_group, strict=True)
    ]
    fig = go.Figure(
        go.Bar(
            x=table[column].astype(str),
            y=table.churn_rate_percent,
            marker_color=colors,
            text=table.churn_rate_percent,
            texttemplate="%{text:.2f} %",
            textposition="outside",
            customdata=table[HOVER_COLUMNS].to_numpy(),
            hovertemplate=(
                "%{x}<br>Churn : %{y:.2f} %<br>Lignes : %{customdata[0]:,.0f}"
                "<br>Clients distincts : %{customdata[1]:,.0f}<br>Lignes positives : %{customdata[2]:,.0f}"
                "<br>Écart : %{customdata[3]:+.2f} points<br>Ratio descriptif : %{customdata[4]:.2f}<br>Part des positifs : %{customdata[5]:.1f} %<extra></extra>"
            ),
        )
    )
    if np.isfinite(baseline_rate):
        fig.add_hline(
            y=baseline_rate,
            line_dash="dash",
            line_color=COLORS["churn"],
            annotation_text=f"Référence brute : {baseline_rate:.2f} %",
        )
    fig.update_xaxes(
        title=column,
        type="category",
        categoryorder="array",
        categoryarray=table[column].astype(str).tolist(),
    )
    fig.update_yaxes(title="Churn observé (%)", rangemode="tozero", ticksuffix=" %")
    return style_figure(fig, title + "<br><sup>Gris : petit groupe ou statut de qualité</sup>")


def plot_interaction_churn(
    table: pd.DataFrame,
    row_column: str,
    column_column: str,
    *,
    title: str,
) -> go.Figure:
    """Croisement de taux ; cellules à faible effectif masquées, sans classement."""
    rates = table.pivot(index=row_column, columns=column_column, values="churn_rate_percent")
    counts = table.pivot(index=row_column, columns=column_column, values="n_observed").reindex_like(
        rates
    )
    small = table.pivot(index=row_column, columns=column_column, values="small_group").reindex_like(
        rates
    )
    events = table.pivot(index=row_column, columns=column_column, values="n_events").reindex_like(
        rates
    )
    z = rates.mask(small.astype("boolean").fillna(True)).to_numpy(dtype=float)
    x, y = rates.columns.astype(str).tolist(), [repr(value) for value in rates.index]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Blues",
            colorbar=dict(title="Churn (%)"),
            customdata=np.stack([counts.to_numpy(), events.to_numpy()], axis=-1),
            text=[[f"{value:.2f} %" if np.isfinite(value) else "" for value in row] for row in z],
            texttemplate="%{text}",
            hoverongaps=False,
            hovertemplate=(
                "%{y} × %{x}<br>Churn : %{z:.2f} %<br>Lignes : %{customdata[0]:,.0f}"
                "<br>Lignes positives : %{customdata[1]:,.0f}<extra></extra>"
            ),
        )
    )
    for row in range(len(y)):
        for col in range(len(x)):
            if np.isfinite(counts.iloc[row, col]) and not np.isfinite(z[row, col]):
                fig.add_annotation(
                    x=x[col],
                    y=y[row],
                    text=f"n={int(counts.iloc[row, col])}<br>petit groupe",
                    showarrow=False,
                    font=dict(size=9, color="#64748B"),
                )
    fig.update_layout(plot_bgcolor="#F1F5F9")
    fig.update_xaxes(title=column_column, type="category", categoryorder="array", categoryarray=x)
    fig.update_yaxes(title=row_column, type="category", categoryorder="array", categoryarray=y)
    fig = style_figure(fig, title, max(450, len(y) * 40 + 160))
    fig.update_layout(plot_bgcolor="#F1F5F9")
    return fig


def plot_missingness_churn(table: pd.DataFrame, *, baseline_rate: float, title: str) -> go.Figure:
    """Comparer les taux selon la disponibilité des valeurs, sans répéter les taux de manquants."""
    fig = px.bar(
        table,
        x="variable",
        y="churn_rate_percent",
        color="availability",
        barmode="group",
        text="churn_rate_percent",
        custom_data=[*HOVER_COLUMNS, "small_group"],
        color_discrete_map={"Renseigné": COLORS["primary"], "Non renseigné": COLORS["churn"]},
    )
    fig.update_traces(
        texttemplate="%{text:.2f} %",
        textposition="outside",
        hovertemplate=(
            "%{x}<br>Churn : %{y:.2f} %<br>Lignes : %{customdata[0]:,.0f}"
            "<br>Clients : %{customdata[1]:,.0f}<br>Lignes positives : %{customdata[2]:,.0f}"
            "<br>Écart : %{customdata[3]:+.2f} points<br>Petit groupe : %{customdata[6]}<extra>%{fullData.name}</extra>"
        ),
    )
    if np.isfinite(baseline_rate):
        fig.add_hline(y=baseline_rate, line_dash="dash", line_color=COLORS["dark"])
    fig.update_yaxes(title="Churn observé (%)", rangemode="tozero", ticksuffix=" %")
    return style_figure(fig, title)


def plot_history_kpis(summary: dict[str, int | float], *, title: str) -> go.Figure:
    """Indicateurs propres à la trajectoire ; pas de répétition des KPI de population."""
    indicators = [
        ("Clients positifs à ≥ 2 dates", "n_customers_positive_on_multiple_dates"),
        ("Clients observés après un positif", "n_customers_observed_after_first_positive"),
        ("Clés client/date à cibles contradictoires", "n_customer_date_target_conflicts"),
    ]
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "indicator"}] * 3])
    for index, (label, key) in enumerate(indicators, 1):
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=summary[key],
                title=dict(text=label, font=dict(size=12)),
                number=dict(valueformat=",.0f", font=dict(size=35, color=COLORS["dark"])),
            ),
            row=1,
            col=index,
        )
    return style_figure(fig, title, height=260)


def plot_age_distribution(
    ages: pd.Series,
    *,
    minimum: float | None,
    maximum: float | None,
    title: str = "Churn Dataset — distribution des âges et boxplot",
    bins: int = 30,
) -> go.Figure:
    """Histogramme, KDE non paramétrique et boxplot horizontal sur un axe partagé.

    Deux vues : domaine proposé et toutes les valeurs finies. Le choix ne modifie
    pas la source. Les points du boxplot suivent la règle 1,5 IQR ; les valeurs
    hors domaine métier et les valeurs atypiques statistiques sont distinctes.
    """
    from scipy.stats import gaussian_kde

    if bins < 2:
        raise ValueError("Au moins deux classes sont nécessaires pour l'histogramme.")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("Le minimum doit être inférieur ou égal au maximum.")
    numeric = (
        pd.to_numeric(ages, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    if numeric.empty:
        raise ValueError("Aucun âge numérique fini disponible pour le graphique.")
    valid = pd.Series(True, index=numeric.index)
    if minimum is not None:
        valid &= numeric.ge(minimum)
    if maximum is not None:
        valid &= numeric.le(maximum)
    views = [("Domaine proposé", numeric.loc[valid]), ("Toutes les valeurs finies", numeric)]
    initial = 0 if valid.any() else 1
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.22, 0.78], vertical_spacing=0.05
    )
    buttons, summaries = [], []
    for index, (label, sample) in enumerate(views):
        visible = index == initial
        x_grid, density = np.array([]), np.array([])
        if sample.nunique() > 1:
            padding = (sample.max() - sample.min()) * 0.05
            x_grid = np.linspace(sample.min() - padding, sample.max() + padding, 512)
            density = gaussian_kde(sample.to_numpy())(x_grid)
        q1, q3 = sample.quantile([0.25, 0.75])
        iqr = q3 - q1
        n_outliers = int((sample.lt(q1 - 1.5 * iqr) | sample.gt(q3 + 1.5 * iqr)).sum())
        summaries.append({"view": label, "n_observed": len(sample), "n_outliers_iqr": n_outliers})
        fig.add_trace(
            go.Box(
                x=sample.to_numpy(),
                orientation="h",
                name="Boxplot — points atypiques",
                boxpoints="outliers",
                quartilemethod="linear",
                showlegend=False,
                marker=dict(color=COLORS["churn"], size=5),
                line=dict(color=COLORS["primary"]),
                visible=visible,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=sample.to_numpy(),
                nbinsx=bins,
                histnorm="probability density",
                name="Histogramme",
                marker_color=COLORS["primary"],
                opacity=0.35,
                visible=visible,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=density,
                mode="lines",
                name="Densité lissée (KDE)",
                line=dict(color=COLORS["primary"], width=3),
                visible=visible,
                hovertemplate="Âge : %{x:.1f}<br>Densité : %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        subtitle = f"{label} · n={len(sample):,} · {n_outliers:,} points atypiques (1,5 IQR)"
        if sample.nunique() < 2:
            subtitle += " · courbe KDE non estimable (dispersion insuffisante)"
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[
                    {"visible": [index == trace // 3 for trace in range(6)]},
                    {
                        "title.text": title + "<br><sup>" + subtitle + "</sup>",
                        "xaxis.autorange": True,
                        "xaxis2.autorange": True,
                    },
                ],
            )
        )
    fig = style_figure(fig, buttons[initial]["args"][1]["title.text"], height=640)
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, active=initial, x=1, y=1.15, xanchor="right")],
        legend=dict(orientation="h", y=-0.18),
        meta={
            "n_input": len(ages),
            "n_non_numeric_or_nonfinite": len(ages) - len(numeric),
            "n_outside_domain": int((~valid).sum()),
            "views": summaries,
        },
    )
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title="Densité", row=2, col=1)
    fig.update_xaxes(title="Âge (années)", row=2, col=1)
    return fig


def plot_contract_tenure_lines(
    table: pd.DataFrame,
    contract_column: str,
    tenure_column: str,
    contracts: list[str],
    *,
    title: str = "Churn Dataset — taux de churn selon l'ancienneté",
) -> go.Figure:
    """Compléter la heatmap avec les mêmes cellules et le même masquage.

    Sélection exacte des libellés demandés, sans normalisation ni réagrégation.
    Les groupes absents ou sous le seuil interrompent les lignes.
    """
    if table.duplicated([contract_column, tenure_column]).any():
        raise ValueError("Une seule cellule par contrat et tranche d'ancienneté est attendue.")
    observed = table[tenure_column].dropna().unique()
    if isinstance(table[tenure_column].dtype, pd.CategoricalDtype):
        order = [value for value in table[tenure_column].cat.categories if value in observed]
    else:
        order = list(observed)
    labels = [str(value).replace("[", "").replace(")", "").replace(", ", "–") for value in order]
    colors = ["#4F95FF", "#64C877", "#FF8752"]
    fig = go.Figure()
    for index, contract in enumerate(contracts):
        subset = (
            table.loc[table[contract_column].eq(contract)].set_index(tenure_column).reindex(order)
        )
        masked = subset.small_group.astype("boolean").fillna(True)
        rates = subset.churn_rate_percent.mask(masked)
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=rates.to_numpy(dtype=float),
                name=contract.capitalize(),
                mode="lines+markers",
                connectgaps=False,
                line=dict(color=colors[index % len(colors)], width=2),
                marker=dict(size=7),
                customdata=subset[["n_observed", "n_customers", "n_events"]].to_numpy(),
                hovertemplate=(
                    "Ancienneté : %{x} mois<br>Churn : %{y:.2f} %"
                    "<br>Observations : %{customdata[0]:,.0f}<br>Clients distincts : %{customdata[1]:,.0f}"
                    "<br>Lignes positives : %{customdata[2]:,.0f}<extra>%{fullData.name}</extra>"
                ),
            )
        )
    fig = style_figure(
        fig,
        title + "<br><sup>Mêmes cellules que la heatmap · petits groupes masqués</sup>",
        height=490,
    )
    fig.update_xaxes(
        title="Ancienneté au snapshot (mois)",
        type="category",
        categoryorder="array",
        categoryarray=labels,
        showgrid=False,
    )
    fig.update_yaxes(
        title="Churn observé (%)", ticksuffix=" %", rangemode="tozero", gridcolor="#E2E8F0"
    )
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.2))
    return fig


def plot_exploratory_grid(
    table: pd.DataFrame,
    row_column: str,
    column_column: str,
    *,
    title: str,
    metrics: dict[str, str] | None = None,
) -> go.Figure:
    """Heatmap multi-KPI : ne jamais utiliser une couleur de volume comme un taux.

    Les petits groupes sont masqués pour les taux, mais restent consultables par
    les compteurs. Les IC concernent le taux marginal du groupe, pas un effet causal.
    """
    metrics = metrics or {
        "churn_rate_percent": "Churn (%)",
        "n_events": "Snapshots positifs",
        "n_positive_customers": "Clients avec un positif",
        "n_observed": "Observations",
        "n_customers": "Clients distincts",
    }
    axis_labels = {
        "eda_support_calls_3m_band": "Appels au support (fenêtre 3m à confirmer)",
        "eda_late_payments_band": "Retards de paiement (nombre)",
        "eda_monthly_fee_band": "Montant d'abonnement (unité à confirmer)",
        "eda_data_gb_band": "Consommation de données (Go supposés)",
        "eda_nps_band": "Note de satisfaction (0–10, à confirmer)",
        "eda_age_band": "Tranche d'âge (années)",
        "eda_contract": "Type de contrat",
        "eda_region": "Région",
        "eda_tenure_band": "Ancienneté au snapshot (mois)",
        "snapshot_date": "Date du snapshot",
    }
    first = next(iter(metrics))
    grid = table.pivot(index=row_column, columns=column_column, values=first)
    x, y = grid.columns.astype(str).tolist(), grid.index.astype(str).tolist()
    custom_columns = [
        column
        for column in [
            "n_observed",
            "n_rows",
            "n_customers",
            "n_events",
            "n_positive_customers",
            "ci_low_percent",
            "ci_high_percent",
        ]
        if column in table
    ]
    custom = np.stack(
        [
            table.pivot(index=row_column, columns=column_column, values=column)
            .reindex_like(grid)
            .to_numpy(dtype=float)
            for column in custom_columns
        ],
        axis=-1,
    )
    labels = {
        "n_observed": "Observations à cible connue",
        "n_rows": "Observations",
        "n_customers": "Clients distincts",
        "n_events": "Snapshots positifs",
        "n_positive_customers": "Clients avec un positif",
        "ci_low_percent": "IC bas (%)",
        "ci_high_percent": "IC haut (%)",
    }
    hover = "<br>".join(
        f"{labels[column]} : %{{customdata[{index}]:{'.2f' if column.startswith('ci_') else ',.0f'}}}"
        for index, column in enumerate(custom_columns)
    )
    buttons = []
    for index, (metric, label) in enumerate(metrics.items()):
        values = table.pivot(index=row_column, columns=column_column, values=metric).reindex_like(
            grid
        )
        if metric in ("churn_rate_percent", "delta_pp") and "small_group" in table:
            small = table.pivot(
                index=row_column, columns=column_column, values="small_group"
            ).reindex_like(grid)
            values = values.mask(small.astype("boolean").fillna(True))
        z = values.to_numpy(dtype=float)
        fig_trace = go.Heatmap(
            z=z,
            x=x,
            y=y,
            customdata=custom,
            visible=index == 0,
            colorscale="RdBu_r" if metric == "delta_pp" else "Blues",
            zmid=0 if metric == "delta_pp" else None,
            colorbar=dict(title=label),
            hoverongaps=False,
            text=[[f"{value:.1f}" if np.isfinite(value) else "" for value in row] for row in z],
            texttemplate="%{text}",
            hovertemplate=f"%{{y}} × %{{x}}<br>{label} : %{{z:.2f}}<br>{hover}<extra></extra>",
        )
        if index == 0:
            fig = go.Figure()
        fig.add_trace(fig_trace)
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[{"visible": [j == index for j in range(len(metrics))]}],
            )
        )
    fig = style_figure(
        fig,
        title + "<br><sup>Menu : taux et volumes séparés · IC par clients quand disponibles</sup>",
        height=max(500, len(y) * 38 + 180),
    )
    fig.update_layout(
        updatemenus=[dict(buttons=buttons, x=1, y=1.16, xanchor="right")], plot_bgcolor="#F1F5F9"
    )
    fig.update_xaxes(
        title=axis_labels.get(column_column, column_column),
        type="category",
        categoryorder="array",
        categoryarray=x,
    )
    fig.update_yaxes(
        title=axis_labels.get(row_column, row_column),
        type="category",
        categoryorder="array",
        categoryarray=y,
    )
    return fig


def plot_temporal_comparison(table: pd.DataFrame, date_column: str, *, title: str) -> go.Figure:
    """Comparer un taux brut à une composition contractuelle fixe sur le même champ."""
    fig = go.Figure()
    for column, label, color in [
        ("churn_rate_percent", "Taux brut", COLORS["primary"]),
        ("standardized_rate_percent", "Composition contractuelle fixe", COLORS["churn"]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=table[date_column],
                y=table[column],
                mode="lines+markers",
                name=label,
                connectgaps=False,
                line=dict(color=color),
                customdata=table[["n_observed", "n_customers"]],
                hovertemplate="%{x}<br>Taux : %{y:.2f} %<br>Lignes : %{customdata[0]:,.0f}<br>Clients : %{customdata[1]:,.0f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_xaxes(title="Date du snapshot")
    fig.update_yaxes(title="Churn par snapshot (%)", rangemode="tozero")
    return style_figure(fig, title)


def plot_eda_time(table: pd.DataFrame, date_column: str, *, title: str) -> go.Figure:
    """Effectifs et taux, avec des axes distincts et des unités explicites."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    for column, label in [
        ("n_observed", "Lignes à cible connue"),
        ("n_customers", "Clients distincts"),
    ]:
        fig.add_trace(
            go.Scatter(x=table[date_column], y=table[column], mode="lines+markers", name=label),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=table[date_column],
            y=table.churn_rate_percent,
            name="Taux de churn",
            mode="lines+markers",
            line=dict(color=COLORS["churn"]),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title="Effectifs", row=1, col=1)
    fig.update_yaxes(title="Churn (%)", row=2, col=1)
    fig.update_xaxes(title="Date du snapshot", row=2, col=1)
    return style_figure(fig, title, height=650)


def plot_numeric_eda(
    values: pd.DataFrame, target: str, columns: list[str], *, showfliers: bool, title: str
) -> go.Figure:
    """Comparer les distributions numériques sans mélange d'échelles entre variables."""
    long = values.melt(id_vars=target, value_vars=columns, var_name="variable", value_name="value")
    long["statut"] = long[target].map({0: "Cible 0", 1: "Cible 1"})
    fig = px.box(
        long,
        x="statut",
        y="value",
        color="statut",
        facet_col="variable",
        facet_col_wrap=2,
        points="outliers" if showfliers else False,
        color_discrete_map={"Cible 0": COLORS["primary"], "Cible 1": COLORS["churn"]},
    )
    fig.update_yaxes(matches=None, title="Valeur de la variable")
    fig.update_xaxes(title=None)
    fig.update_layout(showlegend=False)
    return style_figure(
        fig,
        title + "<br><sup>Échelles propres à chaque variable · données finies uniquement</sup>",
        height=1000,
    )


def plot_eda_correlations(matrix: pd.DataFrame, counts: pd.DataFrame, *, title: str) -> go.Figure:
    """Pearson, avec les effectifs disponibles par paire au survol."""
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
            customdata=counts.to_numpy(),
            text=matrix.round(2).to_numpy(),
            texttemplate="%{text}",
            colorbar=dict(title="Pearson r"),
            hoverongaps=False,
            hovertemplate="%{y} × %{x}<br>r : %{z:.3f}<br>Paires disponibles : %{customdata:,.0f}<extra></extra>",
        )
    )
    return style_figure(fig, title, height=650)
