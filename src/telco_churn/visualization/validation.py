"""Figures dédiées aux fenêtres temporelles et aux diagnostics de drift."""

from datetime import datetime, time, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from telco_churn.modeling.temporal_split import DriftConfig, TemporalConfig
from telco_churn.visualization.eda_interactive import style_figure


def plot_split_timeline(summary: pd.DataFrame, config: TemporalConfig) -> go.Figure:
    """Frise unique : mois calendaires, fenêtres configurées et intervalles hors split.

    La ligne suit les bornes du YAML, pas seulement les dates des snapshots présents.
    Les effectifs restent ceux des partitions réellement conservées après contrôles.
    """
    month_names = [
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
    start = datetime(config.train_start.year, 1, 1)
    end = datetime(config.test_end.year + 1, 1, 1)
    ticks, labels = [], []
    current = start
    while current < end:
        following = (
            datetime(current.year + 1, 1, 1)
            if current.month == 12
            else datetime(current.year, current.month + 1, 1)
        )
        ticks.append(current + (following - current) / 2)
        labels.append(
            month_names[current.month - 1]
            if start.year == end.year - 1
            else f"{month_names[current.month - 1]} {current.year}"
        )
        current = following
    figure = go.Figure(
        go.Scatter(
            x=[start, end],
            y=[0, 0],
            mode="lines",
            name="Hors split",
            line=dict(color="#CBD5E1", width=14),
            hoverinfo="skip",
        )
    )
    partitions = [
        ("train", "TRAIN", config.train_start, config.train_end, "#2563EB"),
        ("validation", "VALIDATION", config.validation_start, config.validation_end, "#F59E0B"),
        ("test", "TEST", config.test_start, config.test_end, "#10B981"),
    ]
    counts = summary.set_index("partition")
    spans = []
    for name, label, first, last, color in partitions:
        left = datetime.combine(first, time.min)
        right = datetime.combine(last, time.min) + timedelta(days=1)
        spans.append((left, right))
        row = counts.loc[name]
        hover = (
            f"{label}<br>{first:%d/%m/%Y} → {last:%d/%m/%Y}"
            f"<br>{int(row.n_rows):,} lignes<br>{int(row.n_clients):,} clients distincts"
        )
        figure.add_trace(
            go.Scatter(
                x=[left, right],
                y=[0, 0],
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=14),
                marker=dict(color=color, size=14),
                text=[hover, hover],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        figure.add_annotation(
            x=left + (right - left) / 2,
            y=-0.28,
            text=f"<b>{label}</b><br>{int(row.n_rows):,} lignes".replace(",", " "),
            showarrow=False,
            font=dict(color=color, size=13),
        )
    for (_, right), (left, _) in zip(spans, spans[1:], strict=False):
        if right < left:
            figure.add_annotation(
                x=right + (left - right) / 2,
                y=-0.28,
                text="Hors<br>split",
                showarrow=False,
                font=dict(color="#64748B", size=11),
            )
    years = (
        str(config.train_start.year)
        if config.train_start.year == config.test_end.year
        else f"{config.train_start.year}–{config.test_end.year}"
    )
    figure = style_figure(figure, f"Découpage du dataset — {years}", height=350)
    figure.update_xaxes(
        type="date",
        range=[start, end],
        tickmode="array",
        tickvals=ticks,
        ticktext=labels,
        side="top",
        showgrid=True,
        gridcolor="#F1F5F9",
        zeroline=False,
        title=None,
        fixedrange=False,
    )
    figure.update_yaxes(visible=False, range=[-0.8, 0.5], fixedrange=True)
    figure.update_layout(
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12), hovermode="closest"
    )
    return figure


def plot_drift_summary(report: pd.DataFrame, config: DriftConfig) -> go.Figure:
    """Comparer les PSI ; couleurs selon l'ensemble des règles de drift."""
    figure = px.bar(
        report,
        x="psi",
        y="feature",
        color="severity",
        facet_col="comparison",
        orientation="h",
        color_discrete_map={"NO_ALERT": "#2563EB", "WARNING": "#F59E0B", "HIGH": "#EF4444"},
        hover_data=[
            "total_variation",
            "missing_delta_pp",
            "unseen_percent_of_nonmissing",
            "outside_train_range_percent",
        ],
    )
    figure.add_vline(x=config.psi_warning, line_dash="dash", line_color="#F59E0B")
    figure.add_vline(x=config.psi_high, line_dash="dot", line_color="#EF4444")
    figure.update_xaxes(title="PSI — seuils heuristiques", rangemode="tozero")
    figure.update_yaxes(title=None)
    return style_figure(
        figure, "Drift par rapport au train — bornes apprises sur train uniquement", height=750
    )


def plot_reference_distribution(table: pd.DataFrame, feature: str) -> go.Figure:
    """Comprendre un écart avec des groupes fixes définis sur la référence train."""
    figure = px.bar(
        table,
        x="bin",
        y="percent",
        color="partition",
        barmode="group",
        color_discrete_map={"train": "#2563EB", "validation": "#F59E0B", "test": "#10B981"},
    )
    figure.update_xaxes(title="Intervalles de référence train")
    figure.update_yaxes(title="Part des lignes (%)")
    return style_figure(
        figure,
        f"Distribution de {feature} — mêmes intervalles pour les trois partitions",
        height=500,
    )


def plot_partition_target_balance(summary: pd.DataFrame) -> go.Figure:
    """Deux barres côte à côte par partition : proportions et effectifs au survol."""
    labels = (
        summary["partition"]
        .map({"train": "Train", "validation": "Validation", "test": "Test"})
        .fillna(summary["partition"])
    )
    totals = summary["n_rows"]
    positives = summary["n_positive_snapshots"]
    if positives.lt(0).any() or positives.gt(totals).any() or totals.lt(0).any():
        raise ValueError("Les effectifs de cible doivent être compris entre zéro et le total.")
    denominator = totals.where(totals.gt(0))
    figure = go.Figure()
    for name, counts, color in [
        ("Non-churn (0)", totals - positives, "#2563EB"),
        ("Churn (1)", positives, "#EF4444"),
    ]:
        percentages = counts / denominator * 100
        figure.add_trace(
            go.Bar(
                x=labels,
                y=percentages,
                name=name,
                marker_color=color,
                customdata=pd.DataFrame({"count": counts, "total": totals}).to_numpy(),
                text=percentages.map(lambda value: f"{value:.2f} %" if pd.notna(value) else ""),
                textposition="outside",
                textfont=dict(color=color, size=14),
                cliponaxis=False,
                hovertemplate=(
                    "%{x}<br>%{fullData.name} : %{y:.2f} %"
                    "<br>Snapshots : %{customdata[0]:,.0f}<br>Total partition : %{customdata[1]:,.0f}<extra></extra>"
                ),
            )
        )
    figure = style_figure(figure, "Répartition du churn et du non-churn par partition", height=440)
    figure.update_layout(
        barmode="group",
        bargap=0.25,
        bargroupgap=0.04,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.18),
    )
    figure.update_xaxes(title=None, categoryorder="array", categoryarray=labels.tolist())
    figure.update_yaxes(
        title="Part des snapshots (%)",
        range=[0, 110],
        tickvals=[0, 25, 50, 75, 100],
        ticksuffix=" %",
    )
    return figure
