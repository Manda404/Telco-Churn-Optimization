"""Graphiques exploratoires construits à partir des statistiques du package."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from telco_churn.visualization.quality import COLORS


def save_figure(fig: Figure, save_path: str | Path | None = None) -> Figure:
    """Sauvegarder si demandé ; l'appelant contrôle l'affichage et la fermeture."""
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_churn_trend(
    table: pd.DataFrame,
    date_column: str,
    *,
    title: str,
    save_path: str | Path | None = None,
) -> Figure:
    """Afficher séparément les effectifs et le taux observé par date."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(
        table[date_column], table["n_observed"], marker="o", label="Lignes à cible observée"
    )
    axes[0].plot(
        table[date_column], table["n_customers"], linestyle="--", label="Clients distincts"
    )
    axes[0].set_ylabel("Effectif")
    axes[0].legend()
    axes[1].plot(
        table[date_column], table["churn_rate_percent"], marker="o", color=COLORS["default"]
    )
    axes[1].set(ylabel="Churn observé (%)", xlabel="Date de photographie")
    axes[1].set_ylim(bottom=0)
    fig.suptitle(title, fontweight="bold")
    return save_figure(fig, save_path)


def plot_segment_churn(
    table: pd.DataFrame,
    column: str,
    *,
    title: str,
    save_path: str | Path | None = None,
) -> Figure:
    """Afficher les taux par modalité avec effectifs et petits groupes en gris."""
    fig, ax = plt.subplots(figsize=(11, max(4, len(table) * 0.4)))
    labels = table[column].astype("string").fillna("<manquant>").map(repr)
    colors = ["#9CA3AF" if small else COLORS["main"] for small in table["small_group"]]
    bars = ax.barh(labels, table["churn_rate_percent"], color=colors)
    for bar, count in zip(bars, table["n_observed"], strict=True):
        ax.annotate(
            f"n={count:,}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.invert_yaxis()
    ax.margins(x=0.25)
    ax.set(xlabel="Churn observé (%) — petits groupes en gris", ylabel=column, title=title)
    return save_figure(fig, save_path)


def plot_numeric_by_target(
    values: pd.DataFrame,
    target_column: str,
    columns: list[str],
    *,
    title: str,
    showfliers: bool = False,
    save_path: str | Path | None = None,
) -> Figure:
    """Comparer les distributions finies ; les effectifs figurent dans les labels."""
    if not columns:
        raise ValueError("Au moins une variable numérique est nécessaire.")
    fig, axes = plt.subplots(
        (len(columns) + 1) // 2, 2, figsize=(12, 3.5 * ((len(columns) + 1) // 2)), squeeze=False
    )
    for ax, column in zip(axes.flat, columns, strict=False):
        groups, labels = [], []
        for target in (0, 1):
            group = values.loc[values[target_column].eq(target), column].dropna()
            if not group.empty:
                groups.append(group.to_numpy())
                labels.append(f"{target_column}={target}\nn={len(group):,}")
        if groups:
            # set_xticks reste compatible avec les versions matplotlib 3.9+.
            ax.boxplot(groups, showfliers=showfliers)
            ax.set_xticks(range(1, len(labels) + 1), labels)
        else:
            ax.text(0.5, 0.5, "Aucune valeur finie", ha="center", transform=ax.transAxes)
        ax.set_title(column)
    for ax in list(axes.flat)[len(columns) :]:
        ax.axis("off")
    suffix = "" if showfliers else "\nPoints hors moustaches masqués à l'affichage uniquement"
    fig.suptitle(title + suffix, fontweight="bold")
    return save_figure(fig, save_path)


def plot_correlations(
    correlations: pd.DataFrame,
    *,
    title: str,
    save_path: str | Path | None = None,
) -> Figure:
    """Matrice Pearson ; les corrélations non définies sont grisées."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#E5E7EB")
    matrix = correlations.to_numpy(dtype=float)
    image = ax.imshow(np.ma.masked_invalid(matrix), vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(correlations)), correlations.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlations)), correlations.index)
    for row in range(len(correlations)):
        for col in range(len(correlations)):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}" if np.isfinite(value) else "NA",
                ha="center",
                va="center",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, label="Corrélation de Pearson")
    ax.set_title(title)
    return save_figure(fig, save_path)
