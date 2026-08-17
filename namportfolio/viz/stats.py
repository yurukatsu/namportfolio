"""統計的頑健性の図。

table = npf.stats.subsample(returns, n_splits=4, statistic="sharpe_ratio")
npf.viz.plot_subsample(table, metric="sharpe_ratio")

dist = npf.stats.bootstrap_distribution(ic, n_boot=10_000)
npf.viz.plot_bootstrap(dist, observed=ic.mean())
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..core.errors import ValidationError
from ..core.panel import require_columns
from . import theme

__all__ = [
    "plot_subsample",
    "plot_regime",
    "plot_bootstrap",
]


def plot_subsample(
    table: pd.DataFrame,
    *,
    metric: str = "mean",
    title: str | None = None,
    ax=None,
) -> Figure:
    """サブサンプルごとの統計量。

    **符号が揃っているか**を見る図。全期間で有意でも、特定の区間だけで
    稼いでいれば棒の向きが揃わない。正負は発散配色の両極で描く。

    Parameters
    ----------
    table :
        :func:`namportfolio.stats.subsample` の出力。
    metric :
        棒の高さに使う列。
    """
    require_columns(table, [metric], context="plot_subsample")
    values = pd.to_numeric(table[metric])

    if {"start", "end"}.issubset(table.columns):
        labels = [
            f"{pd.Timestamp(s):%Y-%m}\n{pd.Timestamp(e):%Y-%m}"
            for s, e in zip(table["start"], table["end"], strict=True)
        ]
    else:
        labels = [str(i) for i in table.index]

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        positions = np.arange(len(values))
        ax.bar(
            positions,
            values.to_numpy(),
            width=0.62,
            color=theme.polarity_colors(values.to_numpy()),
        )
        ax.set_xticks(positions, labels)
        ax.grid(axis="x", visible=False)
        theme.finalize(
            ax,
            title=title or f"{metric} by sub-period",
            ylabel=metric,
            zero_line=True,
        )
    return fig


def plot_regime(
    table: pd.DataFrame,
    *,
    metric: str = "mean",
    show_share: bool = True,
    title: str | None = None,
    ax=None,
) -> Figure:
    """局面ごとの統計量。

    Parameters
    ----------
    table :
        :func:`namportfolio.stats.regime_summary` の出力。
    show_share :
        ``True`` なら各棒にその局面が占める期間の割合を添える。**サンプルの
        少ない局面の数字を鵜呑みにしないため。**
    """
    require_columns(table, [metric], context="plot_regime")
    values = pd.to_numeric(table[metric])

    with theme.styled():
        colors = theme.palette()
        fig, ax = theme.new_axes(ax)
        positions = np.arange(len(values))
        ax.bar(
            positions,
            values.to_numpy(),
            width=0.58,
            color=theme.polarity_colors(values.to_numpy()),
        )

        if show_share and "share" in table.columns:
            offset = np.nanmax(np.abs(values.to_numpy())) * 0.04
            for position, value, share in zip(
                positions, values.to_numpy(), table["share"], strict=True
            ):
                ax.text(
                    position,
                    value + (offset if value >= 0 else -offset),
                    f"{share:.0%}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                    color=colors["muted"],
                )

        ax.set_xticks(positions, [str(i) for i in values.index])
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
        theme.finalize(
            ax,
            title=title or f"{metric} by regime",
            ylabel=metric,
            zero_line=True,
        )
    return fig


def plot_bootstrap(
    distribution: np.ndarray | pd.Series,
    *,
    observed: float | None = None,
    interval: tuple[float, float] | None = None,
    null_value: float = 0.0,
    bins: int = 60,
    title: str | None = "Bootstrap distribution",
    ax=None,
) -> Figure:
    """ブートストラップ分布と信頼区間。

    **区間が帰無値（既定 0）をまたいでいなければ有意**、という読み方をする。

    Parameters
    ----------
    distribution :
        :func:`namportfolio.stats.bootstrap_distribution` の出力。
    observed :
        実際に観測された統計量。縦線で示す。
    interval :
        信頼区間。``None`` なら描かない
        （:func:`namportfolio.stats.bootstrap_ci` の戻り値をそのまま渡せる）。
    """
    values = np.asarray(distribution, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        raise ValidationError("ブートストラップ分布が空です。")

    with theme.styled():
        colors = theme.palette()
        primary, accent = theme.categorical_colors(2)
        fig, ax = theme.new_axes(ax)

        ax.hist(values, bins=bins, color=primary, edgecolor=colors["surface"], linewidth=0.5)

        if interval is not None:
            ax.axvspan(interval[0], interval[1], color=colors["grid"], alpha=0.9, zorder=0)
        ax.axvline(null_value, color=colors["axis"], linewidth=1.0)
        if observed is not None:
            ax.axvline(float(observed), color=accent, linewidth=1.6)
            ax.annotate(
                f" observed {observed:.4g}",
                xy=(float(observed), ax.get_ylim()[1]),
                xytext=(4, -4),
                textcoords="offset points",
                color=accent,
                fontsize=9,
                va="top",
            )

        ax.grid(axis="x", visible=False)
        theme.finalize(ax, title=title, xlabel="Statistic", ylabel="Resamples")
    return fig
