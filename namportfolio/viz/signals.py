"""シグナル診断の図。

入力は :mod:`namportfolio.signals` の出力、または long DataFrame。

    cov = npf.signals.coverage(df, factor="value")
    npf.viz.plot_coverage(cov)

    npf.viz.plot_distribution(df, factor="value", compare="value_z")
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from ..core.errors import ValidationError
from ..core.panel import require_columns
from . import theme

__all__ = [
    "plot_coverage",
    "plot_distribution",
    "plot_distribution_stats",
    "plot_signal_correlation",
    "plot_signal_exposure",
    "plot_explained_ratio",
]


def plot_coverage(
    coverage: pd.DataFrame,
    *,
    metric: str = "n_valid",
    title: str | None = None,
    ax=None,
) -> Figure:
    """有効銘柄数（または被覆率）の推移。

    シグナルが「いつから使えるか」「途中でデータが抜けていないか」を見る図。
    分位分析の結果が特定期間だけおかしいときに最初に見る。

    Parameters
    ----------
    coverage :
        :func:`namportfolio.signals.coverage` の出力。
    metric :
        描く列。``"n_valid"`` / ``"n_total"`` / ``"missing_rate"`` / ``"coverage_rate"``。
        件数と比率は尺度が違うので、1 枚に重ねず列を選んで描く。
    """
    require_columns(coverage, [metric], context="plot_coverage")
    is_rate = metric.endswith("_rate")

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        color = theme.categorical_colors(1)[0]
        series = coverage[metric]
        ax.plot(series.index, series.to_numpy(), color=color)
        ax.fill_between(series.index, series.to_numpy(), 0.0, color=color, alpha=0.15)

        if is_rate:
            theme.percent_axis(ax)
            ax.set_ylim(0, max(1.0, float(series.max()) * 1.05))
        else:
            ax.set_ylim(0, float(series.max()) * 1.1)
        theme.finalize(
            ax,
            title=title or f"Signal {metric.replace('_', ' ')}",
            ylabel=metric.replace("_", " "),
        )
    return fig


def plot_distribution(
    data: pd.DataFrame,
    *,
    factor: str,
    compare: str | None = None,
    bins: int = 60,
    title: str | None = None,
    ax=None,
) -> Figure:
    """シグナル値の分布。

    前処理の効き具合を確認する図。``compare`` に処理後の列を渡すと重ねて比較できる
    （塗りつぶすと重なりが読めないので、2 本のときは輪郭線で描く）。

    Parameters
    ----------
    compare :
        比較する列名（``"value"`` に対する ``"value_z"`` など）。

    Notes
    -----
    全期間をプールして描くので、**winsorize の効果は控えめに見える**。境界は
    日付ごとに計算されるため、ある日の 1% 点が別の日の中央付近ということが
    起きるためで、処理が効いていないわけではない。期間ごとの裾の動きは
    :func:`plot_distribution_stats` で ``kurtosis`` を見るほうが分かりやすい。
    """
    require_columns(data, [factor], context="plot_distribution")
    if compare is not None:
        require_columns(data, [compare], context="plot_distribution")

    with theme.styled():
        colors = theme.palette()
        fig, ax = theme.new_axes(ax)
        primary, secondary = theme.categorical_colors(2)

        if compare is None:
            ax.hist(
                data[factor].dropna().to_numpy(),
                bins=bins,
                color=primary,
                edgecolor=colors["surface"],
                linewidth=0.5,
            )
        else:
            # 2 本のヒストグラムは同じビン境界で切る。系列ごとに範囲から
            # 決めさせると、幅の違いで高さが変わり比較が成立しない
            both = pd.concat([data[factor], data[compare]]).dropna().to_numpy()
            edges = np.histogram_bin_edges(both, bins=bins)
            for column, color in ((factor, primary), (compare, secondary)):
                ax.hist(
                    data[column].dropna().to_numpy(),
                    bins=edges,
                    histtype="step",
                    linewidth=1.8,
                    color=color,
                    label=column,
                )
            ax.legend(loc="best")

        ax.grid(axis="x", visible=False)
        theme.finalize(
            ax,
            title=title or f"Distribution of {factor}",
            xlabel=factor,
            ylabel="Observations",
        )
    return fig


def plot_distribution_stats(
    summary: pd.DataFrame,
    *,
    metric: str = "skew",
    title: str | None = None,
    ax=None,
) -> Figure:
    """分布の形の推移（歪度・尖度など）。

    値が跳ねている期間は外れ値が入っている可能性が高い。

    Parameters
    ----------
    summary :
        :func:`namportfolio.signals.distribution_summary` の出力。
    metric :
        描く列（``"skew"`` / ``"kurtosis"`` / ``"std"`` など）。
    """
    require_columns(summary, [metric], context="plot_distribution_stats")

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        color = theme.categorical_colors(1)[0]
        series = summary[metric]
        ax.plot(series.index, series.to_numpy(), color=color)
        theme.finalize(
            ax,
            title=title or f"Cross-sectional {metric}",
            ylabel=metric,
            zero_line=True,
        )
    return fig


def plot_signal_exposure(
    summary: pd.DataFrame,
    *,
    metric: str = "mean",
    significance: float = 2.0,
    title: str | None = "Signal exposure to existing factors",
    ax=None,
) -> Figure:
    """シグナルの既存ファクターへの曝露。

    **有意でない曝露を薄く描く。** 曝露の大きさだけを並べると、たまたま大きく
    振れただけのファクターが目立ってしまう。

    Parameters
    ----------
    summary :
        :func:`namportfolio.signals.exposure_summary` の出力。
    metric :
        棒の長さに使う列（既定は期間平均）。
    significance :
        濃く描く t 値の閾値。``summary`` に ``t_stat_nw`` があればそれを、
        無ければ ``t_stat`` を使う。
    """
    require_columns(summary, [metric], context="plot_signal_exposure")
    values = summary[metric].sort_values()

    t_column = next((c for c in ("t_stat_nw", "t_stat") if c in summary.columns), None)
    if t_column is None:
        significant = np.ones(len(values), dtype=bool)
    else:
        significant = summary[t_column].reindex(values.index).abs().to_numpy() >= significance

    with theme.styled():
        fig, ax = theme.new_axes(ax, figsize=(8.0, max(2.4, 0.34 * len(values) + 1.2)))
        base = theme.polarity_colors(values.to_numpy())
        faces = [
            to_rgba(color, 1.0 if is_significant else 0.3)
            for color, is_significant in zip(base, significant, strict=True)
        ]
        ax.barh(np.arange(len(values)), values.to_numpy(), height=0.72, color=faces)

        ax.set_yticks(np.arange(len(values)), [str(i) for i in values.index])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.axvline(0.0, color=theme.palette()["axis"], linewidth=0.8)
        theme.finalize(
            ax,
            title=title,
            xlabel=f"{metric}  (solid = |t| >= {significance:g})",
        )
    return fig


def plot_explained_ratio(
    ratio: pd.Series,
    *,
    title: str | None = "Variance explained by existing factors",
    ax=None,
) -> Figure:
    """シグナルの分散のうち既存ファクターで説明される割合の推移。

    高止まりしていれば、そのシグナルは既存ファクターの組み合わせで再現できる。
    """
    if isinstance(ratio, pd.DataFrame):
        raise ValidationError("plot_explained_ratio は Series のみ対応です。")

    values = ratio.dropna()
    with theme.styled():
        colors = theme.palette()
        fig, ax = theme.new_axes(ax)
        color = theme.categorical_colors(1)[0]

        ax.plot(values.index, values.to_numpy(), color=color)
        ax.fill_between(values.index, values.to_numpy(), 0.0, color=color, alpha=0.15)

        mean = float(values.mean())
        ax.axhline(mean, color=colors["ink_secondary"], linewidth=1.0)
        ax.annotate(
            f" mean {mean:.0%}",
            xy=(values.index[-1], mean),
            xytext=(4, 0),
            textcoords="offset points",
            color=colors["ink_secondary"],
            fontsize=9,
            va="center",
        )
        ax.margins(x=0.02)
        ax.set_ylim(0, 1)
        theme.percent_axis(ax)
        theme.finalize(ax, title=title, ylabel="R²")
    return fig


def plot_signal_correlation(
    correlation: pd.DataFrame,
    *,
    labels: Sequence[str] | None = None,
    title: str | None = "Signal correlation",
    ax=None,
) -> Figure:
    """シグナル間相関のヒートマップ。

    相関は正負の極性を持つので発散配色。スケールは ±1 に固定し、図ごとに
    濃さの意味が変わらないようにする。
    """
    if correlation.shape[0] != correlation.shape[1]:
        raise ValidationError(f"正方行列を渡してください: {correlation.shape}")
    names = list(labels) if labels is not None else list(correlation.index)
    matrix = correlation.to_numpy(dtype=float)

    with theme.styled():
        size = max(3.2, 0.6 * len(names) + 1.8)
        fig, ax = theme.new_axes(ax, figsize=(size + 1.2, size))
        mesh = theme.heatmap(
            ax,
            matrix,
            xticks=names,
            yticks=names,
            cmap=theme.diverging_cmap(),
            vmin=-1.0,
            vmax=1.0,
            annotate=True,
            fmt="{:.2f}",
        )
        theme.add_colorbar(fig, mesh, ax, ticks=np.linspace(-1.0, 1.0, 5))
        theme.finalize(ax, title=title)
    return fig
