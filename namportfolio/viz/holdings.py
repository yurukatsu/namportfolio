"""保有ベース分析の図。

入力は :mod:`namportfolio.holdings` の出力。

    alloc = npf.holdings.allocation(df, by="sector")
    npf.viz.plot_allocation(alloc)

    top = npf.holdings.top_contributors(df, forward_return="ret_1m")
    npf.viz.plot_contribution(top)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..core.errors import ValidationError
from ..core.panel import require_columns
from . import theme

__all__ = [
    "plot_allocation",
    "plot_concentration",
    "plot_contribution",
    "plot_characteristics",
    "plot_turnover",
]


def plot_allocation(
    allocation: pd.DataFrame,
    *,
    stacked: bool | None = None,
    title: str | None = "Allocation",
    ax=None,
) -> Figure:
    """セグメント別配分の推移。

    Parameters
    ----------
    allocation :
        :func:`namportfolio.holdings.allocation` の出力。
    stacked :
        積み上げ面にするか。``None`` なら**負の値があれば自動で線に切り替える**
        （アクティブ配分は正負が混ざるため。積み上げ面は負を正しく表現できない）。
    """
    if allocation.shape[1] == 0:
        raise ValidationError("セグメントの列がありません。")
    has_negative = bool((allocation.to_numpy() < 0).any())
    if stacked is None:
        stacked = not has_negative
    elif stacked and has_negative:
        raise ValidationError(
            "負の配分があるので積み上げ面にできません（stacked=False を指定してください）。"
        )

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        colors = theme.categorical_colors(allocation.shape[1])
        filled = allocation.fillna(0.0)

        if stacked:
            ax.stackplot(
                filled.index,
                filled.to_numpy().T,
                labels=[str(c) for c in filled.columns],
                colors=colors,
                edgecolor=theme.palette()["surface"],
                linewidth=0.8,
            )
            ax.set_ylim(0, float(filled.sum(axis=1).max()) * 1.02)
            # 面が全域を覆うので、凡例は軸の外に出さないと必ず重なる
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
        else:
            for (name, series), color in zip(filled.items(), colors, strict=True):
                ax.plot(series.index, series.to_numpy(), color=color, label=str(name))
            ax.legend(loc="best")

        theme.percent_axis(ax)
        theme.finalize(ax, title=title, ylabel="Weight", zero_line=has_negative)
    return fig


def plot_concentration(
    concentration: pd.DataFrame,
    *,
    metric: str = "effective_n",
    title: str | None = None,
    ax=None,
) -> Figure:
    """集中度の推移。

    Parameters
    ----------
    metric :
        描く列。``"effective_n"``（実質銘柄数）/ ``"n_holdings"`` / ``"hhi"`` /
        ``"top10_share"`` など。件数と比率は尺度が違うので 1 枚には重ねない。
    """
    require_columns(concentration, [metric], context="plot_concentration")
    series = concentration[metric]

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        color = theme.categorical_colors(1)[0]
        ax.plot(series.index, series.to_numpy(), color=color)
        if metric.endswith("_share"):
            theme.percent_axis(ax)
        ax.set_ylim(0, float(series.max()) * 1.1)
        theme.finalize(
            ax,
            title=title or f"Concentration ({metric.replace('_', ' ')})",
            ylabel=metric.replace("_", " "),
        )
    return fig


def plot_contribution(
    contributors: pd.DataFrame,
    *,
    column: str = "contribution",
    title: str | None = "Return contribution",
    ax=None,
) -> Figure:
    """寄与度の大きい／小さい銘柄を横棒で並べる。

    正負は「どちら側か」が意味を持つので発散配色の両極を使う（識別色ではない）。
    銘柄名は長いことが多いので横棒にする。

    Parameters
    ----------
    contributors :
        :func:`namportfolio.holdings.top_contributors` の出力。
    """
    require_columns(contributors, [column], context="plot_contribution")
    values = contributors[column].sort_values()

    with theme.styled():
        fig, ax = theme.new_axes(ax, figsize=(8.0, max(2.4, 0.32 * len(values) + 1.2)))
        ax.barh(
            np.arange(len(values)),
            values.to_numpy(),
            height=0.72,
            color=theme.polarity_colors(values.to_numpy()),
        )
        ax.set_yticks(np.arange(len(values)), [str(i) for i in values.index])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        theme.percent_axis(ax, which="x", decimals=1)
        ax.axvline(0.0, color=theme.palette()["axis"], linewidth=0.8)
        theme.finalize(ax, title=title, xlabel="Contribution")
    return fig


def plot_characteristics(
    characteristics: pd.DataFrame,
    *,
    metric: str,
    title: str | None = None,
    ax=None,
) -> Figure:
    """ポートフォリオ特性の推移。

    特性ごとに尺度が違う（PER と配当利回りなど）ので、1 枚に 1 指標だけ描く。
    """
    require_columns(characteristics, [metric], context="plot_characteristics")
    series = characteristics[metric]

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        color = theme.categorical_colors(1)[0]
        ax.plot(series.index, series.to_numpy(), color=color)
        theme.finalize(
            ax,
            title=title or f"Portfolio {metric}",
            ylabel=metric,
            zero_line=bool((series.dropna() < 0).any()),
        )
    return fig


def plot_turnover(
    turnover: pd.Series,
    *,
    title: str | None = "Turnover",
    ax=None,
) -> Figure:
    """ターンオーバーの推移と平均。"""
    if isinstance(turnover, pd.DataFrame):
        raise ValidationError("plot_turnover は Series のみ対応です。")

    values = turnover.dropna()
    with theme.styled():
        colors = theme.palette()
        fig, ax = theme.new_axes(ax)
        color = theme.categorical_colors(1)[0]

        ax.plot(values.index, values.to_numpy(), color=color)
        mean = float(values.mean())
        ax.axhline(mean, color=colors["ink_secondary"], linewidth=1.0)
        ax.annotate(
            f" mean {mean:.1%}",
            xy=(values.index[-1], mean),
            xytext=(4, 0),
            textcoords="offset points",
            color=colors["ink_secondary"],
            fontsize=9,
            va="center",
        )
        ax.margins(x=0.02)
        ax.set_ylim(0, float(values.max()) * 1.15)
        theme.percent_axis(ax)
        theme.finalize(ax, title=title, ylabel="Turnover")
    return fig
