"""分位分析の図。

入力は :mod:`namportfolio.quantile` の出力（DataFrame / Series）。

    q  = npf.quantile.quantile_returns(df, factor="value", forward_return="ret_20d")
    ic = npf.quantile.information_coefficient(df, factor="value", forward_return="ret_20d")

    npf.viz.plot_quantile_returns(q)
    npf.viz.plot_ic(ic)

分位は**順序尺度**なので、識別用の 8 色ではなく単一色相の濃淡（ordinal）で塗る。
薄い＝低分位、濃い＝高分位が色から読める。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .. import performance as perf
from ..core.errors import ValidationError
from ..quantile import MISSING_LABEL
from . import theme

__all__ = [
    "plot_quantile_returns",
    "plot_quantile_cumulative",
    "plot_double_sort",
    "plot_ic",
    "plot_ic_heatmap",
    "plot_factor_decay",
    "plot_quantile_turnover",
    "plot_transition_matrix",
]

_PALETTES = ("ordinal", "categorical")


def plot_quantile_returns(
    returns: pd.DataFrame,
    *,
    annualize: bool = True,
    periods_per_year: float | None = None,
    palette: str = "ordinal",
    missing_label: str = MISSING_LABEL,
    title: str | None = None,
    ax=None,
) -> Figure:
    """分位（またはクラス）別の平均リターンを棒で並べる。

    単調に増えて（減って）いれば、シグナルが分位を跨いで効いている。

    Parameters
    ----------
    returns :
        :func:`namportfolio.quantile.quantile_returns` または
        :func:`namportfolio.quantile.class_returns` の出力。
    annualize :
        ``True`` なら年率リターン、``False`` なら期間平均。
    palette :
        ``"ordinal"`` は順序（分位・格付け）、``"categorical"`` は順序の無い
        クラス（業種など）。
    missing_label :
        欠損クラスの列名。この列だけ中立のグレーで描く。
    """
    _require_quantile_frame(returns)
    if annualize:
        values = perf.annualized_return(returns, periods_per_year=periods_per_year)
        label = "Annualized return"
    else:
        values = returns.mean()
        label = "Mean return"

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        colors = _series_colors(values.index, palette=palette, missing_label=missing_label)
        positions = np.arange(len(values))
        ax.bar(positions, values.to_numpy(), width=0.68, color=colors)
        ax.set_xticks(positions, list(values.index))
        ax.grid(axis="x", visible=False)
        theme.percent_axis(ax, decimals=1)
        theme.finalize(
            ax,
            title=title or f"{label} by quantile",
            ylabel=label,
            zero_line=True,
        )
    return fig


def plot_quantile_cumulative(
    returns: pd.DataFrame,
    *,
    long_short: bool = True,
    palette: str = "ordinal",
    missing_label: str = MISSING_LABEL,
    title: str | None = "Cumulative return by quantile",
    ax=None,
) -> Figure:
    """分位（またはクラス）別の累積リターン。

    Parameters
    ----------
    long_short :
        ``True`` なら「最上位 − 最下位」のスプレッドも重ねる。分位とは別の
        ものなので、順序色ではなく識別色（アクセント）で描く。**欠損クラスは
        順序を持たないので端の選択から除外する。**
    palette :
        ``"ordinal"`` は順序（分位・格付け）、``"categorical"`` は順序の無いクラス。
    """
    _require_quantile_frame(returns)
    with theme.styled():
        fig, ax = theme.new_axes(ax)
        colors = _series_colors(returns.columns, palette=palette, missing_label=missing_label)
        cumulative = perf.cumulative_returns(returns)

        for (name, series), color in zip(cumulative.items(), colors, strict=True):
            ax.plot(series.index, series.to_numpy(), color=color, label=str(name))

        ordered = _ordered_labels(returns.columns, missing_label)
        if long_short and len(ordered) >= 2:
            top, bottom = ordered[-1], ordered[0]
            spread = returns[top] - returns[bottom]
            accent = theme.categorical_colors(2)[1]
            ax.plot(
                spread.index,
                perf.cumulative_returns(spread).to_numpy(),
                color=accent,
                linewidth=2.2,
                label=f"{top}-{bottom}",
            )

        theme.percent_axis(ax)
        theme.finalize(ax, title=title, ylabel="Cumulative return", legend=True, zero_line=True)
    return fig


def plot_double_sort(
    matrix: pd.DataFrame,
    *,
    as_percent: bool = True,
    polarity: bool | None = None,
    annotate: bool = True,
    fmt: str | None = None,
    title: str | None = None,
    ax=None,
) -> Figure:
    """2 次元ソート（QQ 分析）のヒートマップ。

    縦軸が ``factor_1`` の分位、横軸が ``factor_2`` の分位。セルの値は
    :func:`namportfolio.quantile.double_sort_summary` で集計した統計量。

    Parameters
    ----------
    matrix :
        :func:`namportfolio.quantile.double_sort_summary` または
        :func:`namportfolio.quantile.double_sort_counts` の出力。
    as_percent :
        ``True``（既定）なら 100 倍して % として表示する。銘柄数を描くときは
        ``False`` にする。
    polarity :
        ``True`` で発散配色（正負が意味を持つリターン向け）、``False`` で単一色相
        （銘柄数のような非負の量向け）。``None`` なら負の値の有無で自動判定。

    Notes
    -----
    セルの銘柄数が少ないと値が暴れる。**リターンのヒートマップを読む前に
    :func:`namportfolio.quantile.double_sort_counts` で中身を確認すること。**
    """
    if matrix.empty:
        raise ValidationError("セルがありません。")

    values = matrix.to_numpy(dtype=float)
    if as_percent:
        values = values * 100.0
    if np.isnan(values).all():
        raise ValidationError("すべてのセルが欠損しています。")

    if polarity is None:
        polarity = bool(np.nanmin(values) < 0)
    if polarity:
        limit = float(np.nanmax(np.abs(values)))
        bounds = (-limit, limit)
        cmap = theme.diverging_cmap()
        ticks = np.linspace(-limit, limit, 5)
    else:
        bounds = (0.0, float(np.nanmax(values)))
        cmap = theme.sequential_cmap()
        ticks = np.linspace(*bounds, 5)

    with theme.styled():
        rows, columns = matrix.shape
        fig, ax = theme.new_axes(
            ax, figsize=(max(4.0, 0.9 * columns + 2.4), max(3.0, 0.7 * rows + 1.6))
        )
        mesh = theme.heatmap(
            ax,
            values,
            xticks=[str(c) for c in matrix.columns],
            yticks=[str(i) for i in matrix.index],
            cmap=cmap,
            vmin=bounds[0],
            vmax=bounds[1],
            annotate=annotate,
            fmt=fmt or ("{:.2f}" if as_percent else "{:.1f}"),
        )
        theme.add_colorbar(fig, mesh, ax, ticks=ticks)
        theme.finalize(
            ax,
            title=title or "Double sort",
            xlabel=str(matrix.columns.name or ""),
            ylabel=str(matrix.index.name or ""),
        )
    return fig


def plot_ic(
    ic: pd.Series,
    *,
    window: int | None = None,
    title: str | None = "Information coefficient",
    ax=None,
) -> Figure:
    """IC の推移と移動平均。

    IC は期間ごとの振れが大きく、生の系列だけでは傾向が読めない。移動平均を
    重ねて水準の変化を見る。

    Parameters
    ----------
    window :
        移動平均の窓。``None`` なら系列長の 1/10（最低 12、系列が短ければ省略）。
    """
    if isinstance(ic, pd.DataFrame):
        raise ValidationError("plot_ic は Series のみ対応です。列ごとに呼んでください。")

    values = ic.dropna()
    if window is None:
        window = max(12, len(values) // 10)

    with theme.styled():
        colors = theme.palette()
        base, accent = theme.categorical_colors(2)
        fig, ax = theme.new_axes(ax)

        ax.plot(values.index, values.to_numpy(), color=base, linewidth=0.9, alpha=0.55)
        if len(values) > window:
            rolled = values.rolling(window).mean()
            ax.plot(
                rolled.index,
                rolled.to_numpy(),
                color=accent,
                linewidth=2.0,
                label=f"{window}-period mean",
            )
            ax.legend(loc="best")

        mean = float(values.mean())
        ax.axhline(mean, color=colors["ink_secondary"], linewidth=1.0)
        ax.annotate(
            f" mean {mean:.3f}",
            xy=(values.index[-1], mean),
            xytext=(4, 0),
            textcoords="offset points",
            color=colors["ink_secondary"],
            fontsize=9,
            va="center",
        )
        ax.margins(x=0.02)
        theme.finalize(ax, title=title, ylabel="IC", zero_line=True)
    return fig


def plot_ic_heatmap(
    ic: pd.Series,
    *,
    annotate: bool | None = None,
    title: str | None = "Mean IC by month",
    ax=None,
) -> Figure:
    """年 × 月の平均 IC。

    リターンと違って IC は幾何的に積み上がるものではないので、月内の**平均**を取る。
    """
    if isinstance(ic, pd.DataFrame):
        raise ValidationError("plot_ic_heatmap は Series のみ対応です。")

    values = ic.dropna()
    if values.empty:
        raise ValidationError("IC が 1 つもありません。")

    table = (
        pd.DataFrame(
            {"year": values.index.year, "month": values.index.month, "ic": values.to_numpy()}
        )
        .groupby(["year", "month"])["ic"]
        .mean()
        .unstack("month")
    )
    matrix = table.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(matrix)))
    if annotate is None:
        annotate = len(table.index) <= 15

    with theme.styled():
        height = max(2.0, 0.32 * len(table.index) + 1.4)
        fig, ax = theme.new_axes(ax, figsize=(9.0, height))
        mesh = theme.heatmap(
            ax,
            matrix,
            xticks=[f"{m:02d}" for m in table.columns],
            yticks=[str(y) for y in table.index],
            cmap=theme.diverging_cmap(),
            vmin=-limit,
            vmax=limit,
            annotate=annotate,
            fmt="{:.2f}",
        )
        theme.add_colorbar(fig, mesh, ax, ticks=np.linspace(-limit, limit, 5))
        theme.finalize(ax, title=title)
    return fig


def plot_factor_decay(
    decay: pd.DataFrame,
    *,
    metric: str = "mean",
    title: str | None = "IC decay by horizon",
    ax=None,
) -> Figure:
    """保有期間を延ばしたときの IC の減衰。

    Parameters
    ----------
    decay :
        :func:`namportfolio.quantile.factor_decay` の出力。
    metric :
        描画する列（``"mean"`` / ``"icir"`` など）。
    """
    if metric not in decay.columns:
        raise ValidationError(f"decay に '{metric}' 列がありません: {list(decay.columns)}")

    values = decay[metric]
    with theme.styled():
        fig, ax = theme.new_axes(ax)
        color = theme.categorical_colors(1)[0]
        positions = np.arange(len(values))
        ax.plot(positions, values.to_numpy(), color=color, marker="o")
        ax.set_xticks(positions, [str(i) for i in values.index])
        ax.grid(axis="x", visible=False)
        theme.finalize(ax, title=title, xlabel="Horizon", ylabel=f"IC ({metric})", zero_line=True)
    return fig


def plot_quantile_turnover(
    turnover: pd.DataFrame,
    *,
    palette: str = "ordinal",
    missing_label: str = MISSING_LABEL,
    title: str | None = "Quantile turnover",
    ax=None,
) -> Figure:
    """分位ごとの入れ替わり率の推移。"""
    _require_quantile_frame(turnover)
    with theme.styled():
        fig, ax = theme.new_axes(ax)
        colors = _series_colors(turnover.columns, palette=palette, missing_label=missing_label)
        for (name, series), color in zip(turnover.items(), colors, strict=True):
            ax.plot(series.index, series.to_numpy(), color=color, label=str(name))
        theme.percent_axis(ax)
        theme.finalize(ax, title=title, ylabel="Turnover", legend=True)
    return fig


def plot_transition_matrix(
    matrix: pd.DataFrame,
    *,
    title: str | None = "Quantile transition probability",
    ax=None,
) -> Figure:
    """分位間の遷移確率。対角が濃いほどシグナルが安定している。

    確率（0〜1 の量）なので発散配色ではなく単一色相を使う。
    """
    values = matrix.to_numpy(dtype=float)
    with theme.styled():
        size = max(3.2, 0.5 * len(matrix.index) + 1.8)
        fig, ax = theme.new_axes(ax, figsize=(size + 1.2, size))
        mesh = theme.heatmap(
            ax,
            values,
            xticks=list(matrix.columns),
            yticks=list(matrix.index),
            cmap=theme.sequential_cmap(),
            vmin=0.0,
            vmax=float(np.nanmax(values)),
            annotate=True,
            fmt="{:.2f}",
        )
        theme.add_colorbar(fig, mesh, ax)
        theme.finalize(ax, title=title, xlabel="To", ylabel="From")
    return fig


def _require_quantile_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError(
            f"分位ごとの列を持つ DataFrame を渡してください: {type(frame).__name__}"
        )
    if frame.empty or frame.shape[1] == 0:
        raise ValidationError("分位の列がありません。")


def _series_colors(
    columns: Sequence,
    *,
    palette: str = "ordinal",
    missing_label: str = MISSING_LABEL,
) -> list[str]:
    """列ごとの色。**欠損クラスだけは中立のグレー**にする。

    欠損クラスは分位の順序の一部ではないので、濃淡の系列に混ぜると
    「一番大きい分位」に見えてしまう。

    Parameters
    ----------
    palette :
        ``"ordinal"``（順序＝分位や格付け）または ``"categorical"``（順序の無い
        クラス＝業種など）。
    """
    if palette not in _PALETTES:
        raise ValidationError(f"palette は {list(_PALETTES)} のいずれかです: {palette!r}")

    ordered = [column for column in columns if column != missing_label]
    picker = theme.ordinal_colors if palette == "ordinal" else theme.categorical_colors
    mapping = dict(zip(ordered, picker(len(ordered)), strict=True))
    mapping[missing_label] = theme.palette()["muted"]
    return [mapping[column] for column in columns]


def _ordered_labels(columns: Sequence, missing_label: str) -> list:
    """欠損クラスを除いた列。ロング・ショートの端を選ぶのに使う。"""
    return [column for column in columns if column != missing_label]
