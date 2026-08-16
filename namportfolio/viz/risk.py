"""Barra リスク分解と妥当性検証の図。

decomp = npf.risk.risk_decomposition(df, cov, factors=FACTORS)
npf.viz.plot_risk_decomposition(decomp)

contrib = npf.risk.factor_risk_contribution(df, cov, factors=FACTORS)
npf.viz.plot_risk_contribution(contrib)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..core.errors import ValidationError
from ..core.panel import require_columns
from . import theme

__all__ = [
    "plot_exposures",
    "plot_risk_decomposition",
    "plot_risk_contribution",
    "plot_factor_contribution",
    "plot_risk_forecast",
    "plot_bias_statistic",
]

#: 特異リスク・特異リターンの行につけられるラベル。
SPECIFIC_LABEL = "specific"


def plot_exposures(
    exposures: pd.DataFrame,
    *,
    factors: Sequence[str] | None = None,
    title: str | None = "Factor exposures",
    ax=None,
) -> Figure:
    """ファクターエクスポージャーの推移。

    Parameters
    ----------
    exposures :
        :func:`namportfolio.risk.factor_exposures` の出力。
    factors :
        描くファクター。``None`` なら全部。識別色は 8 本までなので、それを超える
        場合は絞るか :func:`plot_risk_contribution` でスナップショットを見る。
    """
    frame = exposures if factors is None else exposures[list(factors)]
    with theme.styled():
        fig, ax = theme.new_axes(ax)
        colors = theme.categorical_colors(frame.shape[1])
        for (name, series), color in zip(frame.items(), colors, strict=True):
            ax.plot(series.index, series.to_numpy(), color=color, label=str(name))
        theme.finalize(
            ax, title=title, ylabel="Exposure", legend=frame.shape[1] >= 2, zero_line=True
        )
    return fig


def plot_risk_decomposition(
    decomposition: pd.DataFrame,
    *,
    title: str | None = "Risk decomposition",
    ax=None,
) -> Figure:
    """トータル／ファクター／特異リスクの推移。

    **積み上げない。** リスクは加法的ではない（分散が加法的）ので、積み上げると
    合計が合わずに誤読させる。内訳を足し算で見たいときは
    :func:`plot_risk_contribution` を使う。
    """
    require_columns(
        decomposition,
        ["total_risk", "factor_risk", "specific_risk"],
        context="plot_risk_decomposition",
    )
    with theme.styled():
        colors = theme.palette()
        fig, ax = theme.new_axes(ax)
        series_colors = theme.categorical_colors(2)

        ax.plot(
            decomposition.index,
            decomposition["total_risk"].to_numpy(),
            color=colors["ink"],
            linewidth=2.2,
            label="total",
        )
        for name, color in zip(("factor_risk", "specific_risk"), series_colors, strict=True):
            ax.plot(
                decomposition.index,
                decomposition[name].to_numpy(),
                color=color,
                label=name.replace("_risk", ""),
            )

        theme.percent_axis(ax, decimals=1)
        ax.set_ylim(0, float(decomposition["total_risk"].max()) * 1.15)
        theme.finalize(ax, title=title, ylabel="Risk (annualised)", legend=True)
    return fig


def plot_risk_contribution(
    contributions: pd.DataFrame,
    *,
    at: object | None = None,
    top: int | None = None,
    title: str | None = None,
    ax=None,
) -> Figure:
    """ある時点のファクター別リスク寄与（CCTR）。

    ``cctr`` は加法的で、合計がトータルリスクに一致する。正負の極性を持つ
    （ヘッジになっているファクターは負）ので発散配色の両極で描く。

    Parameters
    ----------
    at :
        対象日。``None`` なら最終日。
    top :
        寄与の絶対値が大きい順に何本まで描くか。``None`` なら全部。
    """
    require_columns(contributions, ["cctr"], context="plot_risk_contribution")
    if not isinstance(contributions.index, pd.MultiIndex):
        raise ValidationError("MultiIndex (date, factor) の寄与を渡してください。")

    dates = contributions.index.get_level_values(0)
    target = dates.max() if at is None else pd.Timestamp(at)
    snapshot = contributions.xs(target, level=0)["cctr"]
    if top is not None and len(snapshot) > top:
        snapshot = snapshot.reindex(snapshot.abs().sort_values(ascending=False).index[:top])
    values = snapshot.sort_values()

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
        ax.axvline(0.0, color=theme.palette()["axis"], linewidth=0.8)
        theme.percent_axis(ax, which="x", decimals=1)
        theme.finalize(
            ax,
            title=title or f"Risk contribution ({target:%Y-%m-%d})",
            xlabel="Contribution to risk",
        )
    return fig


def plot_factor_contribution(
    attribution: pd.DataFrame,
    *,
    factors: Sequence[str] | None = None,
    title: str | None = "Cumulative factor contribution",
    ax=None,
) -> Figure:
    """ファクター別リターン寄与の累積。

    Parameters
    ----------
    attribution :
        :func:`namportfolio.risk.factor_return_attribution` の出力。
    factors :
        描くファクター。``None`` なら全部（特異リターンを含む）。
    """
    require_columns(attribution, ["contribution"], context="plot_factor_contribution")
    table = attribution["contribution"].unstack(level=-1)
    if factors is not None:
        table = table[list(factors)]
    cumulative = table.fillna(0.0).cumsum()

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        names = list(cumulative.columns)
        # 特異リターンはファクターの 1 つではないので識別色を割り当てない
        regular = [n for n in names if n != SPECIFIC_LABEL]
        mapping = dict(zip(regular, theme.categorical_colors(len(regular)), strict=True))
        mapping[SPECIFIC_LABEL] = theme.palette()["muted"]

        for name in names:
            ax.plot(
                cumulative.index,
                cumulative[name].to_numpy(),
                color=mapping[name],
                label=str(name),
            )
        ax.plot(
            cumulative.index,
            cumulative.sum(axis=1).to_numpy(),
            color=theme.palette()["ink"],
            linewidth=2.2,
            label="total",
        )

        theme.percent_axis(ax, decimals=1)
        theme.finalize(
            ax, title=title, ylabel="Cumulative contribution", legend=True, zero_line=True
        )
    return fig


def plot_risk_forecast(
    comparison: pd.DataFrame,
    *,
    title: str | None = "Predicted vs realised risk",
    ax=None,
) -> Figure:
    """予測リスクと実現ボラティリティの比較。

    2 本が離れ続けていればリスクモデルが偏っている。同じ尺度（年率リスク）なので
    1 枚に重ねてよい。
    """
    require_columns(comparison, ["predicted", "realized"], context="plot_risk_forecast")

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        predicted_color, realized_color = theme.categorical_colors(2)

        ax.plot(
            comparison.index,
            comparison["predicted"].to_numpy(),
            color=predicted_color,
            label="predicted",
        )
        ax.plot(
            comparison.index,
            comparison["realized"].to_numpy(),
            color=realized_color,
            label="realised",
        )
        theme.percent_axis(ax, decimals=1)
        # ローリング窓の先頭は NaN になるので nanmax で拾う
        upper = float(np.nanmax(comparison[["predicted", "realized"]].to_numpy()))
        if np.isfinite(upper):
            ax.set_ylim(0, upper * 1.15)
        theme.finalize(ax, title=title, ylabel="Risk (annualised)", legend=True)
    return fig


def plot_bias_statistic(
    bias: pd.Series,
    *,
    n_periods: int | None = None,
    confidence: float = 0.95,
    title: str | None = "Bias statistic",
    ax=None,
) -> Figure:
    """バイアス統計量の推移と、1 とみなせる範囲。

    帯の外に出ている期間は、リスク予測が偏っている（上に外れ＝過小評価）。

    Parameters
    ----------
    bias :
        :func:`namportfolio.risk.rolling_bias_statistic` の出力。
    n_periods :
        信頼区間の計算に使う標本数。``None`` ならローリング窓の長さを
        欠損の数から推定する。
    """
    if isinstance(bias, pd.DataFrame):
        raise ValidationError("plot_bias_statistic は Series のみ対応です。")

    from ..risk import bias_confidence_interval

    if n_periods is None:
        n_periods = int(bias.isna().sum()) + 1
    lower, upper = bias_confidence_interval(n_periods, confidence=confidence)

    with theme.styled():
        colors = theme.palette()
        fig, ax = theme.new_axes(ax)
        color = theme.categorical_colors(1)[0]

        if np.isfinite(lower):
            ax.fill_between(bias.index, lower, upper, color=colors["grid"], alpha=0.9, linewidth=0)
        ax.axhline(1.0, color=colors["axis"], linewidth=1.0)
        ax.plot(bias.index, bias.to_numpy(), color=color)

        theme.finalize(ax, title=title, ylabel="Bias statistic")
    return fig
