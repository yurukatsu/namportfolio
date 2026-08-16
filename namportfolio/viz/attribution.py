"""Brinson 帰属の図。

summary = npf.attribution.brinson_summary(df, segment="sector", asset_return="ret_1m")
npf.viz.plot_waterfall(summary)
npf.viz.plot_effects_by_segment(summary)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..core.errors import ValidationError
from . import theme

__all__ = [
    "plot_waterfall",
    "plot_effects_by_segment",
    "plot_cumulative_effects",
    "plot_effect_heatmap",
]

#: :func:`namportfolio.attribution.brinson_summary` が付ける合計行のラベル。
TOTAL_LABEL = "Total"


def plot_waterfall(
    summary: pd.DataFrame,
    *,
    total_label: str = TOTAL_LABEL,
    title: str | None = "Attribution breakdown",
    ax=None,
) -> Figure:
    """アクティブリターンの内訳をウォーターフォールで示す。

    各効果を左から積み上げ、最後に合計を置く。「配分で稼いだのか、銘柄選択で
    稼いだのか」を 1 枚で見る図。

    Parameters
    ----------
    summary :
        :func:`namportfolio.attribution.brinson_summary` の出力。合計行を使う。
    """
    if total_label not in summary.index:
        raise ValidationError(f"合計行 '{total_label}' がありません: {list(summary.index)}")
    values = summary.loc[total_label].drop(labels=["total"], errors="ignore")
    total = float(values.sum())

    with theme.styled():
        colors = theme.palette()
        fig, ax = theme.new_axes(ax)

        starts = np.concatenate([[0.0], np.cumsum(values.to_numpy())[:-1]])
        positions = np.arange(len(values) + 1)

        ax.bar(
            positions[:-1],
            values.to_numpy(),
            bottom=starts,
            width=0.62,
            color=theme.polarity_colors(values.to_numpy()),
        )
        # 合計は 0 から積み直す（差分ではなく水準なので中立色にする）
        ax.bar(positions[-1], total, width=0.62, color=colors["ink_secondary"])

        # 各段のつなぎ目を細い線で結び、積み上がりを追えるようにする。
        # 最後の段は合計バーまで延ばす（そこで水準が確定することを示す）
        for i, (start, value) in enumerate(zip(starts, values.to_numpy(), strict=True)):
            ax.plot(
                [i - 0.31, i + 0.69],
                [start + value, start + value],
                color=colors["axis"],
                linewidth=0.8,
                zorder=1,
            )

        labels = [str(name) for name in values.index] + [total_label]
        ax.set_xticks(positions, labels)
        ax.grid(axis="x", visible=False)
        theme.percent_axis(ax, decimals=2)
        theme.finalize(ax, title=title, ylabel="Active return", zero_line=True)
    return fig


def plot_effects_by_segment(
    summary: pd.DataFrame,
    *,
    total_label: str = TOTAL_LABEL,
    include_total_column: bool = False,
    title: str | None = "Attribution by segment",
    ax=None,
) -> Figure:
    """セグメントごとの効果を並べる。

    Parameters
    ----------
    include_total_column :
        ``True`` なら各セグメントの合計列も棒にする。既定では効果の内訳だけを
        描く（合計は内訳の和なので、並べると二重に見える）。
    """
    frame = summary.drop(index=total_label, errors="ignore")
    if not include_total_column:
        frame = frame.drop(columns=["total"], errors="ignore")
    if frame.empty:
        raise ValidationError("セグメントの行がありません。")

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        colors = theme.categorical_colors(frame.shape[1])
        positions = np.arange(len(frame.index))
        width = 0.72 / frame.shape[1]

        for i, ((name, series), color) in enumerate(zip(frame.items(), colors, strict=True)):
            offset = (i - (frame.shape[1] - 1) / 2) * width
            ax.bar(
                positions + offset,
                series.to_numpy(),
                width=width * 0.92,
                color=color,
                label=str(name),
            )

        ax.set_xticks(positions, [str(i) for i in frame.index])
        ax.grid(axis="x", visible=False)
        theme.percent_axis(ax, decimals=2)
        theme.finalize(ax, title=title, ylabel="Effect", legend=True, zero_line=True)
    return fig


def plot_cumulative_effects(
    effects: pd.DataFrame,
    *,
    title: str | None = "Cumulative attribution",
    ax=None,
) -> Figure:
    """効果の累積推移。

    Parameters
    ----------
    effects :
        :func:`namportfolio.attribution.brinson` または :func:`link` の出力
        （``MultiIndex (date, segment)``）。セグメントを合計してから累積する。

    Notes
    -----
    期間リンク済みの効果を渡すこと。リンクしていない効果を累積すると、末尾が
    幾何アクティブリターンと合わない。
    """
    if not isinstance(effects.index, pd.MultiIndex):
        raise ValidationError("MultiIndex (date, segment) の効果を渡してください。")

    cumulative = effects.groupby(level=0).sum().cumsum()
    with theme.styled():
        fig, ax = theme.new_axes(ax)
        colors = theme.categorical_colors(cumulative.shape[1])

        for (name, series), color in zip(cumulative.items(), colors, strict=True):
            ax.plot(series.index, series.to_numpy(), color=color, label=str(name))
        # 合計は内訳の 1 つではないので、識別色ではなく地の色で太く引く
        ax.plot(
            cumulative.index,
            cumulative.sum(axis=1).to_numpy(),
            color=theme.palette()["ink"],
            linewidth=2.2,
            label="total",
        )

        theme.percent_axis(ax, decimals=1)
        theme.finalize(ax, title=title, ylabel="Cumulative effect", legend=True, zero_line=True)
    return fig


def plot_effect_heatmap(
    effects: pd.DataFrame,
    *,
    effect: str = "allocation",
    annotate: bool | None = None,
    title: str | None = None,
    ax=None,
) -> Figure:
    """セグメント × 期間の効果ヒートマップ。

    どのセグメントがいつ効いたかを見る。正負の極性を持つので発散配色。
    """
    if effect not in effects.columns:
        raise ValidationError(f"'{effect}' 列がありません: {list(effects.columns)}")

    table = effects[effect].unstack(level=-1)
    matrix = table.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(matrix)))
    if annotate is None:
        annotate = matrix.size <= 120

    with theme.styled():
        height = max(2.4, 0.3 * len(table.index) + 1.4)
        fig, ax = theme.new_axes(ax, figsize=(9.0, height))
        mesh = theme.heatmap(
            ax,
            matrix * 100.0,
            xticks=[str(c) for c in table.columns],
            yticks=[f"{d:%Y-%m}" if isinstance(d, pd.Timestamp) else str(d) for d in table.index],
            cmap=theme.diverging_cmap(),
            vmin=-limit * 100.0,
            vmax=limit * 100.0,
            annotate=annotate,
            fmt="{:.2f}",
        )
        theme.add_colorbar(fig, mesh, ax, ticks=np.linspace(-limit * 100.0, limit * 100.0, 5))
        theme.finalize(ax, title=title or f"{effect.capitalize()} effect (%)")
    return fig
