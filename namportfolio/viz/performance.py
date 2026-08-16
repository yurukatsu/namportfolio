"""リターン評価の図。

すべての関数は **Figure を返し、``show()`` は呼ばない**。Jupyter でもレポート
生成でも同じように使える。``ax`` を渡せば既存の Axes に描くので、複数の図を
1 枚に組み合わせられる。

    fig = npf.viz.plot_cumulative_returns(returns, benchmark)
    fig.savefig("cumulative.png")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .. import performance as perf
from ..core.errors import ValidationError
from . import theme

__all__ = [
    "plot_cumulative_returns",
    "plot_drawdown",
    "plot_monthly_heatmap",
    "plot_annual_returns",
    "plot_rolling",
    "plot_return_distribution",
]

# 直接ラベルを付ける上限。これを超えると重なって読めなくなるので凡例だけにする。
_MAX_DIRECT_LABELS = 4


def plot_cumulative_returns(
    returns: perf.ReturnsLike,
    benchmark: perf.ReturnsLike | None = None,
    *,
    log: bool = False,
    title: str | None = "Cumulative return",
    ax=None,
) -> Figure:
    """累積リターンの推移。

    系列が 4 本以下なら右端に系列名を直接置く（凡例を目で往復しなくて済む）。

    Parameters
    ----------
    log :
        ``True`` で縦軸を対数にする。長期で倍率の比較をしたいときに使う。
    """
    frame = _to_frame(returns, benchmark)
    with theme.styled():
        fig, ax = theme.new_axes(ax)
        colors = theme.categorical_colors(frame.shape[1])
        cumulative = perf.cumulative_returns(frame)

        for (name, series), color in zip(cumulative.items(), colors, strict=True):
            ax.plot(series.index, series.to_numpy(), color=color, label=str(name))

        if log:
            ax.set_yscale("symlog", linthresh=0.1)
        theme.percent_axis(ax)
        n_series = frame.shape[1]
        if n_series <= _MAX_DIRECT_LABELS:
            _label_line_ends(ax, cumulative, colors)
        theme.finalize(
            ax,
            title=title,
            ylabel="Cumulative return",
            legend=n_series >= 2,
            zero_line=True,
        )
    return fig


def plot_drawdown(
    returns: perf.ReturnsLike,
    benchmark: perf.ReturnsLike | None = None,
    *,
    title: str | None = "Drawdown",
    ax=None,
) -> Figure:
    """アンダーウォーター図（高値からの下落率の推移）。

    先頭の系列だけ塗りつぶし、以降は線で重ねる。
    """
    frame = _to_frame(returns, benchmark)
    with theme.styled():
        fig, ax = theme.new_axes(ax)
        colors = theme.categorical_colors(frame.shape[1])
        underwater = perf.drawdown(frame)

        # 塗りつぶすのは単一系列のときだけ。複数を塗ると下の線が埋もれる
        fill = frame.shape[1] == 1
        for (name, series), color in zip(underwater.items(), colors, strict=True):
            ax.plot(series.index, series.to_numpy(), color=color, label=str(name))
            if fill:
                ax.fill_between(series.index, series.to_numpy(), 0.0, color=color, alpha=0.18)

        theme.percent_axis(ax)
        theme.finalize(
            ax,
            title=title,
            ylabel="Drawdown",
            legend=frame.shape[1] >= 2,
            zero_line=True,
        )
    return fig


def plot_monthly_heatmap(
    returns: pd.Series,
    *,
    annotate: bool | None = None,
    title: str | None = "Monthly return (%)",
    ax=None,
) -> Figure:
    """年 × 月のリターンヒートマップ。

    正負の極性を表すので発散配色（中点はグレー＝ゼロ）。カラースケールは
    ゼロを中心に対称にする（正負で濃さの意味が変わらないように）。

    Parameters
    ----------
    annotate :
        セルに数値を入れるか。``None`` なら 15 年以下のとき自動で入れる。
    """
    if isinstance(returns, pd.DataFrame):
        raise ValidationError("plot_monthly_heatmap は Series のみ対応です。")

    table = perf.monthly_table(returns).drop(columns="year_total")
    matrix = table.to_numpy(dtype=float)
    if np.isnan(matrix).all():
        raise ValidationError("月次リターンが 1 つもありません。")

    limit = float(np.nanmax(np.abs(matrix)))
    if annotate is None:
        annotate = len(table.index) <= 15

    with theme.styled():
        height = max(2.0, 0.32 * len(table.index) + 1.4)
        fig, ax = theme.new_axes(ax, figsize=(9.0, height))

        mesh = theme.heatmap(
            ax,
            matrix * 100.0,
            xticks=list(table.columns),
            yticks=[str(y) for y in table.index],
            cmap=theme.diverging_cmap(),
            vmin=-limit * 100.0,
            vmax=limit * 100.0,
            annotate=annotate,
        )
        # 目盛は等間隔に置く（自動選択だと半端な値が並ぶ）
        theme.add_colorbar(fig, mesh, ax, ticks=np.linspace(-limit * 100.0, limit * 100.0, 5))
        theme.finalize(ax, title=title)
    return fig


def plot_annual_returns(
    returns: perf.ReturnsLike,
    benchmark: perf.ReturnsLike | None = None,
    *,
    title: str | None = "Annual return",
    ax=None,
) -> Figure:
    """暦年ごとのリターンを棒で並べる。"""
    frame = _to_frame(returns, benchmark)
    annual = perf.aggregate_returns(frame, perf.YEAR_END)
    years = [str(d.year) for d in annual.index]

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        colors = theme.categorical_colors(frame.shape[1])
        n_series = frame.shape[1]
        positions = np.arange(len(years))
        # 隣り合う棒の間に背景色の隙間を残す（枠線で区切らない）
        width = 0.72 / n_series

        for i, ((name, series), color) in enumerate(zip(annual.items(), colors, strict=True)):
            offset = (i - (n_series - 1) / 2) * width
            ax.bar(
                positions + offset,
                series.to_numpy(),
                width=width * 0.92,
                color=color,
                label=str(name),
            )

        ax.set_xticks(positions, years)
        ax.grid(axis="x", visible=False)
        theme.percent_axis(ax)
        theme.finalize(ax, title=title, ylabel="Return", legend=n_series >= 2, zero_line=True)
    return fig


def plot_rolling(
    returns: perf.ReturnsLike,
    benchmark: perf.ReturnsLike | None = None,
    *,
    window: int = 60,
    metric: str = "sharpe",
    periods_per_year: float | None = None,
    title: str | None = None,
    ax=None,
) -> Figure:
    """ローリング指標の推移。

    2 つの指標を 1 枚に重ねない（縦軸が 2 本になると読み手が勝手な相関を
    見てしまう）。指標ごとに図を分けること。

    Parameters
    ----------
    metric :
        ``"volatility"`` / ``"sharpe"`` / ``"beta"`` / ``"tracking_error"`` /
        ``"information_ratio"``。後ろ 3 つは ``benchmark`` が必要。
    window :
        窓の期間数（データ点数）。
    """
    if metric not in _ROLLING:
        raise ValidationError(f"metric は {sorted(_ROLLING)} のいずれかです: {metric!r}")
    needs_benchmark = metric in ("beta", "tracking_error", "information_ratio")
    if needs_benchmark and benchmark is None:
        raise ValidationError(f"metric={metric!r} には benchmark が必要です。")

    frame = _to_frame(returns, benchmark if not needs_benchmark else None)
    rolled = _ROLLING[metric](frame, benchmark, window, periods_per_year)
    is_percent = metric in ("volatility", "tracking_error")

    with theme.styled():
        fig, ax = theme.new_axes(ax)
        colors = theme.categorical_colors(rolled.shape[1])
        for (name, series), color in zip(rolled.items(), colors, strict=True):
            ax.plot(series.index, series.to_numpy(), color=color, label=str(name))

        if is_percent:
            theme.percent_axis(ax)
        theme.finalize(
            ax,
            title=title or f"Rolling {metric.replace('_', ' ')} ({window} periods)",
            ylabel=metric.replace("_", " "),
            legend=rolled.shape[1] >= 2,
            zero_line=not is_percent,
        )
    return fig


def plot_return_distribution(
    returns: pd.Series,
    *,
    bins: int = 50,
    var_level: float = 0.05,
    title: str | None = "Return distribution",
    ax=None,
) -> Figure:
    """リターンの分布。平均と VaR の位置を縦線で示す。"""
    if isinstance(returns, pd.DataFrame):
        raise ValidationError("plot_return_distribution は Series のみ対応です。")

    values = returns.dropna()
    with theme.styled():
        colors = theme.palette()
        series_color, accent = theme.categorical_colors(2)
        fig, ax = theme.new_axes(ax)

        ax.hist(
            values.to_numpy(),
            bins=bins,
            color=series_color,
            edgecolor=colors["surface"],
            linewidth=0.5,
        )
        var = perf.value_at_risk(values, level=var_level)
        ax.axvline(float(values.mean()), color=colors["ink_secondary"], linewidth=1.2)
        ax.axvline(float(var), color=accent, linewidth=1.2)

        # 縦線は 2 本だけなので、凡例ではなく線の脇に直接書く
        top = ax.get_ylim()[1]
        ax.text(
            float(values.mean()),
            top * 0.98,
            " mean",
            color=colors["ink_secondary"],
            fontsize=9,
            va="top",
        )
        ax.text(
            float(var),
            top * 0.98,
            f" VaR {var_level:.0%}",
            color=accent,
            fontsize=9,
            va="top",
            ha="right",
        )

        ax.grid(axis="x", visible=False)
        theme.percent_axis(ax, which="x", decimals=1)
        theme.finalize(ax, title=title, xlabel="Return", ylabel="Periods")
    return fig


# --------------------------------------------------------------------------
# 内部ヘルパー
# --------------------------------------------------------------------------


_ROLLING = {
    "volatility": lambda f, b, w, p: perf.rolling_volatility(f, w, periods_per_year=p),
    "sharpe": lambda f, b, w, p: perf.rolling_sharpe(f, w, periods_per_year=p),
    "beta": lambda f, b, w, p: perf.rolling_beta(f, b, w),
    "tracking_error": lambda f, b, w, p: perf.rolling_tracking_error(f, b, w, periods_per_year=p),
    "information_ratio": lambda f, b, w, p: perf.rolling_information_ratio(
        f, b, w, periods_per_year=p
    ),
}


def _to_frame(
    returns: perf.ReturnsLike,
    benchmark: perf.ReturnsLike | None = None,
) -> pd.DataFrame:
    """描画対象を DataFrame（列＝系列）に揃える。

    列の順序＝色の割り当て順なので、ベンチマークは常に最後に置く。
    """
    if isinstance(returns, pd.Series):
        frame = returns.to_frame(returns.name if returns.name is not None else "portfolio")
    elif isinstance(returns, pd.DataFrame):
        frame = returns.copy()
    else:
        raise ValidationError(
            f"returns は Series か DataFrame を渡してください: {type(returns).__name__}"
        )

    if benchmark is not None:
        if isinstance(benchmark, pd.DataFrame):
            if benchmark.shape[1] != 1:
                raise ValidationError("benchmark の DataFrame は 1 列にしてください。")
            benchmark = benchmark.iloc[:, 0]
        name = benchmark.name if benchmark.name is not None else "benchmark"
        frame = frame.join(benchmark.rename(name), how="outer")
    return frame


def _label_line_ends(ax, frame: pd.DataFrame, colors: list[str]) -> None:
    """線の右端に系列名を置く。凡例を目で往復しなくて済むようにする。"""
    for (name, series), color in zip(frame.items(), colors, strict=True):
        last = series.last_valid_index()
        if last is None:
            continue
        ax.annotate(
            f" {name}",
            xy=(last, series[last]),
            xytext=(4, 0),
            textcoords="offset points",
            color=color,
            fontsize=9,
            va="center",
        )
    ax.margins(x=0.02)
