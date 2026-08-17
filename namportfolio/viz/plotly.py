"""plotly 用のスタイルヘルパー。

**図そのものは作らない。** ノートブックで拡大やホバーを効かせたいときに、
plotly express などで自由に描いてもらい、見た目だけ matplotlib 版と揃える。

    import plotly.express as px

    cum = npf.performance.cumulative_returns(pd.DataFrame({"strategy": r, "benchmark": b}))
    fig = px.line(cum)
    npf.viz.plotly.apply_theme(fig, title="Cumulative return", percent_axis="y", zero_line=True)

計算結果はすべて素の DataFrame なので、``px.line`` / ``px.bar`` / ``px.imshow`` に
そのまま渡せる。

.. rubric:: 使い分け

- **matplotlib 版**（``npf.viz.plot_*``）: レポート・保存・組み合わせ図
- **plotly + このヘルパー**: ノートブックでの探索（拡大、ホバー、凡例クリック）

.. rubric:: 注意

plotly express は trace に色を埋め込むため、``layout.colorway`` を設定しても
上書きされない。:func:`apply_theme` は trace 側を直接塗り替えることで揃えている
（``recolor=False`` で無効化できる）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.errors import ValidationError
from . import theme

__all__ = [
    "apply_theme",
    "color_sequence",
    "sequential_scale",
    "diverging_scale",
]

_PERCENT_AXES = {None, "x", "y", "both"}


def apply_theme(
    fig: Any,
    *,
    mode: theme.Mode | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    percent_axis: str | None = None,
    decimals: int = 0,
    zero_line: bool = False,
    ordinal: bool = False,
    recolor: bool = True,
) -> Any:
    """plotly の Figure に matplotlib 版と同じ見た目を当てる。

    Parameters
    ----------
    fig :
        plotly の Figure（``plotly.express`` の戻り値など）。
    percent_axis :
        ``"x"`` / ``"y"`` / ``"both"`` を指定するとその軸を % 表示にする。
    decimals :
        % 表示の小数桁数。
    zero_line :
        ``True`` で 0 の水平線を引く。
    ordinal :
        ``True`` なら順序用の配色（分位など）を使う。系列数に応じて濃淡を割り当てる。
    recolor :
        ``True``（既定）なら trace の色を塗り替える。自分で色を指定した場合は
        ``False`` にする。

    Returns
    -------
    plotly.graph_objects.Figure
        渡された Figure をその場で変更して返す。
    """
    if percent_axis not in _PERCENT_AXES:
        raise ValidationError(f"percent_axis は 'x' / 'y' / 'both' か None です: {percent_axis!r}")

    colors = theme.palette(mode)
    if recolor:
        _recolor_traces(fig, ordinal=ordinal, mode=mode)

    fig.update_layout(
        template="none",
        colorway=color_sequence(max(len(fig.data), 1), ordinal=ordinal, mode=mode),
        paper_bgcolor=colors["surface"],
        plot_bgcolor=colors["surface"],
        font={"family": "system-ui, -apple-system, sans-serif", "size": 12, "color": colors["ink"]},
        legend={"bgcolor": "rgba(0,0,0,0)", "borderwidth": 0, "title_text": ""},
        margin={"l": 70, "r": 40, "t": 60 if title else 30, "b": 50},
        hovermode="x unified",
    )
    if title:
        fig.update_layout(title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 15}})

    # 縦のグリッドは引かない（matplotlib 版と同じ）
    fig.update_xaxes(
        title_text=xlabel,
        showgrid=False,
        zeroline=False,
        linecolor=colors["axis"],
        tickcolor=colors["muted"],
        tickfont={"color": colors["muted"], "size": 11},
    )
    fig.update_yaxes(
        title_text=ylabel,
        gridcolor=colors["grid"],
        griddash="solid",
        zeroline=False,
        linecolor=colors["axis"],
        tickcolor=colors["muted"],
        tickfont={"color": colors["muted"], "size": 11},
    )

    if percent_axis in ("y", "both"):
        fig.update_yaxes(tickformat=f".{decimals}%")
    if percent_axis in ("x", "both"):
        fig.update_xaxes(tickformat=f".{decimals}%")

    fig.update_traces(line_width=1.8, selector={"type": "scatter"})
    if zero_line:
        fig.add_hline(y=0, line_color=colors["axis"], line_width=1)
    return fig


def color_sequence(
    n: int,
    *,
    ordinal: bool = False,
    mode: theme.Mode | None = None,
) -> list[str]:
    """plotly に渡す色の並び。

    ``px.line(..., color_discrete_sequence=npf.viz.plotly.color_sequence(5))`` の
    ように、図を作る時点で色を指定したい場合に使う。
    """
    if ordinal:
        return theme.ordinal_colors(max(n, 1), mode)
    return theme.categorical_colors(min(max(n, 1), 8), mode)


def sequential_scale(mode: theme.Mode | None = None) -> list[list]:
    """量を表す colorscale（単一色相）。``px.imshow(..., color_continuous_scale=...)``。"""
    steps = theme.palette(mode)["sequential"]
    return _to_scale(steps)


def diverging_scale(mode: theme.Mode | None = None) -> list[list]:
    """極性を表す colorscale。中点はグレー。

    ``px.imshow(..., color_continuous_scale=..., color_continuous_midpoint=0)``
    のように**中点を明示すること**。しないとデータの範囲で勝手に中心が決まり、
    「ゼロがグレー」という前提が崩れる。
    """
    colors = theme.palette(mode)
    positive = theme.POLES[theme.POSITIVE_HUE]
    negative = theme.POLES["red" if theme.POSITIVE_HUE == "blue" else "blue"]
    return _to_scale([negative, colors["neutral"], positive])


def _to_scale(steps: Sequence[str]) -> list[list]:
    last = len(steps) - 1
    return [[index / last, color] for index, color in enumerate(steps)]


def _recolor_traces(fig: Any, *, ordinal: bool, mode: theme.Mode | None) -> None:
    """trace の色を塗り替える。

    plotly express は色を trace に埋め込むので、``layout.colorway`` だけでは
    見た目が変わらない。
    """
    palette = color_sequence(len(fig.data), ordinal=ordinal, mode=mode)
    if not palette:
        return
    for index, trace in enumerate(fig.data):
        color = palette[index % len(palette)]
        if trace.type == "scatter":
            trace.line.color = color
        elif trace.type == "bar":
            trace.marker.color = color
