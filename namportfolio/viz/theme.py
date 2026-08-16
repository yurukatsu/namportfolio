"""図の配色とスタイル。

配色は用途ごとに役割が決まっており、手で選ばない。

============ ================================================================
役割          使いどころ
============ ================================================================
categorical  識別（複数戦略、ポート vs ベンチ）。固定順に割り当て、循環させない
ordinal      順序（分位 Q1..Q5、階層）。単一色相の濃淡で順序が色に出る
sequential   量（連続値のヒートマップ）。単一色相 light→dark
diverging    極性（正／負のリターン、アクティブウェイト）。2 色相＋グレー中点
============ ================================================================

**分位を categorical で塗らないこと。** Q1〜Q5 には順序があるので ordinal を使う。

配色は色覚特性（P型・D型）を考慮して検証済み。categorical は隣接ペアの
CVD ΔE 9.1 / 通常視 19.6（OKLab ×100）、ordinal は単調な明度差を満たす。
色を差し替える場合は同じ検証を通すこと。

.. note::
   categorical の 3・4・5 番目（aqua / yellow / magenta）は背景とのコントラストが
   3:1 を下回る。5 系列以上を塗る場合は凡例だけでなく直接ラベルか数表を併記する。
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import FuncFormatter

__all__ = [
    "Mode",
    "set_mode",
    "get_mode",
    "palette",
    "categorical_colors",
    "ordinal_colors",
    "sequential_cmap",
    "diverging_cmap",
    "rc_params",
    "apply_style",
    "styled",
    "new_axes",
    "finalize",
    "heatmap",
    "add_colorbar",
    "percent_axis",
    "use_japanese_font",
]

Mode = Literal["light", "dark"]

# 識別用。固定順に先頭から使う。9 系列目は作らず「その他」にまとめる。
_CATEGORICAL = {
    "light": [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#4a3aa7",
        "#e34948",
    ],
    "dark": [
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ],
}

# 順序用の単一色相ランプ。light 端は背景から十分離れた段階から始める。
_ORDINAL = {
    "light": [
        "#86b6ef",
        "#6da7ec",
        "#5598e7",
        "#3987e5",
        "#2a78d6",
        "#256abf",
        "#1c5cab",
        "#184f95",
        "#104281",
        "#0d366b",
    ],
    "dark": [
        "#cde2fb",
        "#b7d3f6",
        "#9ec5f4",
        "#86b6ef",
        "#6da7ec",
        "#5598e7",
        "#3987e5",
        "#2a78d6",
        "#256abf",
        "#184f95",
    ],
}

# 量用。0 に近い側が背景に溶ける（連続量なので許容）。
_SEQUENTIAL = {
    "light": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    "dark": ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
}

_CHROME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "neutral": "#f0efec",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "neutral": "#383835",
    },
}

#: 極性の正側に置く色。既定は青。日本の株式慣行に合わせるなら
#: ``theme.POSITIVE_HUE = "red"`` で反転できる。
POSITIVE_HUE: Literal["blue", "red"] = "blue"

_POLES = {"blue": "#2a78d6", "red": "#e34948"}

_mode: Mode = "light"


def set_mode(mode: Mode) -> None:
    """既定の配色モードを切り替える（``"light"`` / ``"dark"``）。"""
    if mode not in ("light", "dark"):
        raise ValueError(f"mode は 'light' か 'dark' です: {mode!r}")
    global _mode
    _mode = mode


def get_mode() -> Mode:
    """現在の配色モード。"""
    return _mode


def palette(mode: Mode | None = None) -> dict[str, Any]:
    """指定モードの色一式を返す。"""
    mode = mode or _mode
    return {
        "categorical": list(_CATEGORICAL[mode]),
        "ordinal": list(_ORDINAL[mode]),
        "sequential": list(_SEQUENTIAL[mode]),
        **_CHROME[mode],
    }


def categorical_colors(n: int, mode: Mode | None = None) -> list[str]:
    """識別用の色を固定順に ``n`` 個返す。

    8 を超える要求はエラーにする。9 個目の色を作ると色覚特性下で既存色と
    区別できなくなるため、「その他」にまとめるか図を分けること。
    """
    colors = _CATEGORICAL[mode or _mode]
    if n > len(colors):
        raise ValueError(
            f"識別用の色は最大 {len(colors)} 系列です（要求: {n}）。"
            " 上位のみ表示して残りを『その他』にまとめるか、図を分けてください。"
        )
    return colors[:n]


def ordinal_colors(n: int, mode: Mode | None = None) -> list[str]:
    """順序用の色を薄い→濃いの順に ``n`` 個返す（分位など）。"""
    steps = _ORDINAL[mode or _mode]
    if n <= len(steps):
        idx = np.linspace(0, len(steps) - 1, n).round().astype(int)
        return [steps[i] for i in idx]
    cmap = LinearSegmentedColormap.from_list("npf_ordinal", steps)
    return [to_hex(cmap(x)) for x in np.linspace(0.0, 1.0, n)]


def sequential_cmap(mode: Mode | None = None) -> LinearSegmentedColormap:
    """量を表す連続カラーマップ（単一色相）。"""
    return LinearSegmentedColormap.from_list("npf_sequential", _SEQUENTIAL[mode or _mode])


def diverging_cmap(mode: Mode | None = None) -> LinearSegmentedColormap:
    """極性を表す連続カラーマップ。中点はグレーで「ゼロ」に見えるようにする。

    正側の色相は :data:`POSITIVE_HUE` で切り替える。
    """
    mode = mode or _mode
    positive = _POLES[POSITIVE_HUE]
    negative = _POLES["red" if POSITIVE_HUE == "blue" else "blue"]
    return LinearSegmentedColormap.from_list(
        "npf_diverging", [negative, _CHROME[mode]["neutral"], positive]
    )


def rc_params(mode: Mode | None = None) -> dict[str, Any]:
    """matplotlib の rcParams 辞書を返す。

    細いマーク・ヘアラインのグリッド・控えめな軸。グリッドは実線（破線は
    「閾値」に見えてしまう）。上と右の枠線は消す。
    """
    mode = mode or _mode
    c = _CHROME[mode]
    return {
        "figure.facecolor": c["surface"],
        "figure.figsize": (9.0, 4.5),
        "figure.dpi": 110,
        "savefig.facecolor": c["surface"],
        "savefig.bbox": "tight",
        "axes.facecolor": c["surface"],
        "axes.edgecolor": c["axis"],
        "axes.labelcolor": c["ink_secondary"],
        "axes.titlecolor": c["ink"],
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.prop_cycle": plt.cycler(color=_CATEGORICAL[mode]),
        "grid.color": c["grid"],
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.color": c["muted"],
        "ytick.color": c["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": c["ink"],
        "font.size": 10,
    }


def apply_style(mode: Mode | None = None) -> None:
    """rcParams をグローバルに適用する。自分で ``plt.plot`` を書く場合に使う。

    本パッケージの描画関数は :func:`styled` で内部的にスタイルを当てるため、
    これを呼ばなくても見た目は揃う。
    """
    plt.rcParams.update(rc_params(mode))


@contextmanager
def styled(mode: Mode | None = None):
    """スタイルを一時適用するコンテキスト（グローバル設定を汚さない）。"""
    with plt.rc_context(rc_params(mode)):
        yield


def new_axes(ax=None, *, figsize: tuple[float, float] | None = None):
    """``ax`` が ``None`` なら新しい Figure と Axes を作る。

    既存の Axes を渡せばそこに描く（tearsheet のように図を組み合わせる用途）。
    """
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def finalize(
    ax,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = False,
    zero_line: bool = False,
):
    """軸の仕上げ。凡例は枠なし、ゼロ線は目盛より一段薄く。"""
    mode = _mode
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    if zero_line:
        ax.axhline(0.0, color=_CHROME[mode]["axis"], linewidth=0.8, zorder=1)
    if legend:
        ax.legend(loc="best")
    ax.set_axisbelow(True)
    return ax.figure


def heatmap(
    ax,
    matrix: np.ndarray,
    *,
    xticks: Sequence,
    yticks: Sequence,
    cmap,
    vmin: float,
    vmax: float,
    annotate: bool = False,
    fmt: str = "{:.1f}",
    mode: Mode | None = None,
):
    """セル間に背景色の隙間を空けたヒートマップを描く。

    枠線でセルを区切らず、隙間で分ける。数値を入れる場合は濃いセルで文字色を
    反転する（読めなくなるため）。

    Returns
    -------
    QuadMesh
        カラーバーを付けるために返す。
    """
    colors = palette(mode)
    mesh = ax.pcolormesh(
        matrix,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors=colors["surface"],
        linewidth=1.5,
    )
    ax.set_xticks(np.arange(len(xticks)) + 0.5, [str(x) for x in xticks])
    ax.set_yticks(np.arange(len(yticks)) + 0.5, [str(y) for y in yticks])
    ax.invert_yaxis()
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        limit = max(abs(vmin), abs(vmax))
        for (row, col), value in np.ndenumerate(matrix):
            if np.isnan(value):
                continue
            strong = abs(value) / limit > 0.55 if limit else False
            ax.text(
                col + 0.5,
                row + 0.5,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=7.5,
                color=colors["surface"] if strong else colors["ink"],
            )
    return mesh


def add_colorbar(
    fig,
    mesh,
    ax,
    *,
    ticks: Sequence[float] | None = None,
    percent: bool = False,
    decimals: int = 0,
):
    """ヒートマップ用の細いカラーバー。枠線と目盛の線は消す。"""
    bar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.03, shrink=0.9, ticks=ticks)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)
    if percent:
        percent_axis(bar.ax, decimals=decimals)
    return bar


def percent_axis(ax, which: str = "y", decimals: int = 0) -> None:
    """軸をパーセント表示にする。"""
    formatter = FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f}%")
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_formatter(formatter)


def use_japanese_font(name: str | None = None) -> str:
    """日本語ラベルを使えるようにフォントを設定する。

    既定のフォントには日本語グリフが無く、ラベルが豆腐になる。環境にある
    日本語フォントを順に探して設定する。

    Parameters
    ----------
    name :
        使うフォント名。``None`` なら一般的な候補から自動で探す。

    Returns
    -------
    str
        実際に設定されたフォント名。
    """
    from matplotlib import font_manager

    candidates = (
        [name]
        if name
        else [
            "Hiragino Sans",
            "Hiragino Kaku Gothic ProN",
            "Yu Gothic",
            "Meiryo",
            "MS Gothic",
            "Noto Sans CJK JP",
            "IPAexGothic",
            "TakaoGothic",
        ]
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    raise RuntimeError(
        f"日本語フォントが見つかりません（探した候補: {candidates}）。"
        " matplotlib に認識されるフォント名を name で指定してください。"
    )
