"""Brinson 型のリターン要因分解。

アクティブリターンを「どのセグメントに賭けたか（配分）」と「その中でどの銘柄を
選んだか（選択）」に分ける。

入力は **2 経路**。銘柄レベルのデータから集計する経路と、セグメント集計済みの
データを直接渡す経路。前者を集計して後者に合流させ、以降の計算は共通。

    # 銘柄レベル（date, bid, weight, bench_weight, ret_1m, sector）
    effects = npf.attribution.brinson(df, segment="sector", asset_return="ret_1m")

    # セグメント集計済み（date, sector, wp, wb, rp, rb）
    effects = npf.attribution.brinson(
        seg, segment="sector", portfolio_return="rp", benchmark_return="rb"
    )

.. rubric:: 単期間の効果は足し算で閉じる

``allocation + selection + interaction`` の総和は、その期間のアクティブリターン
（``rp - rb``）に一致する。**期間をまたぐときは一致しない**（複利があるため）。
:func:`link` で期間リンクしてから足し上げること。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core.config import resolve_columns
from .core.errors import ValidationError
from .core.panel import require_columns

__all__ = [
    "aggregate_segments",
    "brinson",
    "effects_from_segments",
    "total_returns",
    "link",
    "brinson_summary",
]

_MODELS = ("bf", "bhb")
_LINKING = ("carino", "grap", "frongello", "simple")
_TOLERANCE = 1e-12


def aggregate_segments(
    data: pd.DataFrame,
    *,
    segment: str,
    portfolio_weight: str = "weight",
    benchmark_weight: str = "bench_weight",
    asset_return: str | None = None,
    portfolio_return: str | None = None,
    benchmark_return: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """帰属計算に必要な 4 つの量をセグメント単位に揃える。

    2 経路のどちらでも同じ形の DataFrame を返す。

    Parameters
    ----------
    asset_return :
        **銘柄レベル経路。** 銘柄リターンのカラム名。指定すると、セグメント内の
        ウェイト加重平均でポート／ベンチのセグメントリターンを計算する。
    portfolio_return, benchmark_return :
        **集計済み経路。** セグメントリターンのカラム名。両方を指定する。

    Returns
    -------
    pd.DataFrame
        ``MultiIndex (date, segment)``、``columns=[wp, wb, rp, rb]``。

    Notes
    -----
    銘柄レベルから集計する場合、``rp`` と ``rb`` は**同じ銘柄リターン**をポート
    ウェイト／ベンチウェイトで加重した値になる。セグメント内の銘柄選択効果は
    ウェイトの違いから生じるので、これが正しい扱い。
    """
    date_col, id_col = resolve_columns(date_col, id_col)
    by_asset = asset_return is not None
    by_segment = portfolio_return is not None and benchmark_return is not None
    if by_asset == by_segment:
        raise ValidationError(
            "asset_return（銘柄レベル）か portfolio_return と benchmark_return"
            "（集計済み）の**どちらか**を指定してください。"
        )

    needed = [date_col, segment, portfolio_weight, benchmark_weight]
    needed += [asset_return] if by_asset else [portfolio_return, benchmark_return]
    require_columns(data, needed, context="aggregate_segments")

    dates = pd.to_datetime(data[date_col])
    keys = [dates.rename(date_col), data[segment].rename(segment)]

    weights = data.groupby(keys, observed=True)[[portfolio_weight, benchmark_weight]].sum()
    frame = weights.rename(columns={portfolio_weight: "wp", benchmark_weight: "wb"})

    if by_asset:
        frame["rp"] = _weighted_return(data, asset_return, portfolio_weight, keys)
        frame["rb"] = _weighted_return(data, asset_return, benchmark_weight, keys)
    else:
        aggregated = data.groupby(keys, observed=True)[[portfolio_return, benchmark_return]].first()
        frame["rp"] = aggregated[portfolio_return]
        frame["rb"] = aggregated[benchmark_return]

    return frame.sort_index()


def brinson(
    data: pd.DataFrame,
    *,
    segment: str,
    portfolio_weight: str = "weight",
    benchmark_weight: str = "bench_weight",
    asset_return: str | None = None,
    portfolio_return: str | None = None,
    benchmark_return: str | None = None,
    model: str = "bf",
    terms: int = 3,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """単期間の Brinson 帰属効果。

    Parameters
    ----------
    model :
        ``"bf"``（Brinson-Fachler、既定）は配分効果を**ベンチマーク全体に対する
        超過**で測る。``"bhb"``（Brinson-Hood-Beebower）はセグメントリターン
        そのもので測る。BF のほうが「市場全体が上がっただけ」の効果を配分に
        計上しないぶん解釈しやすい。
    terms :
        ``3`` なら allocation / selection / interaction の 3 項。``2`` なら
        交互作用を selection に含める（selection がポートウェイト基準になる）。

    Returns
    -------
    pd.DataFrame
        ``MultiIndex (date, segment)``、``columns`` は効果名。
        期間ごとの総和がその期間のアクティブリターンに一致する。
    """
    if model not in _MODELS:
        raise ValidationError(f"model は {list(_MODELS)} のいずれかです: {model!r}")
    if terms not in (2, 3):
        raise ValidationError(f"terms は 2 か 3 です: {terms}")

    segments = aggregate_segments(
        data,
        segment=segment,
        portfolio_weight=portfolio_weight,
        benchmark_weight=benchmark_weight,
        asset_return=asset_return,
        portfolio_return=portfolio_return,
        benchmark_return=benchmark_return,
        date_col=date_col,
        id_col=id_col,
    )
    return effects_from_segments(segments, model=model, terms=terms)


def effects_from_segments(
    segments: pd.DataFrame,
    *,
    model: str = "bf",
    terms: int = 3,
) -> pd.DataFrame:
    """``aggregate_segments`` の出力から効果を計算する（低レベル）。"""
    require_columns(segments, ["wp", "wb", "rp", "rb"], context="effects_from_segments")

    benchmark_total = (segments["wb"] * segments["rb"]).groupby(level=0).sum()
    total_by_row = segments.index.get_level_values(0).map(benchmark_total)

    weight_diff = segments["wp"] - segments["wb"]
    return_diff = segments["rp"] - segments["rb"]

    if model == "bf":
        allocation = weight_diff * (segments["rb"] - total_by_row)
    else:
        allocation = weight_diff * segments["rb"]

    if terms == 3:
        effects = {
            "allocation": allocation,
            "selection": segments["wb"] * return_diff,
            "interaction": weight_diff * return_diff,
        }
    else:
        # 交互作用を選択効果に統合すると、選択効果がポートウェイト基準になる
        effects = {
            "allocation": allocation,
            "selection": segments["wp"] * return_diff,
        }
    return pd.DataFrame(effects)


def total_returns(
    data: pd.DataFrame,
    *,
    segment: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """期間ごとのポート／ベンチ／アクティブリターン。

    :func:`link` に渡すために使う。``segment`` を指定すれば :func:`brinson` と
    同じ引数で呼べるし、``aggregate_segments`` の出力を直接渡してもよい。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns=[portfolio, benchmark, active]``。
    """
    if segment is not None:
        data = aggregate_segments(data, segment=segment, **kwargs)
    require_columns(data, ["wp", "wb", "rp", "rb"], context="total_returns")

    portfolio = (data["wp"] * data["rp"]).groupby(level=0).sum()
    benchmark = (data["wb"] * data["rb"]).groupby(level=0).sum()
    return pd.DataFrame(
        {"portfolio": portfolio, "benchmark": benchmark, "active": portfolio - benchmark}
    )


def link(
    effects: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    method: str = "carino",
) -> pd.DataFrame:
    """効果を期間リンクする。

    単期間の効果をそのまま足すと、複利のぶん幾何アクティブリターンと合わない。
    リンキングは各期間の効果に係数を掛けて、**総和が幾何アクティブリターンに
    一致する**ようにする。

    Parameters
    ----------
    effects :
        :func:`brinson` の出力。
    returns :
        :func:`total_returns` の出力。
    method :
        ``"carino"``（既定、対数ベースの平滑化）/ ``"grap"``（前期までのポート
        成長と後期のベンチ成長を掛ける）/ ``"frongello"``（再帰的）/
        ``"simple"``（そのまま。合計は近似になる）。

    Returns
    -------
    pd.DataFrame
        ``effects`` と同じ形。``simple`` 以外は総和が幾何アクティブリターンに一致する。
    """
    if method not in _LINKING:
        raise ValidationError(f"method は {list(_LINKING)} のいずれかです: {method!r}")
    require_columns(returns, ["portfolio", "benchmark"], context="link")

    if method == "simple":
        return effects.copy()
    if method == "frongello":
        return _frongello(effects, returns)

    factors = _carino_factors(returns) if method == "carino" else _grap_factors(returns)
    return effects.mul(factors, axis=0, level=0)


def brinson_summary(
    data: pd.DataFrame,
    *,
    segment: str,
    portfolio_weight: str = "weight",
    benchmark_weight: str = "bench_weight",
    asset_return: str | None = None,
    portfolio_return: str | None = None,
    benchmark_return: str | None = None,
    model: str = "bf",
    terms: int = 3,
    method: str = "carino",
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """セグメント別の累積効果（集計・効果計算・期間リンクを通しで実行）。

    Returns
    -------
    pd.DataFrame
        ``index=segment``、``columns`` は効果名＋``total``。最終行 ``Total`` に
        全セグメントの合計を入れる。この ``Total`` の ``total`` が幾何アクティブ
        リターンに一致する（``method="simple"`` を除く）。
    """
    segments = aggregate_segments(
        data,
        segment=segment,
        portfolio_weight=portfolio_weight,
        benchmark_weight=benchmark_weight,
        asset_return=asset_return,
        portfolio_return=portfolio_return,
        benchmark_return=benchmark_return,
        date_col=date_col,
        id_col=id_col,
    )
    effects = effects_from_segments(segments, model=model, terms=terms)
    linked = link(effects, total_returns(segments), method=method)

    summary = linked.groupby(level=1, observed=True).sum()
    summary["total"] = summary.sum(axis=1)
    summary.loc["Total"] = summary.sum(axis=0)
    summary.index.name = segment
    return summary


# --------------------------------------------------------------------------
# 内部ヘルパー
# --------------------------------------------------------------------------


def _weighted_return(
    data: pd.DataFrame,
    value: str,
    weight: str,
    keys: list[pd.Series],
) -> pd.Series:
    """セグメント内のウェイト加重平均リターン。ウェイト 0 のセグメントは 0。"""
    weighted = (data[value] * data[weight]).groupby(keys, observed=True).sum()
    total = data[weight].groupby(keys, observed=True).sum()
    return (weighted / total.replace(0, np.nan)).fillna(0.0)


def _carino_factors(returns: pd.DataFrame) -> pd.Series:
    """Carino (1999) の平滑化係数。"""
    portfolio = returns["portfolio"].to_numpy(dtype=float)
    benchmark = returns["benchmark"].to_numpy(dtype=float)

    period = _log_ratio(portfolio, benchmark)
    total_portfolio = float(np.prod(1.0 + portfolio) - 1.0)
    total_benchmark = float(np.prod(1.0 + benchmark) - 1.0)
    overall = _log_ratio(np.array([total_portfolio]), np.array([total_benchmark]))[0]

    return pd.Series(period / overall, index=returns.index)


def _log_ratio(portfolio: np.ndarray, benchmark: np.ndarray) -> np.ndarray:
    """``[ln(1+rp) - ln(1+rb)] / (rp - rb)``。差が 0 なら極限値 ``1/(1+rp)``。"""
    difference = portfolio - benchmark
    close = np.abs(difference) < _TOLERANCE
    safe = np.where(close, 1.0, difference)
    return np.where(
        close,
        1.0 / (1.0 + portfolio),
        (np.log1p(portfolio) - np.log1p(benchmark)) / safe,
    )


def _grap_factors(returns: pd.DataFrame) -> pd.Series:
    """GRAP 係数。前期までのポート成長 × 後期のベンチ成長。"""
    portfolio = 1.0 + returns["portfolio"]
    benchmark = 1.0 + returns["benchmark"]
    before = portfolio.cumprod().shift(1, fill_value=1.0)
    after = benchmark.prod() / benchmark.cumprod()
    return before * after


def _frongello(effects: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Frongello の再帰リンク。

    ``adjusted_t = effect_t * Π_{s<t}(1+rp_s) + rb_t * Σ_{s<t} adjusted_s``
    """
    dates = returns.index
    growth = 1.0
    cumulative: pd.DataFrame | None = None
    adjusted = []

    for date in dates:
        current = effects.xs(date, level=0)
        value = current * growth
        if cumulative is not None:
            value = value + float(returns.loc[date, "benchmark"]) * cumulative
        cumulative = value if cumulative is None else cumulative + value

        adjusted.append(pd.concat({date: value}, names=effects.index.names[:1]))
        growth *= 1.0 + float(returns.loc[date, "portfolio"])

    return pd.concat(adjusted).reindex(effects.index)
