"""Barra 型のリスク分解と、リスクモデルの妥当性検証。

エクスポージャー・ファクター共分散・特異リスク・ファクターリターンは**すべて
社内リスクモデルから供給される前提**で、推定は行わない。

.. rubric:: 入力の形

- ``data``: long 形式（``date`` / ``bid`` ＋ウェイト列＋**ファクター列**）。
  エクスポージャーはウェイトと同じ行に並べておく
- ``covariance``: ファクター共分散。long（``date`` / ``factor_1`` / ``factor_2`` /
  値）または「日付ごとの行列を縦に積んだ」形
- ``factor_returns``: ``index=date``, ``columns=factor`` の wide、または long

.. rubric:: 分散は足せるがリスクは足せない

.. math::

    \\sigma_p^2 = w^\\top X F X^\\top w + w^\\top D w

``factor_risk + specific_risk`` は ``total_risk`` にならない（二乗和で合成される）。
足し算で内訳を見たいときは :func:`factor_risk_contribution` の ``cctr`` を使う。
こちらは合計が ``total_risk`` に一致する。

.. rubric:: アクティブリスク

``benchmark_weight`` を指定すると、以降すべての計算がアクティブウェイト
（ポート − ベンチ）ベースになり、``total_risk`` はトラッキングエラーになる。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .core.config import resolve_columns
from .core.errors import ValidationError
from .core.frequency import resolve_periods_per_year, volatility_scale
from .core.panel import require_columns

__all__ = [
    "factor_exposures",
    "risk_decomposition",
    "factor_risk_contribution",
    "factor_return_attribution",
    "bias_statistic",
    "rolling_bias_statistic",
    "bias_confidence_interval",
    "realized_vs_predicted",
]


# --------------------------------------------------------------------------
# F6: リスク分解
# --------------------------------------------------------------------------


def factor_exposures(
    data: pd.DataFrame,
    *,
    factors: Sequence[str],
    weight: str = "weight",
    benchmark_weight: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """ポートフォリオのファクターエクスポージャー :math:`x = X^\\top w`。

    Parameters
    ----------
    factors :
        ファクター列の名前。
    benchmark_weight :
        指定するとアクティブエクスポージャー（ポート − ベンチ）を返す。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns=factors``。
    """
    date_col, _ = resolve_columns(date_col, id_col)
    needed = [date_col, weight, *factors] + ([benchmark_weight] if benchmark_weight else [])
    require_columns(data, needed, context="factor_exposures")

    dates = pd.to_datetime(data[date_col])
    active = _active_weight(data, weight, benchmark_weight)
    weighted = data[list(factors)].astype(float).mul(active, axis=0)
    exposures = weighted.groupby(dates, observed=True).sum()
    exposures.index.name = date_col
    return exposures.sort_index()


def risk_decomposition(
    data: pd.DataFrame,
    covariance: pd.DataFrame,
    *,
    factors: Sequence[str],
    weight: str = "weight",
    benchmark_weight: str | None = None,
    specific_risk: str = "specific_risk",
    specific_is_variance: bool = False,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """トータル・ファクター・特異リスクに分解する。

    Parameters
    ----------
    covariance :
        ファクター共分散。``date`` ごとの :math:`K \\times K` 行列。
    specific_risk :
        特異リスクのカラム名。既定では**標準偏差**として扱う。
    specific_is_variance :
        ``True`` なら ``specific_risk`` 列を分散として扱う。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns=[total_risk, factor_risk, specific_risk]``。
        単位は共分散行列に合わせた**リスク（標準偏差）**。年率共分散を渡せば年率。

    Notes
    -----
    ``factor_risk + specific_risk != total_risk``。分散が加法的で、リスクはその
    平方根なので、内訳を足し算で見たい場合は :func:`factor_risk_contribution` を使う。
    """
    date_col, id_col = resolve_columns(date_col, id_col)
    require_columns(data, [date_col, specific_risk], context="risk_decomposition")

    exposures = factor_exposures(
        data,
        factors=factors,
        weight=weight,
        benchmark_weight=benchmark_weight,
        date_col=date_col,
        id_col=id_col,
    )
    dates = pd.to_datetime(data[date_col])
    active = _active_weight(data, weight, benchmark_weight)
    specific = data[specific_risk].astype(float)
    specific_variance = specific if specific_is_variance else specific**2

    idiosyncratic = (active**2 * specific_variance).groupby(dates, observed=True).sum()
    matrices = _covariance_by_date(covariance, factors, date_col)

    rows = {}
    for date, exposure in exposures.iterrows():
        matrix = matrices.get(date)
        if matrix is None:
            continue
        vector = exposure.to_numpy(dtype=float)
        factor_variance = float(vector @ matrix @ vector)
        residual = float(idiosyncratic.get(date, 0.0))
        rows[date] = {
            "total_risk": np.sqrt(max(factor_variance + residual, 0.0)),
            "factor_risk": np.sqrt(max(factor_variance, 0.0)),
            "specific_risk": np.sqrt(max(residual, 0.0)),
        }

    frame = pd.DataFrame(rows).T
    frame.index.name = date_col
    return frame.sort_index()


def factor_risk_contribution(
    data: pd.DataFrame,
    covariance: pd.DataFrame,
    *,
    factors: Sequence[str],
    weight: str = "weight",
    benchmark_weight: str | None = None,
    specific_risk: str = "specific_risk",
    specific_is_variance: bool = False,
    factor_groups: Mapping[str, str] | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """ファクター別のリスク寄与（MCTR / CCTR）。

    .. math::

        \\mathrm{MCTR}_k = \\frac{(F x)_k}{\\sigma}, \\qquad
        \\mathrm{CCTR}_k = x_k \\cdot \\mathrm{MCTR}_k

    ``cctr`` は加法的で、**全ファクターの合計＋特異寄与がトータルリスクに一致する**。
    特異寄与は ``factor`` が ``"specific"`` の行として入る。

    Parameters
    ----------
    factor_groups :
        ファクター名 → グループ名（``"style"`` / ``"industry"`` など）の対応。
        指定するとグループ単位に集計する。

    Returns
    -------
    pd.DataFrame
        ``MultiIndex (date, factor)``、``columns=[exposure, mctr, cctr, pct_of_total]``。
        グループ集計時は ``exposure`` と ``mctr`` が意味を持たないので ``NaN``。
    """
    date_col, id_col = resolve_columns(date_col, id_col)
    exposures = factor_exposures(
        data,
        factors=factors,
        weight=weight,
        benchmark_weight=benchmark_weight,
        date_col=date_col,
        id_col=id_col,
    )
    totals = risk_decomposition(
        data,
        covariance,
        factors=factors,
        weight=weight,
        benchmark_weight=benchmark_weight,
        specific_risk=specific_risk,
        specific_is_variance=specific_is_variance,
        date_col=date_col,
        id_col=id_col,
    )
    matrices = _covariance_by_date(covariance, factors, date_col)

    records = []
    for date, exposure in exposures.iterrows():
        matrix = matrices.get(date)
        if matrix is None or date not in totals.index:
            continue
        sigma = float(totals.loc[date, "total_risk"])
        if sigma <= 0:
            continue

        vector = exposure.to_numpy(dtype=float)
        marginal = matrix @ vector / sigma
        contribution = vector * marginal
        for name, exp, mctr, cctr in zip(factors, vector, marginal, contribution, strict=True):
            records.append(
                {
                    date_col: date,
                    "factor": name,
                    "exposure": exp,
                    "mctr": mctr,
                    "cctr": cctr,
                }
            )
        # 特異リスクの寄与。これを足すと合計が sigma になる
        records.append(
            {
                date_col: date,
                "factor": "specific",
                "exposure": np.nan,
                "mctr": np.nan,
                "cctr": float(totals.loc[date, "specific_risk"]) ** 2 / sigma,
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame = frame.set_index([date_col, "factor"]).sort_index()

    if factor_groups is not None:
        frame = _group_contributions(frame, factor_groups, date_col)

    sigma_by_date = frame.index.get_level_values(0).map(totals["total_risk"])
    frame["pct_of_total"] = frame["cctr"] / sigma_by_date
    return frame


def factor_return_attribution(
    data: pd.DataFrame,
    factor_returns: pd.DataFrame,
    *,
    factors: Sequence[str],
    weight: str = "weight",
    benchmark_weight: str | None = None,
    specific_return: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """リターンをファクター寄与と特異リターンに分解する。

    .. math::

        r_p = \\sum_k x_k f_k + \\sum_i w_i u_i

    Parameters
    ----------
    factor_returns :
        ファクターリターン。``index=date`` / ``columns=factor`` の wide、または
        ``date`` / ``factor`` / 値の long。
    specific_return :
        スペシフィックリターンのカラム名。指定すると ``"specific"`` の行が加わり、
        全行の合計が（アクティブ）リターンに一致する。

    Returns
    -------
    pd.DataFrame
        ``MultiIndex (date, factor)``、``columns=[exposure, factor_return, contribution]``。
    """
    date_col, id_col = resolve_columns(date_col, id_col)
    exposures = factor_exposures(
        data,
        factors=factors,
        weight=weight,
        benchmark_weight=benchmark_weight,
        date_col=date_col,
        id_col=id_col,
    )
    returns = _as_factor_frame(factor_returns, factors, date_col)
    exposures, returns = exposures.align(returns, join="inner", axis=0)

    stacked = pd.DataFrame(
        {
            "exposure": exposures.stack(),
            "factor_return": returns.stack(),
        }
    )
    stacked["contribution"] = stacked["exposure"] * stacked["factor_return"]
    stacked.index.names = [date_col, "factor"]

    if specific_return is not None:
        require_columns(data, [specific_return], context="factor_return_attribution")
        dates = pd.to_datetime(data[date_col])
        active = _active_weight(data, weight, benchmark_weight)
        residual = (active * data[specific_return]).groupby(dates, observed=True).sum()
        residual = residual.reindex(exposures.index)
        extra = pd.DataFrame(
            {
                "exposure": np.nan,
                "factor_return": np.nan,
                "contribution": residual,
            }
        )
        extra.index = pd.MultiIndex.from_product(
            [residual.index, ["specific"]], names=[date_col, "factor"]
        )
        stacked = pd.concat([stacked, extra])

    return stacked.sort_index()


# --------------------------------------------------------------------------
# F7: リスクモデルの妥当性検証
# --------------------------------------------------------------------------


def bias_statistic(
    realized: pd.Series,
    predicted: pd.Series,
    *,
    periods_per_year: float | None = None,
    predicted_is_annualized: bool = True,
) -> float:
    """バイアス統計量。

    標準化リターン :math:`z_t = r_t / \\sigma_t` の標準偏差。**1 に近ければ
    リスク予測が正しい**。1 より大きければリスクを過小評価、小さければ過大評価。

    Parameters
    ----------
    realized :
        実現リターン。``predicted`` と同じ期間の値。
    predicted :
        その期間の**事前**リスク予測。``predicted[t]`` は t 期のリスク予測として
        扱う（前期末時点の値を持っている場合は呼び出し側で ``shift`` しない）。
    predicted_is_annualized :
        ``True``（既定）なら予測を期間リスクに割り戻してから使う。

    Returns
    -------
    float
        有効データが 2 点未満なら ``NaN``。
    """
    standardized = _standardized_returns(
        realized, predicted, periods_per_year, predicted_is_annualized
    )
    if len(standardized) < 2:
        return np.nan
    return float(standardized.std(ddof=1))


def rolling_bias_statistic(
    realized: pd.Series,
    predicted: pd.Series,
    window: int,
    *,
    periods_per_year: float | None = None,
    predicted_is_annualized: bool = True,
) -> pd.Series:
    """バイアス統計量の推移。特定の局面だけ予測が外れていないかを見る。"""
    standardized = _standardized_returns(
        realized, predicted, periods_per_year, predicted_is_annualized
    )
    return standardized.rolling(window).std(ddof=1).rename("bias_statistic")


def bias_confidence_interval(n_periods: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """バイアス統計量が 1 とみなせる範囲。

    :math:`1 \\pm z \\big/ \\sqrt{2T}` の近似。この幅の外に出ていれば、リスク予測が
    偏っていると判断できる。

    Parameters
    ----------
    n_periods :
        標本数。
    confidence :
        信頼水準（``0.95`` / ``0.99`` に対応）。
    """
    if n_periods < 2:
        return (np.nan, np.nan)
    critical = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence)
    if critical is None:
        raise ValidationError(f"confidence は 0.90 / 0.95 / 0.99 のいずれかです: {confidence}")
    half_width = critical / np.sqrt(2.0 * n_periods)
    return (1.0 - half_width, 1.0 + half_width)


def realized_vs_predicted(
    realized: pd.Series,
    predicted: pd.Series,
    *,
    window: int = 12,
    periods_per_year: float | None = None,
    predicted_is_annualized: bool = True,
) -> pd.DataFrame:
    """予測リスクと実現ボラティリティを並べる。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns=[predicted, realized, ratio]``。いずれも年率。
        ``ratio`` が 1 を継続的に上回れば、リスクを過小評価している。
    """
    aligned, forecast = realized.align(predicted, join="inner")
    ppy = resolve_periods_per_year(aligned, periods_per_year)
    scale = volatility_scale(ppy)

    realized_annual = aligned.rolling(window).std(ddof=1) * scale
    predicted_annual = forecast if predicted_is_annualized else forecast * scale
    return pd.DataFrame(
        {
            "predicted": predicted_annual,
            "realized": realized_annual,
            "ratio": realized_annual / predicted_annual.replace(0, np.nan),
        }
    )


# --------------------------------------------------------------------------
# 内部ヘルパー
# --------------------------------------------------------------------------


def _active_weight(data: pd.DataFrame, weight: str, benchmark_weight: str | None) -> pd.Series:
    """ベンチが指定されていればアクティブウェイト、なければポートウェイト。"""
    portfolio = data[weight].astype(float).fillna(0.0)
    if benchmark_weight is None:
        return portfolio
    return portfolio - data[benchmark_weight].astype(float).fillna(0.0)


def _covariance_by_date(
    covariance: pd.DataFrame,
    factors: Sequence[str],
    date_col: str,
) -> dict[pd.Timestamp, np.ndarray]:
    """日付 → :math:`K \\times K` 行列の辞書に変換する。

    long（``date`` / ``factor_1`` / ``factor_2`` / 値）と、日付ごとの行列を縦に
    積んだ形（``date`` ＋ファクター名の行と列）の両方を受ける。
    """
    factors = list(factors)
    if date_col not in covariance.columns:
        raise ValidationError(
            f"共分散に '{date_col}' カラムが必要です (存在するカラム: {list(covariance.columns)})"
        )
    frame = covariance.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])

    remaining = [c for c in frame.columns if c != date_col]
    is_long = len(remaining) == 3 and not set(factors).issubset(remaining)

    matrices: dict[pd.Timestamp, np.ndarray] = {}
    if is_long:
        row_col, col_col, value_col = remaining
        for date, chunk in frame.groupby(date_col, observed=True):
            table = chunk.pivot(index=row_col, columns=col_col, values=value_col)
            table = table.reindex(index=factors, columns=factors)
            # 片側しか持たない形式でも対称行列にする
            filled = table.to_numpy(dtype=float)
            filled = np.where(np.isnan(filled), filled.T, filled)
            matrices[date] = np.nan_to_num(filled)
        return matrices

    missing = [f for f in factors if f not in frame.columns]
    if missing:
        raise ValidationError(f"共分散にファクター列がありません: {missing}")
    label_col = next(c for c in remaining if c not in factors)
    for date, chunk in frame.groupby(date_col, observed=True):
        table = chunk.set_index(label_col)[factors].reindex(index=factors)
        matrices[date] = np.nan_to_num(table.to_numpy(dtype=float))
    return matrices


def _as_factor_frame(
    factor_returns: pd.DataFrame,
    factors: Sequence[str],
    date_col: str,
) -> pd.DataFrame:
    """ファクターリターンを ``index=date`` / ``columns=factor`` に揃える。"""
    factors = list(factors)
    if set(factors).issubset(factor_returns.columns):
        frame = factor_returns
        if date_col in frame.columns:
            frame = frame.set_index(date_col)
        frame = frame[factors]
    else:
        if date_col not in factor_returns.columns:
            raise ValidationError(
                f"ファクターリターンに '{date_col}' カラムか、ファクター列 {factors} が必要です。"
            )
        others = [c for c in factor_returns.columns if c != date_col]
        if len(others) != 2:
            raise ValidationError(
                "long 形式のファクターリターンは date / factor / 値 の 3 列にしてください。"
            )
        name_col, value_col = others
        frame = factor_returns.pivot(index=date_col, columns=name_col, values=value_col)
        frame = frame.reindex(columns=factors)

    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def _group_contributions(
    frame: pd.DataFrame,
    factor_groups: Mapping[str, str],
    date_col: str,
) -> pd.DataFrame:
    """ファクターをグループに畳む。cctr は加法的なので単純に合計できる。"""
    names = frame.index.get_level_values("factor")
    labels = [factor_groups.get(name, name) for name in names]
    grouped = (
        frame.assign(_group=labels)
        .groupby([frame.index.get_level_values(0), "_group"], observed=True)["cctr"]
        .sum()
        .to_frame()
    )
    grouped.index.names = [date_col, "factor"]
    # グループ単位ではエクスポージャーと限界寄与が定義できない
    grouped["exposure"] = np.nan
    grouped["mctr"] = np.nan
    return grouped[["exposure", "mctr", "cctr"]]


def _standardized_returns(
    realized: pd.Series,
    predicted: pd.Series,
    periods_per_year: float | None,
    predicted_is_annualized: bool,
) -> pd.Series:
    """:math:`z_t = r_t / \\sigma_t`。予測が年率なら期間リスクに割り戻す。"""
    aligned, forecast = realized.align(predicted, join="inner")
    if predicted_is_annualized:
        if len(aligned) < 2 and periods_per_year is None:
            # 頻度を推定できないので年率を期間に割り戻せない。
            # そもそも 2 点未満では統計量が定義できないため空で返す
            return aligned.iloc[:0]
        ppy = resolve_periods_per_year(aligned, periods_per_year)
        forecast = forecast / volatility_scale(ppy)
    return (aligned / forecast.replace(0, np.nan)).dropna()
