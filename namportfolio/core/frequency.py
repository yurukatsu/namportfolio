"""データ頻度の推定と年率化係数。

``pandas.infer_freq`` は不規則な日付列で ``None`` を返し、頻度文字列の表記も
バージョンで変わる（``"M"`` → ``"ME"``）。社内データは営業日ベースで祝日が
抜けるため、**日付差の中央値**から判定する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import ValidationError

__all__ = [
    "infer_periods_per_year",
    "resolve_periods_per_year",
    "volatility_scale",
    "DAILY",
    "WEEKLY",
    "MONTHLY",
    "QUARTERLY",
    "ANNUAL",
]

DAILY = 252.0
WEEKLY = 52.0
MONTHLY = 12.0
QUARTERLY = 4.0
ANNUAL = 1.0

#: (日付差中央値の上限[日], 年間期間数)。上から順に評価する。
_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (4.0, DAILY),  # 日次・営業日次（週末で 3 日空くため中央値は 1〜3）
    (9.0, WEEKLY),
    (20.0, 26.0),  # 隔週
    (45.0, MONTHLY),
    (75.0, 6.0),  # 隔月
    (135.0, QUARTERLY),
    (250.0, 2.0),  # 半期
    (450.0, ANNUAL),
)

_DAYS_PER_YEAR = 365.25


def infer_periods_per_year(index, *, default: float | None = None) -> float:
    """日付列から年間の期間数を推定する。

    Parameters
    ----------
    index :
        DatetimeIndex、index を持つ Series/DataFrame、または日付の配列。
    default :
        推定できない場合に返す値。``None`` なら例外を送出する。

    Returns
    -------
    float
        日次 252 / 週次 52 / 月次 12 など。既知の区分に当てはまらない場合は
        ``365.25 / 日付差の中央値``。
    """
    try:
        idx = _to_datetime_index(index)
        idx = idx.dropna().unique().sort_values()
        if len(idx) < 2:
            raise ValidationError(
                f"頻度の推定には 2 点以上の異なる日付が必要です（{len(idx)} 点）。"
                " periods_per_year を明示指定してください。"
            )
        diffs = np.diff(idx.to_numpy()).astype("timedelta64[s]").astype(float)
        median_days = float(np.median(diffs)) / 86400.0
        if median_days <= 0:
            raise ValidationError("日付差の中央値が 0 以下です。")
    except ValidationError:
        if default is not None:
            return float(default)
        raise

    for upper, periods in _THRESHOLDS:
        if median_days <= upper:
            return periods
    return _DAYS_PER_YEAR / median_days


def resolve_periods_per_year(data, periods_per_year: float | None = None) -> float:
    """明示指定があればそれを、なければ ``data`` から推定する。"""
    if periods_per_year is None:
        return infer_periods_per_year(data)
    if periods_per_year <= 0:
        raise ValidationError(f"periods_per_year は正の値である必要があります: {periods_per_year}")
    return float(periods_per_year)


def volatility_scale(periods_per_year: float) -> float:
    """ボラティリティ年率化の係数 :math:`\\sqrt{P}`。"""
    if periods_per_year <= 0:
        raise ValidationError(f"periods_per_year は正の値である必要があります: {periods_per_year}")
    return float(np.sqrt(periods_per_year))


def _to_datetime_index(obj) -> pd.DatetimeIndex:
    """Series/DataFrame/Index/配列から DatetimeIndex を取り出す。"""
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj = obj.index
    if isinstance(obj, pd.DatetimeIndex):
        return obj
    if isinstance(obj, pd.MultiIndex):
        raise ValidationError(
            "MultiIndex からは頻度を推定できません。日付レベルを取り出してから渡してください。"
        )
    try:
        return pd.DatetimeIndex(pd.to_datetime(obj))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"日付として解釈できません: {type(obj).__name__}") from exc
