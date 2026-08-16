import numpy as np
import pandas as pd
import pytest

from namportfolio.core.errors import ValidationError
from namportfolio.core.frequency import (
    infer_periods_per_year,
    resolve_periods_per_year,
    volatility_scale,
)


@pytest.mark.parametrize(
    ("freq", "periods", "expected"),
    [
        ("B", 300, 252.0),  # 営業日次
        ("D", 300, 252.0),  # カレンダー日次
        ("W", 100, 52.0),  # 週次
        ("MS", 36, 12.0),  # 月次
        ("QS", 20, 4.0),  # 四半期
        ("YS", 10, 1.0),  # 年次
    ],
)
def test_infer_known_frequencies(freq, periods, expected):
    idx = pd.date_range("2015-01-01", periods=periods, freq=freq)
    assert infer_periods_per_year(idx) == expected


def test_infer_from_series_index():
    idx = pd.date_range("2020-01-01", periods=36, freq="MS")
    assert infer_periods_per_year(pd.Series(np.arange(36.0), index=idx)) == 12.0


def test_business_days_with_holidays_still_daily():
    """祝日が抜けても日次と判定される（中央値ベースのため）。"""
    idx = pd.date_range("2020-01-01", periods=300, freq="B").delete([10, 11, 40, 41, 42, 100])
    assert infer_periods_per_year(idx) == 252.0


def test_unsorted_and_duplicated_dates():
    idx = pd.DatetimeIndex(["2020-03-01", "2020-01-01", "2020-02-01", "2020-01-01", "2020-04-01"])
    assert infer_periods_per_year(idx) == 12.0


def test_irregular_frequency_falls_back_to_calendar_ratio():
    idx = pd.date_range("2020-01-01", periods=10, freq="500D")
    assert infer_periods_per_year(idx) == pytest.approx(365.25 / 500.0)


def test_single_date_raises_or_uses_default():
    idx = pd.DatetimeIndex(["2020-01-01"])
    with pytest.raises(ValidationError, match="2 点以上"):
        infer_periods_per_year(idx)
    assert infer_periods_per_year(idx, default=12.0) == 12.0


def test_multiindex_raises():
    idx = pd.MultiIndex.from_tuples([(pd.Timestamp("2020-01-01"), "A")])
    with pytest.raises(ValidationError, match="MultiIndex"):
        infer_periods_per_year(idx)


def test_resolve_prefers_explicit():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    assert resolve_periods_per_year(idx, 12.0) == 12.0
    assert resolve_periods_per_year(idx, None) == 252.0
    with pytest.raises(ValidationError, match="正の値"):
        resolve_periods_per_year(idx, 0.0)


def test_volatility_scale():
    assert volatility_scale(252.0) == pytest.approx(np.sqrt(252.0))
