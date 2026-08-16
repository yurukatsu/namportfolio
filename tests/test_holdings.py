import numpy as np
import pandas as pd
import pytest

from namportfolio import holdings
from namportfolio.core.errors import ValidationError
from namportfolio.performance import MONTH_END

DATES = pd.date_range("2024-01-31", periods=3, freq=MONTH_END)

# 3 期目に JP0003 を売却し、その分 JP0000 を買い増す
WEIGHTS = [
    {"JP0000": 0.4, "JP0001": 0.3, "JP0002": 0.2, "JP0003": 0.1},
    {"JP0000": 0.4, "JP0001": 0.3, "JP0002": 0.2, "JP0003": 0.1},
    {"JP0000": 0.5, "JP0001": 0.3, "JP0002": 0.2},
]
BENCH = {"JP0000": 0.25, "JP0001": 0.25, "JP0002": 0.25, "JP0003": 0.25}
RETURNS = {"JP0000": 0.10, "JP0001": -0.05, "JP0002": 0.02, "JP0003": 0.00}
SECTORS = {"JP0000": "A", "JP0001": "A", "JP0002": "B", "JP0003": "B"}
PER = {"JP0000": 10.0, "JP0001": 20.0, "JP0002": 30.0, "JP0003": 40.0}


@pytest.fixture
def panel():
    rows = [
        {
            "date": date,
            "bid": bid,
            "weight": weight,
            "bench_weight": BENCH[bid],
            "ret_1m": RETURNS[bid],
            "sector": SECTORS[bid],
            "per": PER[bid],
        }
        for date, snapshot in zip(DATES, WEIGHTS, strict=True)
        for bid, weight in snapshot.items()
    ]
    return pd.DataFrame(rows)


class TestConcentration:
    def test_counts_and_hhi(self, panel):
        result = holdings.concentration(panel, top=(2,))
        assert result["n_holdings"].iloc[0] == 4
        # 0.4^2 + 0.3^2 + 0.2^2 + 0.1^2
        assert result["hhi"].iloc[0] == pytest.approx(0.30)
        assert result["effective_n"].iloc[0] == pytest.approx(1 / 0.30)
        assert result["top2_share"].iloc[0] == pytest.approx(0.7)

    def test_missing_rows_count_as_zero(self, panel):
        """3 期目に行が無い JP0003 は「保有していない」扱い。"""
        result = holdings.concentration(panel)
        assert result["n_holdings"].iloc[2] == 3

    def test_top_share_uses_largest(self, panel):
        result = holdings.concentration(panel, top=(1, 4))
        assert result["top1_share"].iloc[0] == pytest.approx(0.4)
        assert result["top4_share"].iloc[0] == pytest.approx(1.0)


class TestAllocation:
    def test_portfolio_allocation(self, panel):
        result = holdings.allocation(panel, by="sector")
        assert list(result.columns) == ["A", "B"]
        assert result.loc[DATES[0], "A"] == pytest.approx(0.7)
        assert result.loc[DATES[0], "B"] == pytest.approx(0.3)

    def test_active_allocation(self, panel):
        result = holdings.allocation(panel, by="sector", benchmark_weight="bench_weight")
        assert result.loc[DATES[0], "A"] == pytest.approx(0.7 - 0.5)
        assert result.loc[DATES[0], "B"] == pytest.approx(0.3 - 0.5)

    def test_missing_segment_column(self, panel):
        with pytest.raises(ValidationError, match="必須カラム"):
            holdings.allocation(panel, by="country")


class TestTopHoldings:
    def test_defaults_to_last_date(self, panel):
        result = holdings.top_holdings(panel, n=2)
        assert result.index[0] == "JP0000", "ウェイト最大"
        assert "JP0003" not in result.index, "最終日は売却済み"

    def test_explicit_date(self, panel):
        result = holdings.top_holdings(panel, at=DATES[0], n=1)
        assert list(result.index) == ["JP0000", "JP0003"], "上位と下位"

    def test_active_sorting(self, panel):
        result = holdings.top_holdings(panel, at=DATES[0], n=1, benchmark_weight="bench_weight")
        assert "active" in result.columns
        assert result["active"].iloc[0] == pytest.approx(0.4 - 0.25)
        assert result.index[-1] == "JP0003", "最大のアンダーウェイト"

    def test_unknown_date(self, panel):
        with pytest.raises(ValidationError, match="データがありません"):
            holdings.top_holdings(panel, at="2030-01-31")


class TestCharacteristics:
    def test_weighted_average(self, panel):
        result = holdings.characteristics(panel, columns=["per"])
        # 0.4*10 + 0.3*20 + 0.2*30 + 0.1*40
        assert result.loc[DATES[0], "per"] == pytest.approx(20.0)

    def test_versus_benchmark(self, panel):
        result = holdings.characteristics(panel, columns=["per"], benchmark_weight="bench_weight")
        assert result.loc[DATES[0], "per"] == pytest.approx(20.0 - 25.0)

    def test_missing_attribute_is_renormalised(self, panel):
        """属性が欠損した銘柄は除外し、残りのウェイトで割り直す。"""
        holed = panel.copy()
        holed.loc[holed["bid"] == "JP0003", "per"] = np.nan
        result = holdings.characteristics(holed, columns=["per"])
        # (0.4*10 + 0.3*20 + 0.2*30) / 0.9
        assert result.loc[DATES[0], "per"] == pytest.approx(16.0 / 0.9)


class TestContribution:
    def test_by_asset(self, panel):
        result = holdings.contribution(panel, forward_return="ret_1m")
        assert result.loc[DATES[0], "JP0000"] == pytest.approx(0.04)
        assert result.loc[DATES[0], "JP0001"] == pytest.approx(-0.015)

    def test_sums_to_portfolio_return(self, panel):
        result = holdings.contribution(panel, forward_return="ret_1m")
        assert result.iloc[0].sum() == pytest.approx(0.04 - 0.015 + 0.004 + 0.0)

    def test_by_segment(self, panel):
        result = holdings.contribution(panel, forward_return="ret_1m", by="sector")
        assert list(result.columns) == ["A", "B"]
        assert result.loc[DATES[0], "A"] == pytest.approx(0.04 - 0.015)
        assert result.loc[DATES[0], "B"] == pytest.approx(0.004)

    def test_top_contributors(self, panel):
        result = holdings.top_contributors(panel, forward_return="ret_1m", n=1)
        assert result.index[0] == "JP0000", "最大の貢献"
        assert result.loc["JP0000", "contribution"] == pytest.approx(0.04 * 2 + 0.05)
        assert result["side"].iloc[0] == "top"
        assert result["side"].iloc[-1] == "bottom"

    def test_top_contributors_by_segment(self, panel):
        result = holdings.top_contributors(panel, forward_return="ret_1m", by="sector", n=1)
        assert set(result.index) == {"A", "B"}


class TestTurnover:
    def test_one_way(self, panel):
        result = holdings.turnover(panel)
        assert np.isnan(result.iloc[0]), "前期が無い"
        assert result.iloc[1] == pytest.approx(0.0), "変化なし"
        # |0.5-0.4| + |0-0.1| = 0.2 の片道
        assert result.iloc[2] == pytest.approx(0.1)

    def test_two_way(self, panel):
        result = holdings.turnover(panel, method="two_way")
        assert result.iloc[2] == pytest.approx(0.2)

    def test_unknown_method(self, panel):
        with pytest.raises(ValidationError, match="method は"):
            holdings.turnover(panel, method="round_trip")


class TestTrades:
    def test_counts(self, panel):
        result = holdings.trades(panel)
        assert result.iloc[1]["n_new"] == 0
        assert result.iloc[2]["n_closed"] == 1, "JP0003 を売却"
        assert result.iloc[2]["n_new"] == 0

    def test_traded_weight(self, panel):
        result = holdings.trades(panel)
        assert result.iloc[2]["weight_bought"] == pytest.approx(0.1)
        assert result.iloc[2]["weight_sold"] == pytest.approx(0.1)

    def test_first_period_is_nan(self, panel):
        assert holdings.trades(panel).iloc[0].isna().all()


class TestHoldingPeriod:
    def test_average(self, panel):
        result = holdings.average_holding_period(panel)
        assert result["JP0000"] == pytest.approx(3.0), "3 期間持ち続けた"
        assert result["JP0003"] == pytest.approx(2.0), "2 期間で売却"

    def test_counts_separate_spells(self):
        """一度売って買い戻した銘柄は、平均が短くなる。"""
        rows = [
            {"date": DATES[0], "bid": "JP0000", "weight": 1.0},
            {"date": DATES[1], "bid": "JP0000", "weight": 0.0},
            {"date": DATES[2], "bid": "JP0000", "weight": 1.0},
        ]
        result = holdings.average_holding_period(pd.DataFrame(rows))
        assert result["JP0000"] == pytest.approx(1.0), "2 期間 / 2 回"
