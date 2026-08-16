import numpy as np
import pandas as pd
import pytest

from namportfolio import performance as perf
from namportfolio.core.errors import ValidationError


@pytest.fixture
def simple():
    """手計算しやすい 6 期間の系列。"""
    return pd.Series(
        [0.1, -0.2, 0.05, 0.2, -0.1, 0.15],
        index=pd.date_range("2024-01-01", periods=6, freq="B"),
    )


@pytest.fixture
def yearly():
    """年次 2 期間。年率化の検算に使う。"""
    return pd.Series([0.1, 0.1], index=pd.to_datetime(["2023-12-31", "2024-12-31"]))


class TestBasics:
    def test_cumulative_returns(self):
        r = pd.Series([0.1, -0.1], index=pd.date_range("2024-01-01", periods=2, freq="B"))
        # 1.1 * 0.9 = 0.99
        assert list(perf.cumulative_returns(r)) == pytest.approx([0.1, -0.01])

    def test_cumulative_returns_treats_nan_as_zero(self):
        r = pd.Series([0.1, np.nan, 0.1], index=pd.date_range("2024-01-01", periods=3, freq="B"))
        out = perf.cumulative_returns(r)
        assert out.notna().all(), "欠損があっても以降が NaN にならない"
        assert out.iloc[-1] == pytest.approx(1.1 * 1.1 - 1.0)

    def test_cumulative_returns_log(self):
        r = pd.Series([0.1, 0.1], index=pd.date_range("2024-01-01", periods=2, freq="B"))
        assert perf.cumulative_returns(r, log=True).iloc[-1] == pytest.approx(np.log(1.21))

    def test_total_return(self, simple):
        expected = np.prod(1.0 + simple.to_numpy()) - 1.0
        assert perf.total_return(simple) == pytest.approx(expected)

    def test_annualized_return_geometric(self, yearly):
        assert perf.annualized_return(yearly, periods_per_year=1.0) == pytest.approx(0.1)

    def test_annualized_return_uses_valid_count(self):
        """欠損期間は分母に数えない。"""
        r = pd.Series(
            [0.1, np.nan, 0.1], index=pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"])
        )
        assert perf.annualized_return(r, periods_per_year=1.0) == pytest.approx(0.1)

    def test_annualized_volatility(self, simple):
        expected = simple.std(ddof=1) * np.sqrt(252.0)
        assert perf.annualized_volatility(simple) == pytest.approx(expected)

    def test_frequency_inferred_from_index(self, simple):
        """営業日次なので 252 で年率化される。"""
        assert perf.annualized_volatility(simple) == pytest.approx(
            perf.annualized_volatility(simple, periods_per_year=252.0)
        )


class TestRatios:
    def test_sharpe_ratio(self, simple):
        expected = simple.mean() / simple.std(ddof=1) * np.sqrt(252.0)
        assert perf.sharpe_ratio(simple) == pytest.approx(expected)

    def test_sharpe_with_risk_free_is_lower(self, simple):
        assert perf.sharpe_ratio(simple, risk_free=0.05) < perf.sharpe_ratio(simple)

    def test_risk_free_converted_geometrically(self):
        """年率 rf は単純除算ではなく幾何的に期間率へ変換される。"""
        idx = pd.date_range("2020-01-31", periods=24, freq=perf.MONTH_END)
        r = pd.Series(np.random.default_rng(0).normal(0.01, 0.02, 24), index=idx)

        geometric = 1.12 ** (1 / 12) - 1  # 0.009489...
        excess = r - geometric
        expected = excess.mean() / excess.std(ddof=1) * np.sqrt(12.0)

        assert perf.sharpe_ratio(r, risk_free=0.12, periods_per_year=12.0) == pytest.approx(
            expected
        )
        naive = r - 0.12 / 12
        assert expected != pytest.approx(naive.mean() / naive.std(ddof=1) * np.sqrt(12.0)), (
            "単純除算 (rf/P) とは異なる値になる"
        )

    def test_zero_volatility_gives_nan(self):
        """超過リターンが定数なら分母 0 で NaN（0 除算で inf を返さない）。"""
        r = pd.Series([0.1, 0.1], index=pd.to_datetime(["2023-12-31", "2024-12-31"]))
        assert np.isnan(perf.sharpe_ratio(r, risk_free=0.1, periods_per_year=1.0))

    def test_sortino_only_penalizes_downside(self):
        upside = pd.Series([0.1, 0.1, 0.1], index=pd.date_range("2024-01-01", periods=3, freq="B"))
        assert np.isnan(perf.sortino_ratio(upside)), "下振れが無ければ分母 0 で NaN"

    def test_sortino_manual(self):
        r = pd.Series([0.1, -0.1], index=pd.date_range("2024-01-01", periods=2, freq="B"))
        # downside = [0, -0.1] -> rms = sqrt(0.01/2) = 0.0707...
        expected = r.mean() / np.sqrt(0.01 / 2) * np.sqrt(252.0)
        assert perf.sortino_ratio(r) == pytest.approx(expected)

    def test_calmar_ratio(self, simple):
        expected = perf.annualized_return(simple) / abs(perf.max_drawdown(simple))
        assert perf.calmar_ratio(simple) == pytest.approx(expected)


class TestDrawdown:
    def test_drawdown_series(self, simple):
        dd = perf.drawdown(simple)
        # wealth: 1.1, 0.88, 0.924, 1.1088, 0.99792, 1.147608
        assert dd.iloc[0] == pytest.approx(0.0)
        assert dd.iloc[1] == pytest.approx(-0.2)
        assert dd.iloc[2] == pytest.approx(-0.16)
        assert dd.iloc[3] == pytest.approx(0.0)

    def test_max_drawdown(self, simple):
        assert perf.max_drawdown(simple) == pytest.approx(-0.2)

    def test_drawdown_table_periods(self, simple):
        table = perf.drawdown_table(simple)
        assert len(table) == 2, "水面下の区間は 2 つ"

        worst = table.iloc[0]
        assert worst["max_drawdown"] == pytest.approx(-0.2)
        assert worst["peak"] == simple.index[0]
        assert worst["valley"] == simple.index[1]
        assert worst["recovery"] == simple.index[3]
        assert worst["length"] == 3
        assert worst["recovery_length"] == 2

    def test_drawdown_table_unrecovered(self):
        r = pd.Series([0.1, -0.2], index=pd.date_range("2024-01-01", periods=2, freq="B"))
        row = perf.drawdown_table(r).iloc[0]
        assert pd.isna(row["recovery"])
        assert pd.isna(row["recovery_length"])
        assert row["peak"] == r.index[0]

    def test_drawdown_table_sorted_and_limited(self):
        r = pd.Series(
            [-0.01, 0.02, -0.05, 0.06, -0.03, 0.04, -0.02, 0.03],
            index=pd.date_range("2024-01-01", periods=8, freq="B"),
        )
        table = perf.drawdown_table(r, top=2)
        assert len(table) == 2
        assert table["max_drawdown"].is_monotonic_increasing, "深い順（昇順）"

    def test_drawdown_table_no_drawdown(self):
        r = pd.Series([0.1, 0.1], index=pd.date_range("2024-01-01", periods=2, freq="B"))
        table = perf.drawdown_table(r)
        assert table.empty
        assert "max_drawdown" in table.columns

    def test_drawdown_table_rejects_frame(self, simple):
        with pytest.raises(ValidationError, match="Series のみ"):
            perf.drawdown_table(simple.to_frame())


class TestDistribution:
    def test_var_and_cvar(self):
        r = pd.Series(
            np.arange(-0.10, 0.10, 0.01), index=pd.date_range("2024-01-01", periods=20, freq="B")
        )
        var = perf.value_at_risk(r, level=0.25)
        assert var == pytest.approx(r.quantile(0.25))
        assert perf.conditional_value_at_risk(r, level=0.25) <= var, "CVaR は VaR 以下"

    def test_hit_rate(self, simple):
        assert perf.hit_rate(simple) == pytest.approx(4 / 6)

    def test_win_loss_ratio(self, simple):
        wins = simple[simple > 0].mean()
        losses = abs(simple[simple < 0].mean())
        assert perf.win_loss_ratio(simple) == pytest.approx(wins / losses)


class TestBenchmarkRelative:
    @pytest.fixture
    def bench(self, simple):
        return simple * 0.5

    def test_active_returns(self, simple, bench):
        assert perf.active_returns(simple, bench).iloc[0] == pytest.approx(0.05)

    def test_beta_of_scaled_series(self, simple, bench):
        assert perf.beta(simple, bench) == pytest.approx(2.0)

    def test_beta_of_self_is_one(self, simple):
        assert perf.beta(simple, simple) == pytest.approx(1.0)

    def test_alpha_of_self_is_zero(self, simple):
        assert perf.alpha(simple, simple) == pytest.approx(0.0, abs=1e-12)

    def test_tracking_error_of_self_is_zero(self, simple):
        assert perf.tracking_error(simple, simple) == pytest.approx(0.0)

    def test_information_ratio(self, simple, bench):
        active = simple - bench
        expected = active.mean() * 252.0 / (active.std(ddof=1) * np.sqrt(252.0))
        assert perf.information_ratio(simple, bench) == pytest.approx(expected)

    def test_capture_of_self_is_one(self, simple):
        assert perf.capture_ratio(simple, simple, side="up") == pytest.approx(1.0)
        assert perf.capture_ratio(simple, simple, side="down") == pytest.approx(1.0)

    def test_capture_side_validation(self, simple):
        with pytest.raises(ValidationError, match="side"):
            perf.capture_ratio(simple, simple, side="sideways")

    def test_alignment_on_mismatched_dates(self, simple):
        shifted = simple.iloc[2:]
        assert len(perf.active_returns(simple, shifted)) == 4

    def test_benchmark_type_validation(self, simple):
        with pytest.raises(ValidationError, match="benchmark"):
            perf.active_returns(simple, [0.1, 0.2])


class TestAggregation:
    @pytest.fixture
    def daily_year(self):
        idx = pd.date_range("2023-01-02", "2024-12-31", freq="B")
        rng = np.random.default_rng(0)
        return pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)

    def test_monthly_matches_compounding(self, daily_year):
        monthly = perf.aggregate_returns(daily_year, perf.MONTH_END)
        assert len(monthly) == 24
        jan = daily_year["2023-01"]
        assert monthly.iloc[0] == pytest.approx(np.prod(1.0 + jan.to_numpy()) - 1.0)

    def test_yearly_matches_total(self, daily_year):
        yearly = perf.aggregate_returns(daily_year, perf.YEAR_END)
        assert len(yearly) == 2
        assert (1.0 + yearly).prod() - 1.0 == pytest.approx(perf.total_return(daily_year))

    def test_monthly_table_shape(self, daily_year):
        table = perf.monthly_table(daily_year)
        assert list(table.index) == [2023, 2024]
        assert "01" in table.columns and "year_total" in table.columns
        assert table.loc[2023, "year_total"] == pytest.approx(perf.total_return(daily_year["2023"]))

    def test_monthly_table_rejects_frame(self, daily_year):
        with pytest.raises(ValidationError, match="Series のみ"):
            perf.monthly_table(daily_year.to_frame())


class TestRolling:
    @pytest.fixture
    def daily(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="B")
        rng = np.random.default_rng(1)
        return pd.Series(rng.normal(0.0005, 0.01, 100), index=idx)

    def test_rolling_volatility(self, daily):
        out = perf.rolling_volatility(daily, 20)
        assert out.iloc[:19].isna().all()
        assert out.iloc[19] == pytest.approx(daily.iloc[:20].std() * np.sqrt(252.0))

    def test_rolling_sharpe(self, daily):
        out = perf.rolling_sharpe(daily, 20)
        window = daily.iloc[:20]
        assert out.iloc[19] == pytest.approx(window.mean() / window.std() * np.sqrt(252.0))

    def test_rolling_beta_of_self_is_one(self, daily):
        out = perf.rolling_beta(daily, daily, 20)
        assert out.iloc[19] == pytest.approx(1.0)

    def test_rolling_tracking_error_of_self_is_zero(self, daily):
        assert perf.rolling_tracking_error(daily, daily, 20).iloc[19] == pytest.approx(0.0)

    def test_rolling_information_ratio_runs(self, daily):
        out = perf.rolling_information_ratio(daily, daily * 0.5, 20)
        assert out.notna().sum() == 81


class TestSummary:
    def test_absolute_only(self, simple):
        s = perf.performance_summary(simple)
        assert isinstance(s, pd.Series)
        assert s["max_drawdown"] == pytest.approx(-0.2)
        assert s["n_periods"] == 6
        assert "beta" not in s.index

    def test_with_benchmark_adds_relative(self, simple):
        s = perf.performance_summary(simple, simple * 0.5)
        for key in ("beta", "alpha", "tracking_error", "information_ratio", "up_capture"):
            assert key in s.index
        assert s["beta"] == pytest.approx(2.0)

    def test_var_level_in_key_name(self, simple):
        s = perf.performance_summary(simple, var_level=0.01)
        assert "var_0.01" in s.index

    def test_dataframe_input_returns_frame(self, simple):
        df = pd.DataFrame({"strategy_a": simple, "strategy_b": simple * 2})
        out = perf.performance_summary(df)
        assert list(out.columns) == ["strategy_a", "strategy_b"]
        assert out.loc["max_drawdown", "strategy_a"] == pytest.approx(-0.2)
