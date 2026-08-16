import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from namportfolio import holdings, quantile, signals, viz
from namportfolio.core.errors import ValidationError
from namportfolio.performance import MONTH_END
from namportfolio.viz import theme


def n_series(fig) -> int:
    """描かれた系列の本数。ゼロ線などラベルの無い線は数えない。"""
    return len(fig.axes[0].get_legend_handles_labels()[0])


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def restore_mode():
    saved = theme.get_mode()
    yield
    theme.set_mode(saved)


@pytest.fixture
def returns():
    idx = pd.date_range("2020-01-01", "2023-12-31", freq="B")
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx, name="strategy")


@pytest.fixture
def benchmark(returns):
    rng = np.random.default_rng(1)
    return pd.Series(rng.normal(0.0002, 0.009, len(returns)), index=returns.index, name="benchmark")


class TestCumulativeReturns:
    def test_returns_figure(self, returns):
        assert isinstance(viz.plot_cumulative_returns(returns), Figure)

    def test_benchmark_adds_a_line(self, returns, benchmark):
        assert n_series(viz.plot_cumulative_returns(returns, benchmark)) == 2

    def test_dataframe_input(self, returns, benchmark):
        frame = pd.DataFrame({"a": returns, "b": returns * 2})
        assert n_series(viz.plot_cumulative_returns(frame, benchmark)) == 3

    def test_direct_labels_up_to_four_series(self, returns):
        frame = pd.DataFrame({f"s{i}": returns * (i + 1) for i in range(4)})
        fig = viz.plot_cumulative_returns(frame)
        assert len(fig.axes[0].texts) == 4

    def test_no_direct_labels_past_four_series(self, returns):
        frame = pd.DataFrame({f"s{i}": returns * (i + 1) for i in range(5)})
        fig = viz.plot_cumulative_returns(frame)
        assert len(fig.axes[0].texts) == 0, "重なって読めなくなるので凡例だけにする"

    def test_uses_given_axes(self, returns):
        _, ax = plt.subplots()
        fig = viz.plot_cumulative_returns(returns, ax=ax)
        assert fig is ax.figure

    def test_log_scale(self, returns):
        fig = viz.plot_cumulative_returns(returns, log=True)
        assert fig.axes[0].get_yscale() == "symlog"

    def test_too_many_series_is_rejected(self, returns):
        frame = pd.DataFrame({f"s{i}": returns for i in range(9)})
        with pytest.raises(ValueError, match="最大 8 系列"):
            viz.plot_cumulative_returns(frame)

    def test_invalid_input_type(self):
        with pytest.raises(ValidationError, match="Series か DataFrame"):
            viz.plot_cumulative_returns([0.1, 0.2])


class TestDrawdown:
    def test_single_series_is_filled(self, returns):
        fig = viz.plot_drawdown(returns)
        assert len(fig.axes[0].collections) == 1

    def test_multiple_series_are_not_filled(self, returns, benchmark):
        fig = viz.plot_drawdown(returns, benchmark)
        assert len(fig.axes[0].collections) == 0, "塗ると下の線が埋もれる"
        assert n_series(fig) == 2


class TestMonthlyHeatmap:
    def test_returns_figure_with_colorbar(self, returns):
        fig = viz.plot_monthly_heatmap(returns)
        assert len(fig.axes) == 2, "本体とカラーバー"

    def test_annotates_by_default_for_short_history(self, returns):
        fig = viz.plot_monthly_heatmap(returns)
        assert len(fig.axes[0].texts) == 48, "4 年 × 12 か月"

    def test_annotation_can_be_disabled(self, returns):
        fig = viz.plot_monthly_heatmap(returns, annotate=False)
        assert len(fig.axes[0].texts) == 0

    def test_rejects_frame(self, returns):
        with pytest.raises(ValidationError, match="Series のみ"):
            viz.plot_monthly_heatmap(returns.to_frame())


class TestAnnualReturns:
    def test_bar_count(self, returns, benchmark):
        fig = viz.plot_annual_returns(returns, benchmark)
        assert len(fig.axes[0].containers) == 2, "系列ごとに 1 グループ"
        assert len(fig.axes[0].containers[0]) == 4, "2020〜2023"


class TestRolling:
    def test_sharpe(self, returns):
        assert n_series(viz.plot_rolling(returns, window=60, metric="sharpe")) == 1

    def test_benchmark_metrics(self, returns, benchmark):
        for metric in ("beta", "tracking_error", "information_ratio"):
            fig = viz.plot_rolling(returns, benchmark, window=60, metric=metric)
            assert n_series(fig) == 1

    def test_benchmark_required(self, returns):
        with pytest.raises(ValidationError, match="benchmark が必要"):
            viz.plot_rolling(returns, window=60, metric="beta")

    def test_unknown_metric(self, returns):
        with pytest.raises(ValidationError, match="metric は"):
            viz.plot_rolling(returns, window=60, metric="omega")

    def test_dataframe_input(self, returns, benchmark):
        frame = pd.DataFrame({"a": returns, "b": returns * 2})
        fig = viz.plot_rolling(frame, benchmark, window=60, metric="beta")
        assert n_series(fig) == 2


class TestReturnDistribution:
    def test_draws_mean_and_var_lines(self, returns):
        fig = viz.plot_return_distribution(returns)
        assert len(fig.axes[0].lines) == 2
        assert len(fig.axes[0].texts) == 2

    def test_rejects_frame(self, returns):
        with pytest.raises(ValidationError, match="Series のみ"):
            viz.plot_return_distribution(returns.to_frame())


class TestQuantileCharts:
    @pytest.fixture
    def panel(self):
        dates = pd.date_range("2020-01-31", periods=48, freq=MONTH_END)
        rng = np.random.default_rng(3)
        rows = [
            {
                "date": date,
                "bid": f"JP{i:04d}",
                "factor": float(i) + rng.normal(0, 2),
                "fwd_ret": i * 0.002 + rng.normal(0, 0.03),
            }
            for date in dates
            for i in range(30)
        ]
        return pd.DataFrame(rows)

    @pytest.fixture
    def quantile_rets(self, panel):
        return quantile.quantile_returns(panel, factor="factor", forward_return="fwd_ret")

    @pytest.fixture
    def ic(self, panel):
        return quantile.information_coefficient(panel, factor="factor", forward_return="fwd_ret")

    def test_quantile_bar(self, quantile_rets):
        fig = viz.plot_quantile_returns(quantile_rets)
        assert len(fig.axes[0].containers[0]) == 5

    def test_quantile_bar_uses_ordinal_colors(self, quantile_rets):
        fig = viz.plot_quantile_returns(quantile_rets)
        bars = fig.axes[0].containers[0]
        colors = [b.get_facecolor() for b in bars]
        assert len(set(colors)) == 5, "分位ごとに濃さが変わる"

    def test_cumulative_adds_long_short(self, quantile_rets):
        assert n_series(viz.plot_quantile_cumulative(quantile_rets)) == 6
        assert n_series(viz.plot_quantile_cumulative(quantile_rets, long_short=False)) == 5

    def test_rejects_series(self, quantile_rets):
        with pytest.raises(ValidationError, match="DataFrame"):
            viz.plot_quantile_returns(quantile_rets["Q1"])

    def test_ic_line_with_rolling_mean(self, ic):
        fig = viz.plot_ic(ic, window=12)
        assert n_series(fig) == 1, "移動平均だけラベルを持つ"
        assert len(fig.axes[0].lines) == 4, "生 IC・移動平均・平均線・ゼロ線"

    def test_ic_skips_rolling_when_too_short(self, ic):
        fig = viz.plot_ic(ic.iloc[:5], window=12)
        assert n_series(fig) == 0

    def test_ic_rejects_frame(self, ic):
        with pytest.raises(ValidationError, match="Series のみ"):
            viz.plot_ic(ic.to_frame())

    def test_ic_heatmap(self, ic):
        fig = viz.plot_ic_heatmap(ic)
        assert len(fig.axes) == 2
        assert len(fig.axes[0].texts) == 48

    def test_decay(self, panel):
        panel = panel.assign(fwd_ret_2=lambda d: d["fwd_ret"] * 0.5)
        decay = quantile.factor_decay(
            panel, factor="factor", forward_returns=["fwd_ret", "fwd_ret_2"]
        )
        assert isinstance(viz.plot_factor_decay(decay), Figure)

    def test_decay_unknown_metric(self, panel):
        decay = quantile.factor_decay(panel, factor="factor", forward_returns=["fwd_ret"])
        with pytest.raises(ValidationError, match="列がありません"):
            viz.plot_factor_decay(decay, metric="omega")

    def test_turnover(self, panel):
        turnover = quantile.quantile_turnover(panel, factor="factor")
        assert n_series(viz.plot_quantile_turnover(turnover)) == 5

    def test_transition_matrix(self, panel):
        matrix = quantile.quantile_transition_matrix(panel, factor="factor")
        fig = viz.plot_transition_matrix(matrix)
        assert len(fig.axes[0].texts) == 25


class TestClassAndMissingCharts:
    @pytest.fixture
    def panel(self):
        dates = pd.date_range("2022-01-31", periods=24, freq=MONTH_END)
        rng = np.random.default_rng(9)
        rows = [
            {
                "date": date,
                "bid": f"JP{i:04d}",
                "factor": np.nan if i < 4 else float(i) + rng.normal(0, 2),
                "fwd_ret": i * 0.002 + rng.normal(0, 0.03),
                "rating": "A" if i < 10 else "B" if i < 20 else "C",
            }
            for date in dates
            for i in range(30)
        ]
        return pd.DataFrame(rows)

    @pytest.fixture
    def with_missing(self, panel):
        return quantile.quantile_returns(
            panel, factor="factor", forward_return="fwd_ret", include_missing=True
        )

    def test_missing_class_is_grey(self, with_missing):
        """欠損クラスだけ中立色。順序の濃淡に混ぜない。"""
        fig = viz.plot_quantile_returns(with_missing)
        colors = [bar.get_facecolor() for bar in fig.axes[0].containers[0]]
        muted = to_rgba(theme.palette()["muted"])
        assert colors[-1] == muted
        assert muted not in colors[:-1]

    def test_long_short_ignores_missing_class(self, with_missing):
        fig = viz.plot_quantile_cumulative(with_missing)
        labels = fig.axes[0].get_legend_handles_labels()[1]
        assert "Q5-Q1" in labels, "NA を端に選ばない"

    def test_class_returns_with_categorical_palette(self, panel):
        rets = quantile.class_returns(panel, classes="rating", forward_return="fwd_ret")
        fig = viz.plot_quantile_returns(rets, palette="categorical")
        colors = [bar.get_facecolor() for bar in fig.axes[0].containers[0]]
        assert colors[0] == to_rgba(theme.categorical_colors(1)[0])

    def test_unknown_palette(self, with_missing):
        with pytest.raises(ValidationError, match="palette は"):
            viz.plot_quantile_returns(with_missing, palette="rainbow")

    def test_turnover_with_missing_class(self, panel):
        turnover = quantile.quantile_turnover(panel, factor="factor", include_missing=True)
        assert n_series(viz.plot_quantile_turnover(turnover)) == 6


class TestSignalCharts:
    @pytest.fixture
    def panel(self):
        dates = pd.date_range("2022-01-31", periods=24, freq=MONTH_END)
        rng = np.random.default_rng(5)
        rows = [
            {
                "date": date,
                "bid": f"JP{i:04d}",
                "value": rng.normal(0, 1),
                "other": rng.normal(0, 1),
            }
            for date in dates
            for i in range(40)
        ]
        frame = pd.DataFrame(rows)
        return frame.assign(value_z=signals.standardize(frame, factor="value"))

    def test_coverage(self, panel):
        cov = signals.coverage(panel, factor="value")
        assert isinstance(viz.plot_coverage(cov), Figure)

    def test_coverage_rate_axis_is_capped(self, panel):
        cov = signals.coverage(panel, factor="value")
        fig = viz.plot_coverage(cov, metric="coverage_rate")
        assert fig.axes[0].get_ylim()[1] >= 1.0

    def test_coverage_unknown_metric(self, panel):
        cov = signals.coverage(panel, factor="value")
        with pytest.raises(ValidationError, match="必須カラム"):
            viz.plot_coverage(cov, metric="omega")

    def test_distribution_single(self, panel):
        fig = viz.plot_distribution(panel, factor="value")
        assert n_series(fig) == 0, "1 系列なら凡例は不要"

    def test_distribution_comparison(self, panel):
        fig = viz.plot_distribution(panel, factor="value", compare="value_z")
        assert n_series(fig) == 2

    def test_distribution_stats(self, panel):
        summary = signals.distribution_summary(panel, factor="value")
        assert isinstance(viz.plot_distribution_stats(summary, metric="skew"), Figure)

    def test_correlation_heatmap(self, panel):
        corr = signals.signal_correlation(panel, factors=["value", "other", "value_z"])
        fig = viz.plot_signal_correlation(corr)
        assert len(fig.axes[0].texts) == 9

    def test_correlation_requires_square(self, panel):
        with pytest.raises(ValidationError, match="正方行列"):
            viz.plot_signal_correlation(pd.DataFrame(np.zeros((2, 3))))


class TestHoldingCharts:
    @pytest.fixture
    def panel(self):
        dates = pd.date_range("2023-01-31", periods=12, freq=MONTH_END)
        rng = np.random.default_rng(13)
        rows = []
        for date in dates:
            raw = rng.random(20)
            weights = raw / raw.sum()
            for i, w in enumerate(weights):
                rows.append(
                    {
                        "date": date,
                        "bid": f"JP{i:04d}",
                        "weight": w,
                        "bench_weight": 0.05,
                        "ret_1m": rng.normal(0, 0.04),
                        "sector": ["A", "B", "C"][i % 3],
                        "per": rng.uniform(8, 30),
                    }
                )
        return pd.DataFrame(rows)

    def test_allocation_stacked(self, panel):
        alloc = holdings.allocation(panel, by="sector")
        fig = viz.plot_allocation(alloc)
        assert len(fig.axes[0].collections) == 3, "積み上げ面（セグメントごとに 1 面）"
        assert len(fig.axes[0].lines) == 0, "線ではない"

    def test_active_allocation_falls_back_to_lines(self, panel):
        active = holdings.allocation(panel, by="sector", benchmark_weight="bench_weight")
        fig = viz.plot_allocation(active)
        assert n_series(fig) == 3, "負の値があるので線になる"

    def test_stacked_rejected_when_negative(self, panel):
        active = holdings.allocation(panel, by="sector", benchmark_weight="bench_weight")
        with pytest.raises(ValidationError, match="積み上げ面にできません"):
            viz.plot_allocation(active, stacked=True)

    def test_concentration(self, panel):
        conc = holdings.concentration(panel)
        assert isinstance(viz.plot_concentration(conc), Figure)

    def test_contribution_uses_polarity_colors(self, panel):
        top = holdings.top_contributors(panel, forward_return="ret_1m", n=3)
        fig = viz.plot_contribution(top)
        colors = [bar.get_facecolor() for bar in fig.axes[0].containers[0]]
        assert len(set(colors)) == 2, "正と負の 2 色だけ"

    def test_characteristics(self, panel):
        chars = holdings.characteristics(panel, columns=["per"])
        assert isinstance(viz.plot_characteristics(chars, metric="per"), Figure)

    def test_characteristics_unknown_metric(self, panel):
        chars = holdings.characteristics(panel, columns=["per"])
        with pytest.raises(ValidationError, match="必須カラム"):
            viz.plot_characteristics(chars, metric="pbr")

    def test_turnover(self, panel):
        series = holdings.turnover(panel)
        fig = viz.plot_turnover(series)
        assert len(fig.axes[0].texts) == 1, "平均値のラベル"

    def test_turnover_rejects_frame(self, panel):
        with pytest.raises(ValidationError, match="Series のみ"):
            viz.plot_turnover(holdings.turnover(panel).to_frame())


class TestTheme:
    def test_categorical_is_fixed_order(self):
        assert theme.categorical_colors(3) == theme.categorical_colors(8)[:3], (
            "系列が減っても残った系列の色は変わらない"
        )

    def test_categorical_caps_at_eight(self):
        with pytest.raises(ValueError, match="最大 8 系列"):
            theme.categorical_colors(9)

    def test_ordinal_is_monotone_and_sized(self):
        for n in (3, 5, 10, 20):
            colors = theme.ordinal_colors(n)
            assert len(colors) == n
            assert len(set(colors)) == n, "同じ色が重複しない"

    def test_mode_switch_changes_surface(self):
        light = theme.palette("light")["surface"]
        dark = theme.palette("dark")["surface"]
        assert light != dark

    def test_mode_validation(self):
        with pytest.raises(ValueError, match="mode は"):
            theme.set_mode("sepia")

    def test_dark_mode_applies_to_plots(self, returns):
        theme.set_mode("dark")
        fig = viz.plot_cumulative_returns(returns)
        assert fig.get_facecolor()[:3] != (1.0, 1.0, 1.0)

    def test_diverging_midpoint_is_neutral(self):
        cmap = theme.diverging_cmap("light")
        r, g, b, _ = cmap(0.5)
        assert abs(r - g) < 0.1 and abs(g - b) < 0.1, "中点はグレー（ゼロに見える必要がある）"

    def test_styled_does_not_leak(self, returns):
        before = plt.rcParams["lines.linewidth"]
        viz.plot_cumulative_returns(returns)
        assert plt.rcParams["lines.linewidth"] == before
