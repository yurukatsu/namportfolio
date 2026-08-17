import numpy as np
import pandas as pd
import pytest

from namportfolio import stats
from namportfolio.core.errors import ValidationError
from namportfolio.performance import MONTH_END


@pytest.fixture
def monthly():
    """月次 10 年。平均 0.5%、月次ボラ 3%。"""
    idx = pd.date_range("2015-01-31", periods=120, freq=MONTH_END)
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.005, 0.03, 120), index=idx)


class TestTStatistic:
    def test_manual(self):
        values = [1.0, 2.0, 3.0, 4.0]
        expected = np.mean(values) / (np.std(values, ddof=1) / np.sqrt(4))
        assert stats.t_statistic(values) == pytest.approx(expected)

    def test_against_hypothesised_mean(self):
        assert stats.t_statistic([1.0, 2.0, 3.0], mu=2.0) == pytest.approx(0.0)

    def test_ignores_nan(self):
        assert stats.t_statistic([1.0, 2.0, np.nan, 3.0]) == pytest.approx(
            stats.t_statistic([1.0, 2.0, 3.0])
        )

    def test_too_few_points(self):
        assert np.isnan(stats.t_statistic([1.0]))

    def test_zero_variance(self):
        assert np.isnan(stats.t_statistic([2.0, 2.0, 2.0]))


class TestNeweyWest:
    def test_zero_lags_matches_plain_t(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.5, 1.0, 200)
        assert stats.newey_west_tstat(values, lags=0) == pytest.approx(
            stats.t_statistic(values), rel=0.01
        )

    def test_positive_autocorrelation_lowers_t(self):
        """自己相関があると素の t 値は有意性を過大評価する。"""
        rng = np.random.default_rng(1)
        noise = rng.normal(0.0, 1.0, 400)
        ar = np.zeros(400)
        for i in range(1, 400):
            ar[i] = 0.7 * ar[i - 1] + noise[i]
        ar += 0.5
        assert abs(stats.newey_west_tstat(ar)) < abs(stats.t_statistic(ar))

    def test_lag_rule(self):
        assert stats.newey_west_lags(100) == 4
        assert stats.newey_west_lags(1) == 0

    def test_lags_capped_at_sample_size(self):
        assert not np.isnan(stats.newey_west_tstat([1.0, 2.0, 3.0], lags=50))


class TestBootstrap:
    def test_distribution_size_and_seed(self, monthly):
        first = stats.bootstrap_distribution(monthly, n_boot=500, seed=1)
        second = stats.bootstrap_distribution(monthly, n_boot=500, seed=1)
        assert len(first) == 500
        assert np.array_equal(first, second), "seed が同じなら再現する"

    def test_distribution_centres_on_observed(self, monthly):
        distribution = stats.bootstrap_distribution(monthly, n_boot=2000)
        assert distribution.mean() == pytest.approx(monthly.mean(), abs=0.002)

    def test_ci_contains_observed(self, monthly):
        lower, upper = stats.bootstrap_ci(monthly, n_boot=2000)
        assert lower < monthly.mean() < upper

    def test_ci_narrows_with_confidence(self, monthly):
        wide = stats.bootstrap_ci(monthly, confidence=0.99, n_boot=2000)
        narrow = stats.bootstrap_ci(monthly, confidence=0.80, n_boot=2000)
        assert wide[0] < narrow[0] and narrow[1] < wide[1]

    def test_block_length_preserves_autocorrelation(self):
        """自己相関のある系列では、ブロックを長くすると区間が広がる。"""
        rng = np.random.default_rng(2)
        noise = rng.normal(0, 1, 300)
        ar = np.zeros(300)
        for i in range(1, 300):
            ar[i] = 0.8 * ar[i - 1] + noise[i]
        series = pd.Series(ar + 0.3, index=pd.date_range("2000-01-31", periods=300, freq=MONTH_END))

        short = stats.bootstrap_ci(series, block_length=1, n_boot=2000)
        long = stats.bootstrap_ci(series, block_length=20, n_boot=2000)
        assert (long[1] - long[0]) > (short[1] - short[0])

    def test_sharpe_statistic_is_annualised(self, monthly):
        distribution = stats.bootstrap_distribution(monthly, statistic="sharpe_ratio", n_boot=1000)
        observed = monthly.mean() / monthly.std(ddof=1) * np.sqrt(12)
        assert np.median(distribution) == pytest.approx(observed, abs=0.3)

    def test_pvalue_small_for_strong_signal(self):
        idx = pd.date_range("2015-01-31", periods=120, freq=MONTH_END)
        strong = pd.Series(np.random.default_rng(3).normal(0.02, 0.01, 120), index=idx)
        assert stats.bootstrap_pvalue(strong, n_boot=2000) < 0.05

    def test_pvalue_large_for_noise(self):
        idx = pd.date_range("2015-01-31", periods=120, freq=MONTH_END)
        noise = pd.Series(np.random.default_rng(4).normal(0.0, 0.03, 120), index=idx)
        assert stats.bootstrap_pvalue(noise, n_boot=2000) > 0.10

    def test_unknown_statistic(self, monthly):
        with pytest.raises(ValidationError, match="statistic は"):
            stats.bootstrap_distribution(monthly, statistic="skew")

    def test_invalid_confidence(self, monthly):
        with pytest.raises(ValidationError, match="confidence"):
            stats.bootstrap_ci(monthly, confidence=1.5)


class TestSubsample:
    def test_split_shape(self, monthly):
        table = stats.subsample(monthly, n_splits=4)
        assert len(table) == 4
        assert list(table.columns) == ["start", "end", "n_periods", "mean", "t_stat"]
        assert table["n_periods"].sum() == len(monthly)

    def test_splits_cover_period_in_order(self, monthly):
        table = stats.subsample(monthly, n_splits=3)
        assert table.loc[1, "start"] == monthly.index[0]
        assert table.loc[3, "end"] == monthly.index[-1]
        assert table.loc[1, "end"] < table.loc[2, "start"]

    def test_detects_regime_break(self):
        """後半だけ稼いだ系列は、前半と後半で符号が変わる。"""
        idx = pd.date_range("2015-01-31", periods=100, freq=MONTH_END)
        values = pd.Series(np.r_[np.full(50, -0.01), np.full(50, 0.02)], index=idx)
        table = stats.subsample(values, n_splits=2)
        assert table.loc[1, "mean"] < 0 < table.loc[2, "mean"]

    def test_statistic_choice(self, monthly):
        table = stats.subsample(monthly, n_splits=2, statistic="sharpe_ratio")
        assert "sharpe_ratio" in table.columns

    def test_validation(self, monthly):
        with pytest.raises(ValidationError, match="n_splits"):
            stats.subsample(monthly, n_splits=1)
        with pytest.raises(ValidationError, match="statistic は"):
            stats.subsample(monthly, statistic="omega")


class TestStability:
    def test_consistent_series_has_full_agreement(self):
        idx = pd.date_range("2015-01-31", periods=100, freq=MONTH_END)
        steady = pd.Series(np.full(100, 0.01), index=idx)
        result = stats.stability(steady, n_splits=4)
        assert result["sign_agreement"] == pytest.approx(1.0)
        assert result["positive_ratio"] == pytest.approx(1.0)

    def test_broken_series_has_partial_agreement(self):
        idx = pd.date_range("2015-01-31", periods=100, freq=MONTH_END)
        values = pd.Series(np.r_[np.full(50, -0.01), np.full(50, 0.03)], index=idx)
        result = stats.stability(values, n_splits=2)
        assert result["sign_agreement"] == pytest.approx(0.5)

    def test_keys(self, monthly):
        result = stats.stability(monthly)
        for key in ("sign_agreement", "positive_ratio", "dispersion", "min", "max"):
            assert key in result.index


class TestRegimes:
    @pytest.fixture
    def benchmark(self):
        idx = pd.date_range("2015-01-31", periods=120, freq=MONTH_END)
        return pd.Series(np.random.default_rng(5).normal(0.004, 0.04, 120), index=idx)

    def test_direction_labels(self, benchmark):
        regimes = stats.make_regimes(benchmark, method="direction")
        assert set(regimes.dropna().unique()) == {"up", "down"}
        assert (regimes[benchmark > 0] == "up").all()

    def test_volatility_labels(self, benchmark):
        regimes = stats.make_regimes(benchmark, method="volatility", window=12)
        assert set(regimes.dropna().unique()) == {"low", "high"}
        assert regimes.isna().sum() == 11, "窓が埋まるまでは判定できない"

    def test_quantile_labels(self, benchmark):
        regimes = stats.make_regimes(benchmark, method="quantile", n_quantiles=3)
        assert set(regimes.dropna().unique()) == {"low", "mid", "high"}

    def test_custom_labels(self, benchmark):
        regimes = stats.make_regimes(
            benchmark, method="quantile", n_quantiles=2, labels=["bear", "bull"]
        )
        assert set(regimes.dropna().unique()) == {"bear", "bull"}

    def test_label_count_validation(self, benchmark):
        with pytest.raises(ValidationError, match="labels は"):
            stats.make_regimes(benchmark, method="quantile", n_quantiles=3, labels=["a", "b"])

    def test_unknown_method(self, benchmark):
        with pytest.raises(ValidationError, match="method は"):
            stats.make_regimes(benchmark, method="sentiment")

    def test_summary_splits_by_regime(self, monthly, benchmark):
        regimes = stats.make_regimes(benchmark, method="direction")
        summary = stats.regime_summary(monthly, regimes)
        assert set(summary.index) == {"up", "down"}
        assert summary["n_periods"].sum() == len(monthly)
        assert summary["share"].sum() == pytest.approx(1.0)

    def test_summary_detects_regime_dependence(self, benchmark):
        """ベンチが上昇した月だけ稼ぐ系列は、up の平均だけが正になる。"""
        values = pd.Series(np.where(benchmark > 0, 0.02, -0.01), index=benchmark.index)
        summary = stats.regime_summary(values, stats.make_regimes(benchmark))
        assert summary.loc["up", "mean"] > 0 > summary.loc["down", "mean"]

    def test_summary_order(self, monthly, benchmark):
        regimes = stats.make_regimes(benchmark, method="direction")
        summary = stats.regime_summary(monthly, regimes, regime_order=["up", "down"])
        assert list(summary.index) == ["up", "down"]

    def test_summary_requires_overlap(self, monthly):
        other = pd.Series(["up"] * 5, index=pd.date_range("1990-01-31", periods=5, freq=MONTH_END))
        with pytest.raises(ValidationError, match="重なりがありません"):
            stats.regime_summary(monthly, other)


class TestProbabilisticSharpe:
    def test_zero_threshold_is_high_for_good_track_record(self):
        idx = pd.date_range("2000-01-31", periods=240, freq=MONTH_END)
        good = pd.Series(np.random.default_rng(6).normal(0.01, 0.02, 240), index=idx)
        assert stats.probabilistic_sharpe_ratio(good) > 0.99

    def test_noise_is_not_significant(self):
        idx = pd.date_range("2000-01-31", periods=60, freq=MONTH_END)
        noise = pd.Series(np.random.default_rng(7).normal(0.0, 0.03, 60), index=idx)
        assert stats.probabilistic_sharpe_ratio(noise) < 0.95

    def test_negative_skew_lowers_psr(self):
        """同じシャープでも、負に歪んだ分布は評価が下がる。"""
        idx = pd.date_range("2000-01-31", periods=240, freq=MONTH_END)
        rng = np.random.default_rng(8)
        symmetric = rng.normal(0.008, 0.02, 240)

        skewed = symmetric.copy()
        skewed[:12] -= 0.06  # 大きな損失をまとめて入れる
        skewed = skewed - skewed.mean() + symmetric.mean()
        skewed = skewed / skewed.std(ddof=1) * symmetric.std(ddof=1)
        skewed = skewed - skewed.mean() + symmetric.mean()

        a = pd.Series(symmetric, index=idx)
        b = pd.Series(skewed, index=idx)
        assert pd.Series(b).skew() < pd.Series(a).skew()
        assert stats.probabilistic_sharpe_ratio(b) < stats.probabilistic_sharpe_ratio(a)

    def test_higher_threshold_lowers_psr(self):
        idx = pd.date_range("2000-01-31", periods=240, freq=MONTH_END)
        good = pd.Series(np.random.default_rng(9).normal(0.01, 0.02, 240), index=idx)
        assert stats.probabilistic_sharpe_ratio(
            good, benchmark_sharpe=2.0
        ) < stats.probabilistic_sharpe_ratio(good, benchmark_sharpe=0.0)

    def test_too_few_points(self):
        idx = pd.date_range("2024-01-31", periods=2, freq=MONTH_END)
        assert np.isnan(stats.probabilistic_sharpe_ratio(pd.Series([0.01, 0.02], index=idx)))


class TestDeflatedSharpe:
    @pytest.fixture
    def track_record(self):
        idx = pd.date_range("2005-01-31", periods=240, freq=MONTH_END)
        return pd.Series(np.random.default_rng(10).normal(0.009, 0.02, 240), index=idx)

    def test_expected_max_grows_with_trials(self):
        few = stats.expected_max_sharpe(5, variance=0.25)
        many = stats.expected_max_sharpe(500, variance=0.25)
        assert many > few > 0

    def test_expected_max_grows_with_variance(self):
        assert stats.expected_max_sharpe(100, variance=1.0) > stats.expected_max_sharpe(
            100, variance=0.1
        )

    def test_more_trials_lowers_dsr(self, track_record):
        """試行回数が増えるほど、同じ成績でも有意性は下がる。"""
        few = stats.deflated_sharpe_ratio(track_record, n_trials=5, trial_variance=0.25)
        many = stats.deflated_sharpe_ratio(track_record, n_trials=1000, trial_variance=0.25)
        assert many < few

    def test_dsr_below_psr(self, track_record):
        """閾値を上げるので、DSR は必ず PSR より小さい。"""
        psr = stats.probabilistic_sharpe_ratio(track_record)
        dsr = stats.deflated_sharpe_ratio(track_record, n_trials=100, trial_variance=0.25)
        assert dsr < psr

    def test_trials_derive_count_and_variance(self, track_record):
        sharpes = [0.2, 0.5, 0.9, 1.4, 0.1, -0.3]
        from_trials = stats.deflated_sharpe_ratio(track_record, trials=sharpes)
        explicit = stats.deflated_sharpe_ratio(
            track_record, n_trials=len(sharpes), trial_variance=float(np.var(sharpes, ddof=1))
        )
        assert from_trials == pytest.approx(explicit)

    def test_requires_trials_or_parameters(self, track_record):
        with pytest.raises(ValidationError, match="trials を渡すか"):
            stats.deflated_sharpe_ratio(track_record, n_trials=10)

    def test_trials_need_two_values(self, track_record):
        with pytest.raises(ValidationError, match="2 つ以上"):
            stats.deflated_sharpe_ratio(track_record, trials=[0.5])

    def test_n_trials_validation(self, track_record):
        with pytest.raises(ValidationError, match="n_trials"):
            stats.expected_max_sharpe(1, variance=0.25)


class TestNormalDistAccuracy:
    def test_matches_scipy(self):
        """標準ライブラリの NormalDist が scipy と一致することの確認。"""
        scipy_stats = pytest.importorskip("scipy.stats")
        for x in (-3.0, -1.0, 0.0, 0.5, 2.5):
            assert stats._NORMAL.cdf(x) == pytest.approx(scipy_stats.norm.cdf(x), abs=1e-12)
        for p in (0.001, 0.05, 0.5, 0.975, 0.999):
            assert stats._NORMAL.inv_cdf(p) == pytest.approx(scipy_stats.norm.ppf(p), abs=1e-9)
