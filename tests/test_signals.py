import numpy as np
import pandas as pd
import pytest

from namportfolio import signals
from namportfolio.core.errors import ValidationError
from namportfolio.performance import MONTH_END

DATES = pd.date_range("2024-01-31", periods=4, freq=MONTH_END)


@pytest.fixture
def panel():
    """業種 A/B が 5 銘柄ずつ。factor は 0〜9、size は factor に比例。"""
    rows = [
        {
            "date": date,
            "bid": f"JP{i:04d}",
            "factor": float(i),
            "sector": "A" if i < 5 else "B",
            "size": float(i) * 2.0,
            "mktcap": 1.0 + i,
        }
        for date in DATES
        for i in range(10)
    ]
    return pd.DataFrame(rows)


class TestWinsorize:
    def test_clips_tails_without_dropping_rows(self, panel):
        result = signals.winsorize(panel, factor="factor", lower=0.1, upper=0.9)
        assert len(result) == len(panel), "行数は減らない"
        first = result[panel["date"] == DATES[0]]
        assert first.min() == pytest.approx(np.quantile(np.arange(10.0), 0.1))
        assert first.max() == pytest.approx(np.quantile(np.arange(10.0), 0.9))

    def test_per_date(self, panel):
        """日付ごとに境界を計算する（全期間をまとめない）。"""
        shifted = panel.copy()
        shifted.loc[shifted["date"] == DATES[1], "factor"] += 100.0
        result = signals.winsorize(shifted, factor="factor", lower=0.1, upper=0.9)
        assert result[shifted["date"] == DATES[1]].min() > 100.0

    def test_group_wise(self, panel):
        result = signals.winsorize(panel, factor="factor", lower=0.2, upper=0.8, group="sector")
        first = panel[panel["date"] == DATES[0]].assign(w=result)
        assert first[first["sector"] == "A"]["w"].max() < 5.0, "業種内で丸める"

    def test_invalid_limits(self, panel):
        with pytest.raises(ValidationError, match="lower < upper"):
            signals.winsorize(panel, factor="factor", lower=0.9, upper=0.1)


class TestClipOutliers:
    def test_clips_at_sigma(self, panel):
        outlier = panel.copy()
        outlier.loc[outlier["factor"] == 9.0, "factor"] = 1000.0
        result = signals.clip_outliers(outlier, factor="factor", n_sigma=2.0)
        assert result.max() < 1000.0

    def test_rejects_non_positive_sigma(self, panel):
        with pytest.raises(ValidationError, match="n_sigma"):
            signals.clip_outliers(panel, factor="factor", n_sigma=0.0)


class TestStandardize:
    def test_zscore_has_zero_mean_unit_std(self, panel):
        result = signals.standardize(panel, factor="factor")
        first = result[panel["date"] == DATES[0]]
        assert first.mean() == pytest.approx(0.0, abs=1e-12)
        assert first.std(ddof=1) == pytest.approx(1.0)

    def test_rank_is_between_zero_and_one(self, panel):
        result = signals.standardize(panel, factor="factor", method="rank")
        assert result.min() > 0.0
        assert result.max() == pytest.approx(1.0)

    def test_rank_is_robust_to_outliers(self, panel):
        outlier = panel.copy()
        outlier.loc[outlier["factor"] == 9.0, "factor"] = 1e6
        ranked = signals.standardize(outlier, factor="factor", method="rank")
        zscored = signals.standardize(outlier, factor="factor")
        assert ranked.std() == pytest.approx(
            signals.standardize(panel, factor="factor", method="rank").std()
        ), "順位は外れ値で変わらない"
        assert zscored.max() > 2.5, "z-score は外れ値に引っ張られる"

    def test_group_wise(self, panel):
        result = signals.standardize(panel, factor="factor", group="sector")
        first = panel[panel["date"] == DATES[0]].assign(z=result)
        assert first.groupby("sector")["z"].mean().abs().max() == pytest.approx(0.0, abs=1e-12)

    def test_weighted_zscore_differs(self, panel):
        plain = signals.standardize(panel, factor="factor")
        weighted = signals.standardize(panel, factor="factor", weight="mktcap")
        assert not np.allclose(plain.to_numpy(), weighted.to_numpy())

    def test_unknown_method(self, panel):
        with pytest.raises(ValidationError, match="method は"):
            signals.standardize(panel, factor="factor", method="minmax")


class TestNeutralize:
    def test_categorical_removes_group_means(self, panel):
        """業種で中立化すると、業種内平均が 0 になる。"""
        result = signals.neutralize(panel, factor="factor", by="sector")
        first = panel[panel["date"] == DATES[0]].assign(n=result)
        assert first.groupby("sector")["n"].mean().abs().max() == pytest.approx(0.0, abs=1e-10)

    def test_numeric_removes_correlation(self, panel):
        """サイズで中立化すると、残差はサイズと無相関になる。"""
        rng = np.random.default_rng(0)
        noisy = panel.assign(factor=panel["size"] * 0.5 + rng.normal(0, 1, len(panel)))
        result = signals.neutralize(noisy, factor="factor", by="size")
        first = noisy[noisy["date"] == DATES[0]].assign(n=result)
        assert first["n"].corr(first["size"]) == pytest.approx(0.0, abs=1e-10)

    def test_multiple_factors(self, panel):
        result = signals.neutralize(panel, factor="factor", by=["sector", "size"])
        assert result.notna().all()

    def test_perfectly_explained_signal_leaves_no_residual(self, panel):
        """factor が size の定数倍なら、中立化後は何も残らない。"""
        result = signals.neutralize(panel, factor="factor", by="size")
        assert result.to_numpy() == pytest.approx(0.0, abs=1e-12)

    def test_weighted(self, panel):
        rng = np.random.default_rng(3)
        noisy = panel.assign(factor=panel["size"] * 0.5 + rng.normal(0, 1, len(panel)))
        plain = signals.neutralize(noisy, factor="factor", by="size")
        weighted = signals.neutralize(noisy, factor="factor", by="size", weight="mktcap")
        assert not np.allclose(plain.to_numpy(), weighted.to_numpy())

    def test_insufficient_observations(self, panel):
        """観測数が説明変数より少ない日は NaN。"""
        thin = panel[panel["bid"].isin(["JP0000", "JP0005"])]
        result = signals.neutralize(thin, factor="factor", by="sector")
        assert result.isna().all()

    def test_missing_explanatory_column(self, panel):
        with pytest.raises(ValidationError, match="必須カラム"):
            signals.neutralize(panel, factor="factor", by="nonexistent")

    def test_index_is_preserved(self, panel):
        shuffled = panel.sample(frac=1.0, random_state=2)
        result = signals.neutralize(shuffled, factor="factor", by="sector")
        assert result.index.equals(shuffled.index)


class TestCoverage:
    def test_counts(self, panel):
        result = signals.coverage(panel, factor="factor")
        assert list(result.columns) == ["n_valid", "n_total", "missing_rate", "coverage_rate"]
        assert (result["n_valid"] == 10).all()
        assert (result["missing_rate"] == 0.0).all()

    def test_missing_rate(self, panel):
        holed = panel.copy()
        holed.loc[holed["bid"] == "JP0000", "factor"] = np.nan
        result = signals.coverage(holed, factor="factor")
        assert (result["n_valid"] == 9).all()
        assert result["missing_rate"].iloc[0] == pytest.approx(0.1)

    def test_against_universe(self, panel):
        universe = pd.DataFrame(
            [{"date": d, "bid": f"JP{i:04d}"} for d in DATES for i in range(20)]
        )
        result = signals.coverage(panel, factor="factor", universe=universe)
        assert result["coverage_rate"].iloc[0] == pytest.approx(0.5), "10 / 20 銘柄"


class TestDistributionSummary:
    def test_columns(self, panel):
        result = signals.distribution_summary(panel, factor="factor")
        for column in ("mean", "std", "skew", "kurtosis", "p01", "p50", "p99"):
            assert column in result.columns
        assert result["mean"].iloc[0] == pytest.approx(4.5)
        assert result["p50"].iloc[0] == pytest.approx(4.5)

    def test_detects_outlier_period(self, panel):
        spiked = panel.copy()
        spiked.loc[(spiked["date"] == DATES[2]) & (spiked["factor"] == 9.0), "factor"] = 1e5
        result = signals.distribution_summary(spiked, factor="factor")
        assert result["skew"].iloc[2] > result["skew"].iloc[0]


class TestSignalCorrelation:
    @pytest.fixture
    def multi(self, panel):
        rng = np.random.default_rng(1)
        return panel.assign(
            same=panel["factor"],
            noisy=panel["factor"] + rng.normal(0, 3, len(panel)),
        )

    def test_identical_signals_correlate_perfectly(self, multi):
        corr = signals.signal_correlation(multi, factors=["factor", "same"])
        assert corr.loc["factor", "same"] == pytest.approx(1.0)

    def test_noisy_signal_correlates_less(self, multi):
        corr = signals.signal_correlation(multi, factors=["factor", "same", "noisy"])
        assert list(corr.index) == ["factor", "same", "noisy"]
        assert corr.loc["factor", "noisy"] < 1.0

    def test_requires_two_factors(self, multi):
        with pytest.raises(ValidationError, match="2 つ以上"):
            signals.signal_correlation(multi, factors=["factor"])

    def test_rolling(self, multi):
        rolling = signals.rolling_signal_correlation(multi, factors=["factor", "same"])
        assert len(rolling) == len(DATES)
        assert rolling.to_numpy() == pytest.approx(1.0)

    def test_rolling_requires_exactly_two(self, multi):
        with pytest.raises(ValidationError, match="2 つ渡して"):
            signals.rolling_signal_correlation(multi, factors=["factor", "same", "noisy"])
