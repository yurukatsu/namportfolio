import numpy as np
import pandas as pd
import pytest

from namportfolio import risk
from namportfolio.core.errors import ValidationError
from namportfolio.performance import MONTH_END

DATES = pd.date_range("2024-01-31", periods=2, freq=MONTH_END)
FACTORS = ["F1", "F2"]

# エクスポージャー: JP0000 は F1 のみ、JP0001 は F2 のみ、JP0002 は両方 0.5
EXPOSURES = {
    "JP0000": (1.0, 0.0),
    "JP0001": (0.0, 1.0),
    "JP0002": (0.5, 0.5),
}
WEIGHTS = {"JP0000": 0.5, "JP0001": 0.3, "JP0002": 0.2}
BENCH = {"JP0000": 1 / 3, "JP0001": 1 / 3, "JP0002": 1 / 3}

# x = (0.6, 0.4)
COV = np.array([[0.04, 0.01], [0.01, 0.09]])
FACTOR_VARIANCE = 0.6**2 * 0.04 + 2 * 0.6 * 0.4 * 0.01 + 0.4**2 * 0.09  # 0.0336
SPECIFIC_VARIANCE = (0.5**2 + 0.3**2 + 0.2**2) * 0.01  # 0.0038


@pytest.fixture
def panel():
    rows = [
        {
            "date": date,
            "bid": bid,
            "weight": WEIGHTS[bid],
            "bench_weight": BENCH[bid],
            "F1": EXPOSURES[bid][0],
            "F2": EXPOSURES[bid][1],
            "specific_risk": 0.10,
            "specific_ret": 0.01 if bid == "JP0000" else -0.005,
        }
        for date in DATES
        for bid in EXPOSURES
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def covariance_long():
    """date / factor_1 / factor_2 / cov の long 形式。"""
    rows = [
        {"date": date, "factor_1": a, "factor_2": b, "cov": COV[i, j]}
        for date in DATES
        for i, a in enumerate(FACTORS)
        for j, b in enumerate(FACTORS)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def covariance_matrix():
    """date + 行ラベル + ファクター列 の形式。"""
    rows = [
        {"date": date, "factor": a, "F1": COV[i, 0], "F2": COV[i, 1]}
        for date in DATES
        for i, a in enumerate(FACTORS)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def factor_returns():
    return pd.DataFrame(
        {"F1": [0.02, -0.01], "F2": [-0.01, 0.03]},
        index=DATES,
    )


class TestFactorExposures:
    def test_weighted_sum(self, panel):
        result = risk.factor_exposures(panel, factors=FACTORS)
        assert result.loc[DATES[0], "F1"] == pytest.approx(0.6)
        assert result.loc[DATES[0], "F2"] == pytest.approx(0.4)

    def test_active_exposures(self, panel):
        result = risk.factor_exposures(panel, factors=FACTORS, benchmark_weight="bench_weight")
        # ベンチは等ウェイト -> x_b = (0.5, 0.5)
        assert result.loc[DATES[0], "F1"] == pytest.approx(0.6 - 0.5)
        assert result.loc[DATES[0], "F2"] == pytest.approx(0.4 - 0.5)

    def test_missing_factor_column(self, panel):
        with pytest.raises(ValidationError, match="必須カラム"):
            risk.factor_exposures(panel, factors=["F1", "F3"])


class TestRiskDecomposition:
    def test_matches_hand_calculation(self, panel, covariance_long):
        result = risk.risk_decomposition(panel, covariance_long, factors=FACTORS)
        assert result.loc[DATES[0], "factor_risk"] == pytest.approx(np.sqrt(FACTOR_VARIANCE))
        assert result.loc[DATES[0], "specific_risk"] == pytest.approx(np.sqrt(SPECIFIC_VARIANCE))
        assert result.loc[DATES[0], "total_risk"] == pytest.approx(
            np.sqrt(FACTOR_VARIANCE + SPECIFIC_VARIANCE)
        )

    def test_variances_add_up_not_risks(self, panel, covariance_long):
        """リスクは足せないが分散は足せる。"""
        row = risk.risk_decomposition(panel, covariance_long, factors=FACTORS).iloc[0]
        assert row["factor_risk"] + row["specific_risk"] != pytest.approx(row["total_risk"])
        assert row["factor_risk"] ** 2 + row["specific_risk"] ** 2 == pytest.approx(
            row["total_risk"] ** 2
        )

    def test_matrix_format_covariance(self, panel, covariance_long, covariance_matrix):
        from_long = risk.risk_decomposition(panel, covariance_long, factors=FACTORS)
        from_matrix = risk.risk_decomposition(panel, covariance_matrix, factors=FACTORS)
        pd.testing.assert_frame_equal(from_long, from_matrix)

    def test_specific_as_variance(self, panel, covariance_long):
        as_variance = panel.assign(specific_risk=panel["specific_risk"] ** 2)
        result = risk.risk_decomposition(
            as_variance, covariance_long, factors=FACTORS, specific_is_variance=True
        )
        assert result.loc[DATES[0], "specific_risk"] == pytest.approx(np.sqrt(SPECIFIC_VARIANCE))

    def test_active_risk_is_smaller(self, panel, covariance_long):
        """アクティブウェイトのほうがエクスポージャーが小さいのでリスクも小さい。"""
        total = risk.risk_decomposition(panel, covariance_long, factors=FACTORS)
        active = risk.risk_decomposition(
            panel, covariance_long, factors=FACTORS, benchmark_weight="bench_weight"
        )
        assert active.loc[DATES[0], "total_risk"] < total.loc[DATES[0], "total_risk"]

    def test_covariance_needs_date(self, panel, covariance_long):
        with pytest.raises(ValidationError, match="date"):
            risk.risk_decomposition(panel, covariance_long.drop(columns="date"), factors=FACTORS)


class TestRiskContribution:
    def test_contributions_sum_to_total_risk(self, panel, covariance_long):
        """cctr は加法的。全ファクター＋特異でトータルリスクになる。"""
        contributions = risk.factor_risk_contribution(panel, covariance_long, factors=FACTORS)
        totals = risk.risk_decomposition(panel, covariance_long, factors=FACTORS)

        by_date = contributions.groupby(level=0)["cctr"].sum()
        assert by_date.to_numpy() == pytest.approx(totals["total_risk"].to_numpy())

    def test_mctr_by_hand(self, panel, covariance_long):
        contributions = risk.factor_risk_contribution(panel, covariance_long, factors=FACTORS)
        sigma = np.sqrt(FACTOR_VARIANCE + SPECIFIC_VARIANCE)
        expected = (COV @ np.array([0.6, 0.4])) / sigma
        first = contributions.xs(DATES[0], level=0)
        assert first.loc["F1", "mctr"] == pytest.approx(expected[0])
        assert first.loc["F2", "mctr"] == pytest.approx(expected[1])

    def test_specific_row_exists(self, panel, covariance_long):
        contributions = risk.factor_risk_contribution(panel, covariance_long, factors=FACTORS)
        assert "specific" in contributions.xs(DATES[0], level=0).index

    def test_pct_of_total_sums_to_one(self, panel, covariance_long):
        contributions = risk.factor_risk_contribution(panel, covariance_long, factors=FACTORS)
        by_date = contributions.groupby(level=0)["pct_of_total"].sum()
        assert by_date.to_numpy() == pytest.approx(1.0)

    def test_factor_groups(self, panel, covariance_long):
        contributions = risk.factor_risk_contribution(
            panel,
            covariance_long,
            factors=FACTORS,
            factor_groups={"F1": "style", "F2": "style"},
        )
        first = contributions.xs(DATES[0], level=0)
        assert set(first.index) == {"style", "specific"}
        assert first["cctr"].sum() == pytest.approx(
            risk.risk_decomposition(panel, covariance_long, factors=FACTORS).loc[
                DATES[0], "total_risk"
            ]
        )


class TestReturnAttribution:
    def test_contribution_by_hand(self, panel, factor_returns):
        result = risk.factor_return_attribution(panel, factor_returns, factors=FACTORS)
        first = result.xs(DATES[0], level=0)
        assert first.loc["F1", "contribution"] == pytest.approx(0.6 * 0.02)
        assert first.loc["F2", "contribution"] == pytest.approx(0.4 * -0.01)

    def test_with_specific_return_sums_to_portfolio(self, panel, factor_returns):
        result = risk.factor_return_attribution(
            panel, factor_returns, factors=FACTORS, specific_return="specific_ret"
        )
        by_date = result.groupby(level=0)["contribution"].sum()

        # ポートフォリオリターン = Σ w_i (x_i·f + u_i)
        expected = []
        for date in DATES:
            f = factor_returns.loc[date].to_numpy()
            total = 0.0
            for bid, (e1, e2) in EXPOSURES.items():
                specific = 0.01 if bid == "JP0000" else -0.005
                total += WEIGHTS[bid] * (np.array([e1, e2]) @ f + specific)
            expected.append(total)
        assert by_date.to_numpy() == pytest.approx(expected)

    def test_long_format_factor_returns(self, panel, factor_returns):
        long = factor_returns.reset_index(names="date").melt(
            id_vars="date", var_name="factor", value_name="ret"
        )
        pd.testing.assert_frame_equal(
            risk.factor_return_attribution(panel, long, factors=FACTORS),
            risk.factor_return_attribution(panel, factor_returns, factors=FACTORS),
        )

    def test_active_attribution(self, panel, factor_returns):
        result = risk.factor_return_attribution(
            panel, factor_returns, factors=FACTORS, benchmark_weight="bench_weight"
        )
        first = result.xs(DATES[0], level=0)
        assert first.loc["F1", "exposure"] == pytest.approx(0.1)


class TestBiasStatistic:
    @pytest.fixture
    def series(self):
        idx = pd.date_range("2020-01-31", periods=120, freq=MONTH_END)
        rng = np.random.default_rng(0)
        predicted = pd.Series(0.15, index=idx)  # 年率 15%
        monthly = 0.15 / np.sqrt(12)
        realized = pd.Series(rng.normal(0, monthly, len(idx)), index=idx)
        return realized, predicted

    def test_correct_forecast_gives_one(self, series):
        realized, predicted = series
        assert risk.bias_statistic(realized, predicted) == pytest.approx(1.0, abs=0.12)

    def test_underestimated_risk_gives_above_one(self, series):
        realized, predicted = series
        assert risk.bias_statistic(realized, predicted / 2) > 1.5

    def test_overestimated_risk_gives_below_one(self, series):
        realized, predicted = series
        assert risk.bias_statistic(realized, predicted * 2) < 0.7

    def test_period_scale_matters(self, series):
        realized, predicted = series
        annualized = risk.bias_statistic(realized, predicted)
        as_period = risk.bias_statistic(realized, predicted, predicted_is_annualized=False)
        assert as_period == pytest.approx(annualized / np.sqrt(12), rel=1e-9)

    def test_too_few_points(self):
        idx = pd.date_range("2024-01-31", periods=1, freq=MONTH_END)
        assert np.isnan(
            risk.bias_statistic(pd.Series([0.01], index=idx), pd.Series([0.15], index=idx))
        )

    def test_rolling(self, series):
        realized, predicted = series
        rolled = risk.rolling_bias_statistic(realized, predicted, 24)
        assert rolled.iloc[:23].isna().all()
        assert rolled.notna().sum() == 120 - 23

    def test_confidence_interval_narrows_with_sample(self):
        wide = risk.bias_confidence_interval(24)
        narrow = risk.bias_confidence_interval(240)
        assert wide[0] < narrow[0] < 1.0 < narrow[1] < wide[1]

    def test_confidence_validation(self):
        with pytest.raises(ValidationError, match="confidence"):
            risk.bias_confidence_interval(100, confidence=0.5)


class TestRealizedVsPredicted:
    def test_ratio(self):
        idx = pd.date_range("2020-01-31", periods=60, freq=MONTH_END)
        rng = np.random.default_rng(1)
        realized = pd.Series(rng.normal(0, 0.10 / np.sqrt(12), len(idx)), index=idx)
        predicted = pd.Series(0.10, index=idx)

        result = risk.realized_vs_predicted(realized, predicted, window=24)
        assert list(result.columns) == ["predicted", "realized", "ratio"]
        assert result["ratio"].dropna().mean() == pytest.approx(1.0, abs=0.25)
