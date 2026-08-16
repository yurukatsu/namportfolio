import numpy as np
import pandas as pd
import pytest

from namportfolio.stats import newey_west_lags, newey_west_tstat, t_statistic


class TestTStatistic:
    def test_manual(self):
        values = [1.0, 2.0, 3.0, 4.0]
        expected = np.mean(values) / (np.std(values, ddof=1) / np.sqrt(4))
        assert t_statistic(values) == pytest.approx(expected)

    def test_against_hypothesised_mean(self):
        assert t_statistic([1.0, 2.0, 3.0], mu=2.0) == pytest.approx(0.0)

    def test_ignores_nan(self):
        assert t_statistic([1.0, 2.0, np.nan, 3.0]) == pytest.approx(t_statistic([1.0, 2.0, 3.0]))

    def test_too_few_points(self):
        assert np.isnan(t_statistic([1.0]))

    def test_zero_variance(self):
        assert np.isnan(t_statistic([2.0, 2.0, 2.0]))

    def test_accepts_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert t_statistic(s) == pytest.approx(t_statistic(s.to_list()))


class TestNeweyWest:
    def test_zero_lags_matches_plain_t(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.5, 1.0, 200)
        # ddof の違いだけずれるので、大標本ではほぼ一致する
        assert newey_west_tstat(values, lags=0) == pytest.approx(t_statistic(values), rel=0.01)

    def test_positive_autocorrelation_lowers_t(self):
        """自己相関があると素の t 値は有意性を過大評価する。"""
        rng = np.random.default_rng(1)
        noise = rng.normal(0.0, 1.0, 400)
        ar = np.zeros(400)
        for i in range(1, 400):
            ar[i] = 0.7 * ar[i - 1] + noise[i]
        ar += 0.5

        assert abs(newey_west_tstat(ar)) < abs(t_statistic(ar))

    def test_lag_rule(self):
        assert newey_west_lags(100) == 4
        assert newey_west_lags(1) == 0

    def test_lags_capped_at_sample_size(self):
        assert not np.isnan(newey_west_tstat([1.0, 2.0, 3.0], lags=50))

    def test_too_few_points(self):
        assert np.isnan(newey_west_tstat([1.0]))
