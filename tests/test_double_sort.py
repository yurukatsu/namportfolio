import numpy as np
import pandas as pd
import pytest

from namportfolio import quantile as q
from namportfolio.core.errors import ValidationError
from namportfolio.performance import MONTH_END

DATES = pd.date_range("2024-01-31", periods=6, freq=MONTH_END)


@pytest.fixture
def panel():
    """25 銘柄。factor_a と factor_b が独立に 5 分位ずつ、全セルに 1 銘柄。

    リターンは factor_a にだけ比例させる（factor_b は無関係）。
    """
    rows = []
    for date in DATES:
        for i in range(5):  # factor_a の分位
            for j in range(5):  # factor_b の分位
                rows.append(
                    {
                        "date": date,
                        "bid": f"JP{i}{j}",
                        "factor_a": float(i),
                        "factor_b": float(j),
                        "fwd_ret": i * 0.01,
                        "mktcap": 1.0,
                    }
                )
    return pd.DataFrame(rows)


class TestDoubleSortReturns:
    def test_shape_and_labels(self, panel):
        result = q.double_sort_returns(
            panel, factor_1="factor_a", factor_2="factor_b", forward_return="fwd_ret"
        )
        assert list(result.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
        assert result.index.names == ["date", "factor_a"]
        assert result.columns.name == "factor_b"
        assert len(result) == len(DATES) * 5

    def test_cell_values(self, panel):
        """リターンは factor_a にのみ依存するので、行方向に一定・列方向に増える。"""
        result = q.double_sort_returns(
            panel, factor_1="factor_a", factor_2="factor_b", forward_return="fwd_ret"
        )
        first = result.xs(DATES[0], level=0)
        assert first.loc["Q1"].to_numpy() == pytest.approx(0.0)
        assert first.loc["Q5"].to_numpy() == pytest.approx(0.04)

    def test_axes_are_swappable(self, panel):
        """factor を入れ替えると行と列が転置される。"""
        forward = q.double_sort_returns(
            panel, factor_1="factor_a", factor_2="factor_b", forward_return="fwd_ret"
        )
        reversed_axes = q.double_sort_returns(
            panel, factor_1="factor_b", factor_2="factor_a", forward_return="fwd_ret"
        )
        assert forward.xs(DATES[0], level=0).to_numpy() == pytest.approx(
            reversed_axes.xs(DATES[0], level=0).to_numpy().T
        )

    def test_different_quantile_counts(self, panel):
        result = q.double_sort_returns(
            panel,
            factor_1="factor_a",
            factor_2="factor_b",
            forward_return="fwd_ret",
            n_quantiles_1=5,
            n_quantiles_2=2,
        )
        assert list(result.columns) == ["Q1", "Q2"]

    def test_weighted(self, panel):
        weighted = panel.copy()
        # Q1 行 × Q1 列 のセルは 1 銘柄なので、ウェイトを変えても値は変わらない
        weighted.loc[weighted["bid"] == "JP00", "mktcap"] = 9.0
        result = q.double_sort_returns(
            panel, factor_1="factor_a", factor_2="factor_b", forward_return="fwd_ret"
        )
        result_w = q.double_sort_returns(
            weighted,
            factor_1="factor_a",
            factor_2="factor_b",
            forward_return="fwd_ret",
            weight="mktcap",
        )
        assert result.to_numpy() == pytest.approx(result_w.to_numpy())

    def test_missing_column(self, panel):
        with pytest.raises(ValidationError, match="必須カラム"):
            q.double_sort_returns(
                panel, factor_1="factor_a", factor_2="nope", forward_return="fwd_ret"
            )


class TestDoubleSortSummary:
    @pytest.fixture
    def cells(self, panel):
        return q.double_sort_returns(
            panel, factor_1="factor_a", factor_2="factor_b", forward_return="fwd_ret"
        )

    def test_matrix_shape(self, cells):
        matrix = q.double_sort_summary(cells, statistic="mean")
        assert matrix.shape == (5, 5)
        assert list(matrix.index) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
        assert list(matrix.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]

    def test_mean_matches_cells(self, cells):
        matrix = q.double_sort_summary(cells, statistic="mean")
        assert matrix.loc["Q5"].to_numpy() == pytest.approx(0.04)
        assert matrix.loc["Q1"].to_numpy() == pytest.approx(0.0)

    def test_annualized(self, cells):
        matrix = q.double_sort_summary(cells, statistic="annualized_return")
        # 月次 0.04 -> 年率 (1.04^12 - 1)
        assert matrix.loc["Q5", "Q1"] == pytest.approx(1.04**12 - 1)

    def test_n_periods(self, cells):
        matrix = q.double_sort_summary(cells, statistic="n_periods")
        assert matrix.to_numpy() == pytest.approx(6.0)

    def test_statistic_validation(self, cells):
        with pytest.raises(ValidationError, match="statistic は"):
            q.double_sort_summary(cells, statistic="skew")


class TestDoubleSortCounts:
    def test_independent_factors_fill_every_cell(self, panel):
        counts = q.double_sort_counts(panel, factor_1="factor_a", factor_2="factor_b")
        assert counts.to_numpy() == pytest.approx(1.0), "25 銘柄が 25 セルに 1 つずつ"

    def test_correlated_factors_concentrate_on_diagonal(self, panel):
        """2 つのファクターが同一なら対角セルにしか銘柄が入らない。"""
        same = panel.assign(factor_b=panel["factor_a"])
        counts = q.double_sort_counts(same, factor_1="factor_a", factor_2="factor_b")
        matrix = counts.fillna(0.0).to_numpy()
        assert np.diag(matrix).sum() == pytest.approx(matrix.sum())

    def test_shape(self, panel):
        counts = q.double_sort_counts(
            panel, factor_1="factor_a", factor_2="factor_b", n_quantiles_2=2
        )
        assert counts.shape == (5, 2)
