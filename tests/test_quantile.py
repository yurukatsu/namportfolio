import sys

import numpy as np
import pandas as pd
import pytest

from namportfolio import quantile as q
from namportfolio.core.errors import ValidationError
from namportfolio.performance import MONTH_END

DATES = pd.date_range("2024-01-31", periods=6, freq=MONTH_END)
BIDS = [f"JP{i:04d}" for i in range(10)]


@pytest.fixture
def panel():
    """factor は 0〜9、fwd_ret は factor に完全比例（IC = 1 になる）。"""
    rows = [
        {
            "date": date,
            "bid": bid,
            "factor": float(i),
            "fwd_ret": i * 0.01,
            "sector": "A" if i < 5 else "B",
            "mktcap": 1.0,
        }
        for date in DATES
        for i, bid in enumerate(BIDS)
    ]
    return pd.DataFrame(rows)


class TestAssignQuantiles:
    def test_splits_evenly(self, panel):
        assigned = q.assign_quantiles(panel, factor="factor", n_quantiles=5)
        first_day = assigned[panel["date"] == DATES[0]]
        # factor 0,1 -> Q1 / 2,3 -> Q2 / ... / 8,9 -> Q5
        assert list(first_day) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    def test_descending(self, panel):
        assigned = q.assign_quantiles(panel, factor="factor", n_quantiles=5, ascending=False)
        first_day = assigned[panel["date"] == DATES[0]]
        assert list(first_day) == [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]

    def test_group_neutral(self, panel):
        """業種内で分位を作ると、各業種から均等に選ばれる。"""
        assigned = q.assign_quantiles(panel, factor="factor", n_quantiles=5, group="sector")
        first_day = panel[panel["date"] == DATES[0]].assign(q=assigned)
        top = first_day[first_day["q"] == 5]
        assert set(top["sector"]) == {"A", "B"}, "各業種の上位が Q5 に入る"

    def test_index_is_preserved(self, panel):
        shuffled = panel.sample(frac=1.0, random_state=0)
        assigned = q.assign_quantiles(shuffled, factor="factor", n_quantiles=5)
        assert assigned.index.equals(shuffled.index)

    def test_too_few_assets_gives_nan(self, panel):
        thin = panel[panel["bid"].isin(BIDS[:3])]
        assigned = q.assign_quantiles(thin, factor="factor", n_quantiles=5)
        assert assigned.isna().all(), "3 銘柄で 5 分位は作らない"

    def test_min_assets_override(self, panel):
        thin = panel[panel["bid"].isin(BIDS[:3])]
        assigned = q.assign_quantiles(thin, factor="factor", n_quantiles=5, min_assets=3)
        assert assigned.notna().all()

    def test_ties_are_split(self):
        """同値だらけでも分位のサイズが揃う（qcut は境界重複で失敗する）。"""
        flat = pd.DataFrame({"date": [DATES[0]] * 10, "bid": BIDS, "factor": [1.0] * 10})
        assigned = q.assign_quantiles(flat, factor="factor", n_quantiles=5)
        assert assigned.value_counts().to_dict() == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}

    def test_rejects_single_quantile(self, panel):
        with pytest.raises(ValidationError, match="n_quantiles"):
            q.assign_quantiles(panel, factor="factor", n_quantiles=1)

    def test_missing_column(self, panel):
        with pytest.raises(ValidationError, match="必須カラム"):
            q.assign_quantiles(panel, factor="nonexistent")


class TestQuantileReturns:
    def test_equal_weighted_means(self, panel):
        result = q.quantile_returns(panel, factor="factor", forward_return="fwd_ret")
        assert list(result.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
        assert len(result) == 6
        assert result["Q1"].iloc[0] == pytest.approx(0.005)  # (0.00 + 0.01) / 2
        assert result["Q5"].iloc[0] == pytest.approx(0.085)  # (0.08 + 0.09) / 2

    def test_monotonic_across_quantiles(self, panel):
        result = q.quantile_returns(panel, factor="factor", forward_return="fwd_ret")
        assert result.iloc[0].is_monotonic_increasing

    def test_weighted(self, panel):
        weighted = panel.copy()
        # Q1 = {factor 0, 1}。ウェイトを 1:3 にすると平均は 0.0075 に寄る
        weighted.loc[weighted["factor"] == 1.0, "mktcap"] = 3.0
        result = q.quantile_returns(
            weighted, factor="factor", forward_return="fwd_ret", weight="mktcap"
        )
        assert result["Q1"].iloc[0] == pytest.approx((0.0 * 1 + 0.01 * 3) / 4)

    def test_all_quantiles_present_even_if_empty(self, panel):
        thin = panel[panel["bid"].isin(BIDS[:5])]
        result = q.quantile_returns(thin, factor="factor", n_quantiles=5, forward_return="fwd_ret")
        assert list(result.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]

    def test_index_is_sorted_datetime(self, panel):
        shuffled = panel.sample(frac=1.0, random_state=1)
        result = q.quantile_returns(shuffled, factor="factor", forward_return="fwd_ret")
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.is_monotonic_increasing


class TestClassReturns:
    @pytest.fixture
    def rated(self, panel):
        """factor 0〜2 を C、3〜6 を B、7〜9 を A とする格付け列を足す。"""
        return panel.assign(
            rating=np.select([panel["factor"] < 3, panel["factor"] < 7], ["C", "B"], default="A")
        )

    def test_groups_by_class_values(self, rated):
        result = q.class_returns(rated, classes="rating", forward_return="fwd_ret")
        assert list(result.columns) == ["A", "B", "C"], "既定は値のソート順"
        assert result["C"].iloc[0] == pytest.approx(0.01)  # (0 + 1 + 2) / 3 * 0.01
        assert result["A"].iloc[0] == pytest.approx(0.08)  # (7 + 8 + 9) / 3 * 0.01

    def test_explicit_order(self, rated):
        result = q.class_returns(
            rated, classes="rating", forward_return="fwd_ret", class_order=["C", "B", "A"]
        )
        assert list(result.columns) == ["C", "B", "A"]

    def test_order_can_filter(self, rated):
        result = q.class_returns(
            rated, classes="rating", forward_return="fwd_ret", class_order=["A", "C"]
        )
        assert list(result.columns) == ["A", "C"], "指定した列だけ返る"

    def test_weighted(self, rated):
        weighted = rated.copy()
        weighted.loc[weighted["factor"] == 2.0, "mktcap"] = 8.0
        result = q.class_returns(
            weighted, classes="rating", forward_return="fwd_ret", weight="mktcap"
        )
        assert result["C"].iloc[0] == pytest.approx((0.0 * 1 + 0.01 * 1 + 0.02 * 8) / 10)

    def test_missing_class_excluded_by_default(self, rated):
        holed = rated.copy()
        holed.loc[holed["factor"] < 3, "rating"] = None
        result = q.class_returns(holed, classes="rating", forward_return="fwd_ret")
        assert list(result.columns) == ["A", "B"]

    def test_missing_class_included(self, rated):
        holed = rated.copy()
        holed.loc[holed["factor"] < 3, "rating"] = None
        result = q.class_returns(
            holed, classes="rating", forward_return="fwd_ret", include_missing=True
        )
        assert list(result.columns) == ["A", "B", "NA"], "欠損クラスは常に末尾"
        assert result["NA"].iloc[0] == pytest.approx(0.01)

    def test_custom_missing_label(self, rated):
        holed = rated.copy()
        holed.loc[holed["factor"] < 3, "rating"] = None
        result = q.class_returns(
            holed,
            classes="rating",
            forward_return="fwd_ret",
            include_missing=True,
            missing_label="未格付",
        )
        assert list(result.columns) == ["A", "B", "未格付"]

    def test_missing_column(self, rated):
        with pytest.raises(ValidationError, match="必須カラム"):
            q.class_returns(rated, classes="nonexistent", forward_return="fwd_ret")


class TestMissingQuantileClass:
    @pytest.fixture
    def holed(self, panel):
        """factor 0 と 1 の銘柄だけ値が欠損している。"""
        holed = panel.copy()
        holed.loc[holed["bid"].isin([BIDS[0], BIDS[1]]), "factor"] = np.nan
        return holed

    def test_assign_marks_missing_with_zero(self, holed):
        assigned = q.assign_quantiles(holed, factor="factor", missing_class=True)
        first = assigned[holed["date"] == DATES[0]]
        assert list(first)[:2] == [0.0, 0.0], "欠損は 0（分位は 1 から）"
        assert first.iloc[2:].min() >= 1.0

    def test_assign_leaves_missing_as_nan_by_default(self, holed):
        assigned = q.assign_quantiles(holed, factor="factor")
        assert assigned[holed["date"] == DATES[0]].iloc[:2].isna().all()

    def test_thin_day_is_not_a_missing_class(self, panel):
        """銘柄数不足で分位が作れない日は、欠損クラスにも入れない。"""
        thin = panel[panel["bid"].isin(BIDS[:3])]
        assigned = q.assign_quantiles(thin, factor="factor", missing_class=True)
        assert assigned.isna().all(), "値はあるが分位が作れない → NaN のまま"

    def test_quantile_returns_adds_na_column(self, holed):
        result = q.quantile_returns(
            holed, factor="factor", forward_return="fwd_ret", include_missing=True
        )
        assert list(result.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5", "NA"]
        assert result["NA"].iloc[0] == pytest.approx(0.005)  # (0.00 + 0.01) / 2

    def test_quantile_returns_excludes_missing_by_default(self, holed):
        result = q.quantile_returns(holed, factor="factor", forward_return="fwd_ret")
        assert list(result.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]

    def test_missing_does_not_shift_quantiles(self, holed):
        """欠損を別クラスにしても、値がある銘柄の分位割当は変わらない。"""
        with_na = q.quantile_returns(
            holed, factor="factor", forward_return="fwd_ret", include_missing=True
        )
        without = q.quantile_returns(holed, factor="factor", forward_return="fwd_ret")
        for label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            assert with_na[label].to_numpy() == pytest.approx(without[label].to_numpy())

    def test_turnover_includes_missing(self, holed):
        turnover = q.quantile_turnover(holed, factor="factor", include_missing=True)
        assert list(turnover.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5", "NA"]

    def test_transition_includes_missing(self, holed):
        matrix = q.quantile_transition_matrix(holed, factor="factor", include_missing=True)
        assert list(matrix.index) == ["Q1", "Q2", "Q3", "Q4", "Q5", "NA"]
        assert matrix.loc["NA", "NA"] == pytest.approx(1.0), "欠損のままの銘柄は動かない"


class TestSummaries:
    @pytest.fixture
    def returns(self, panel):
        return q.quantile_returns(panel, factor="factor", forward_return="fwd_ret")

    def test_summary_rows_and_columns(self, returns):
        summary = q.quantile_summary(returns)
        assert list(summary.index) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
        for column in ("mean", "annualized_return", "sharpe_ratio", "t_stat", "t_stat_nw"):
            assert column in summary.columns
        assert summary.loc["Q5", "mean"] == pytest.approx(0.085)

    def test_long_short_default_is_top_minus_bottom(self, returns):
        spread = q.long_short_returns(returns)
        assert spread.name == "Q5-Q1"
        assert spread.iloc[0] == pytest.approx(0.08)

    def test_long_short_explicit_labels(self, returns):
        spread = q.long_short_returns(returns, long="Q4", short="Q2")
        assert spread.name == "Q4-Q2"

    def test_long_short_unknown_label(self, returns):
        with pytest.raises(ValidationError, match="必須カラム"):
            q.long_short_returns(returns, long="Q9")


class TestInformationCoefficient:
    def test_perfect_signal(self, panel):
        ic = q.information_coefficient(panel, factor="factor", forward_return="fwd_ret")
        assert ic.to_numpy() == pytest.approx(1.0)

    def test_inverted_signal(self, panel):
        inverted = panel.assign(fwd_ret=-panel["fwd_ret"])
        ic = q.information_coefficient(inverted, factor="factor", forward_return="fwd_ret")
        assert ic.to_numpy() == pytest.approx(-1.0)

    def test_spearman_is_robust_to_outliers(self, panel):
        """1 銘柄だけ極端な値にすると Pearson は崩れるが Spearman は変わらない。"""
        outlier = panel.copy()
        mask = outlier["factor"] == 9.0
        outlier.loc[mask, "fwd_ret"] = -5.0

        spearman = q.information_coefficient(outlier, factor="factor", forward_return="fwd_ret")
        pearson = q.information_coefficient(
            outlier, factor="factor", forward_return="fwd_ret", method="pearson"
        )
        assert spearman.iloc[0] > pearson.iloc[0]

    def test_min_assets_filter(self, panel):
        ic = q.information_coefficient(
            panel, factor="factor", forward_return="fwd_ret", min_assets=20
        )
        assert ic.isna().all()

    def test_by_group(self, panel):
        ic = q.information_coefficient(
            panel, factor="factor", forward_return="fwd_ret", group="sector", min_assets=3
        )
        assert list(ic.columns) == ["A", "B"]
        assert ic.to_numpy().ravel() == pytest.approx(1.0)

    def test_unknown_method(self, panel):
        with pytest.raises(ValidationError, match="method は"):
            q.information_coefficient(
                panel, factor="factor", forward_return="fwd_ret", method="kendall"
            )

    def test_spearman_does_not_require_scipy(self, panel, monkeypatch):
        """pandas の corr(method="spearman") は scipy を要求するので使わない。

        本パッケージは pandas + numpy だけで動く必要がある（社内の制限環境）。
        順位に変換してから Pearson を取ることで依存を避けている。
        """
        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.stats", None)

        overall = q.information_coefficient(panel, factor="factor", forward_return="fwd_ret")
        by_group = q.information_coefficient(
            panel, factor="factor", forward_return="fwd_ret", group="sector", min_assets=3
        )
        assert overall.to_numpy() == pytest.approx(1.0)
        assert by_group.to_numpy().ravel() == pytest.approx(1.0)

    def test_summary_keys(self, panel):
        ic = q.information_coefficient(panel, factor="factor", forward_return="fwd_ret")
        summary = q.ic_summary(ic)
        for key in ("mean", "icir", "icir_annualized", "t_stat", "t_stat_nw", "hit_rate"):
            assert key in summary.index
        assert summary["mean"] == pytest.approx(1.0)
        assert summary["hit_rate"] == pytest.approx(1.0)

    def test_summary_of_group_frame(self, panel):
        ic = q.information_coefficient(
            panel, factor="factor", forward_return="fwd_ret", group="sector", min_assets=3
        )
        summary = q.ic_summary(ic)
        assert list(summary.columns) == ["A", "B"]


class TestDecayAndPersistence:
    def test_decay_table(self, panel):
        panel = panel.assign(fwd_ret_2=lambda d: -d["fwd_ret"])
        decay = q.factor_decay(panel, factor="factor", forward_returns=["fwd_ret", "fwd_ret_2"])
        assert list(decay.index) == ["fwd_ret", "fwd_ret_2"]
        assert decay.loc["fwd_ret", "mean"] == pytest.approx(1.0)
        assert decay.loc["fwd_ret_2", "mean"] == pytest.approx(-1.0)

    def test_decay_requires_columns(self, panel):
        with pytest.raises(ValidationError, match="forward_returns"):
            q.factor_decay(panel, factor="factor", forward_returns=[])

    def test_autocorrelation_of_static_signal_is_one(self, panel):
        auto = q.factor_autocorrelation(panel, factor="factor", lags=(1, 2))
        assert auto.loc[1] == pytest.approx(1.0)
        assert auto.loc[2] == pytest.approx(1.0)


class TestTurnover:
    def test_static_signal_has_zero_turnover(self, panel):
        turnover = q.quantile_turnover(panel, factor="factor")
        assert turnover.iloc[0].isna().all(), "最初の期間は定義できない"
        assert turnover.iloc[1:].to_numpy().ravel() == pytest.approx(0.0)

    def test_reversed_signal_has_full_turnover(self, panel):
        """毎期シグナルの符号が入れ替わると Q1 と Q5 が総入れ替えになる。"""
        flipping = panel.copy()
        flip = flipping["date"].isin(DATES[1::2])
        flipping.loc[flip, "factor"] = -flipping.loc[flip, "factor"]

        turnover = q.quantile_turnover(flipping, factor="factor")
        assert turnover["Q1"].iloc[1] == pytest.approx(1.0)
        assert turnover["Q5"].iloc[1] == pytest.approx(1.0)

    def test_transition_matrix_rows_sum_to_one(self, panel):
        matrix = q.quantile_transition_matrix(panel, factor="factor")
        assert list(matrix.index) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
        assert matrix.sum(axis=1).to_numpy() == pytest.approx(np.ones(5))

    def test_transition_matrix_static_signal_is_diagonal(self, panel):
        matrix = q.quantile_transition_matrix(panel, factor="factor")
        assert np.diag(matrix.to_numpy()) == pytest.approx(np.ones(5))

    def test_transition_needs_two_periods(self, panel):
        one_day = panel[panel["date"] == DATES[0]]
        with pytest.raises(ValidationError, match="2 期間以上"):
            q.quantile_transition_matrix(one_day, factor="factor")
