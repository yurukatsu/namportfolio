import numpy as np
import pandas as pd
import pytest

from namportfolio import attribution as attr
from namportfolio.core.errors import ValidationError
from namportfolio.performance import MONTH_END

DATES = pd.date_range("2024-01-31", periods=2, freq=MONTH_END)

# 期間 1: A は買い持ち・銘柄選択も良い / B は減らして選択は悪い
#   rb_total = 0.5*0.08 + 0.5*0.04 = 0.06
#   rp_total = 0.6*0.10 + 0.4*0.02 = 0.068  -> active = 0.008
SEGMENT_ROWS = [
    {"date": DATES[0], "sector": "A", "wp": 0.6, "wb": 0.5, "rp": 0.10, "rb": 0.08},
    {"date": DATES[0], "sector": "B", "wp": 0.4, "wb": 0.5, "rp": 0.02, "rb": 0.04},
    {"date": DATES[1], "sector": "A", "wp": 0.5, "wb": 0.5, "rp": -0.03, "rb": -0.02},
    {"date": DATES[1], "sector": "B", "wp": 0.5, "wb": 0.5, "rp": 0.05, "rb": 0.03},
]


@pytest.fixture
def segments():
    return pd.DataFrame(SEGMENT_ROWS)


@pytest.fixture
def assets():
    """銘柄レベル。セグメント A/B に 2 銘柄ずつ。"""
    rows = []
    for date, sector, wp, wb, rp, rb in [
        (r["date"], r["sector"], r["wp"], r["wb"], r["rp"], r["rb"]) for r in SEGMENT_ROWS
    ]:
        # 銘柄 1 と 2 のリターンを決め、ウェイト配分で rp / rb を再現する
        low, high = rb - 0.02, rb + 0.02
        # rp = a*high + (1-a)*low を満たす a
        share = (rp - low) / (high - low)
        rows.append(
            {
                "date": date,
                "bid": f"{sector}1",
                "sector": sector,
                "weight": wp * share,
                "bench_weight": wb * 0.5,
                "ret": high,
            }
        )
        rows.append(
            {
                "date": date,
                "bid": f"{sector}2",
                "sector": sector,
                "weight": wp * (1 - share),
                "bench_weight": wb * 0.5,
                "ret": low,
            }
        )
    return pd.DataFrame(rows)


def _effects(segments, **kwargs):
    return attr.brinson(
        segments,
        segment="sector",
        portfolio_weight="wp",
        benchmark_weight="wb",
        portfolio_return="rp",
        benchmark_return="rb",
        **kwargs,
    )


class TestAggregateSegments:
    def test_from_segment_level(self, segments):
        result = attr.aggregate_segments(
            segments,
            segment="sector",
            portfolio_weight="wp",
            benchmark_weight="wb",
            portfolio_return="rp",
            benchmark_return="rb",
        )
        assert list(result.columns) == ["wp", "wb", "rp", "rb"]
        assert result.loc[(DATES[0], "A"), "rp"] == pytest.approx(0.10)

    def test_from_asset_level(self, assets, segments):
        """銘柄から集計しても、セグメント直接指定と同じ量になる。"""
        from_assets = attr.aggregate_segments(assets, segment="sector", asset_return="ret")
        from_segments = attr.aggregate_segments(
            segments,
            segment="sector",
            portfolio_weight="wp",
            benchmark_weight="wb",
            portfolio_return="rp",
            benchmark_return="rb",
        )
        pd.testing.assert_frame_equal(from_assets, from_segments, check_like=True)

    def test_requires_exactly_one_route(self, segments):
        with pytest.raises(ValidationError, match="どちらか"):
            attr.aggregate_segments(segments, segment="sector")

    def test_rejects_both_routes(self, assets):
        with pytest.raises(ValidationError, match="どちらか"):
            attr.aggregate_segments(
                assets,
                segment="sector",
                asset_return="ret",
                portfolio_return="ret",
                benchmark_return="ret",
            )


class TestBrinsonEffects:
    def test_bf_by_hand(self, segments):
        result = _effects(segments)
        first = result.xs(DATES[0], level=0)
        # (0.6-0.5) * (0.08-0.06)
        assert first.loc["A", "allocation"] == pytest.approx(0.002)
        # (0.4-0.5) * (0.04-0.06)
        assert first.loc["B", "allocation"] == pytest.approx(0.002)
        assert first.loc["A", "selection"] == pytest.approx(0.5 * 0.02)
        assert first.loc["A", "interaction"] == pytest.approx(0.1 * 0.02)

    def test_bhb_by_hand(self, segments):
        result = _effects(segments, model="bhb")
        first = result.xs(DATES[0], level=0)
        assert first.loc["A", "allocation"] == pytest.approx(0.1 * 0.08)
        assert first.loc["B", "allocation"] == pytest.approx(-0.1 * 0.04)

    def test_sums_to_active_return(self, segments):
        """どのモデル・項数でも、期間の総和はアクティブリターンに一致する。"""
        totals = attr.total_returns(
            segments,
            segment="sector",
            portfolio_weight="wp",
            benchmark_weight="wb",
            portfolio_return="rp",
            benchmark_return="rb",
        )
        for model in ("bf", "bhb"):
            for terms in (2, 3):
                result = _effects(segments, model=model, terms=terms)
                by_date = result.groupby(level=0).sum().sum(axis=1)
                assert by_date.to_numpy() == pytest.approx(totals["active"].to_numpy())

    def test_two_term_folds_interaction(self, segments):
        result = _effects(segments, terms=2)
        assert list(result.columns) == ["allocation", "selection"]
        first = result.xs(DATES[0], level=0)
        assert first.loc["A", "selection"] == pytest.approx(0.6 * 0.02), "ポートウェイト基準"

    def test_model_validation(self, segments):
        with pytest.raises(ValidationError, match="model は"):
            _effects(segments, model="brinson")

    def test_terms_validation(self, segments):
        with pytest.raises(ValidationError, match="terms は"):
            _effects(segments, terms=4)

    def test_from_asset_level_matches(self, assets, segments):
        from_assets = attr.brinson(assets, segment="sector", asset_return="ret")
        pd.testing.assert_frame_equal(from_assets, _effects(segments), check_like=True)


class TestTotalReturns:
    def test_values(self, segments):
        result = attr.total_returns(
            segments,
            segment="sector",
            portfolio_weight="wp",
            benchmark_weight="wb",
            portfolio_return="rp",
            benchmark_return="rb",
        )
        assert result.loc[DATES[0], "portfolio"] == pytest.approx(0.068)
        assert result.loc[DATES[0], "benchmark"] == pytest.approx(0.06)
        assert result.loc[DATES[0], "active"] == pytest.approx(0.008)


class TestLinking:
    @pytest.fixture
    def parts(self, segments):
        effects = _effects(segments)
        totals = attr.total_returns(
            segments,
            segment="sector",
            portfolio_weight="wp",
            benchmark_weight="wb",
            portfolio_return="rp",
            benchmark_return="rb",
        )
        geometric = float((1 + totals["portfolio"]).prod() - (1 + totals["benchmark"]).prod())
        return effects, totals, geometric

    @pytest.mark.parametrize("method", ["carino", "grap", "frongello"])
    def test_sums_to_geometric_active_return(self, parts, method):
        effects, totals, geometric = parts
        linked = attr.link(effects, totals, method=method)
        assert linked.to_numpy().sum() == pytest.approx(geometric)

    def test_simple_does_not_match_geometric(self, parts):
        effects, totals, geometric = parts
        linked = attr.link(effects, totals, method="simple")
        assert linked.to_numpy().sum() != pytest.approx(geometric, abs=1e-9)
        pd.testing.assert_frame_equal(linked, effects)

    def test_preserves_shape(self, parts):
        effects, totals, _ = parts
        linked = attr.link(effects, totals, method="carino")
        assert linked.index.equals(effects.index)
        assert list(linked.columns) == list(effects.columns)

    def test_unknown_method(self, parts):
        effects, totals, _ = parts
        with pytest.raises(ValidationError, match="method は"):
            attr.link(effects, totals, method="geometric")

    def test_identical_returns_do_not_divide_by_zero(self):
        """ポートとベンチが同一でも係数が発散しない（極限値を使う）。"""
        rows = [
            {"date": DATES[0], "sector": "A", "wp": 0.5, "wb": 0.5, "rp": 0.05, "rb": 0.05},
            {"date": DATES[1], "sector": "A", "wp": 0.5, "wb": 0.5, "rp": 0.05, "rb": 0.05},
        ]
        frame = pd.DataFrame(rows)
        effects = _effects(frame)
        totals = attr.total_returns(
            frame,
            segment="sector",
            portfolio_weight="wp",
            benchmark_weight="wb",
            portfolio_return="rp",
            benchmark_return="rb",
        )
        linked = attr.link(effects, totals, method="carino")
        assert np.isfinite(linked.to_numpy()).all()


class TestSummary:
    def test_shape_and_total(self, segments):
        summary = attr.brinson_summary(
            segments,
            segment="sector",
            portfolio_weight="wp",
            benchmark_weight="wb",
            portfolio_return="rp",
            benchmark_return="rb",
        )
        assert list(summary.index) == ["A", "B", "Total"]
        assert "total" in summary.columns
        assert summary.loc["Total", "total"] == pytest.approx(
            summary.loc[["A", "B"], "total"].sum()
        )

    def test_total_matches_geometric_active(self, segments):
        totals = attr.total_returns(
            segments,
            segment="sector",
            portfolio_weight="wp",
            benchmark_weight="wb",
            portfolio_return="rp",
            benchmark_return="rb",
        )
        geometric = float((1 + totals["portfolio"]).prod() - (1 + totals["benchmark"]).prod())
        summary = attr.brinson_summary(
            segments,
            segment="sector",
            portfolio_weight="wp",
            benchmark_weight="wb",
            portfolio_return="rp",
            benchmark_return="rb",
        )
        assert summary.loc["Total", "total"] == pytest.approx(geometric)

    def test_from_asset_level(self, assets):
        summary = attr.brinson_summary(assets, segment="sector", asset_return="ret")
        assert list(summary.index) == ["A", "B", "Total"]
