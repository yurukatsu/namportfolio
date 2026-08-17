import numpy as np
import pandas as pd
import pytest

from namportfolio import performance as perf
from namportfolio.core.errors import ValidationError
from namportfolio.viz import plotly as viz_plotly
from namportfolio.viz import theme

px = pytest.importorskip("plotly.express")


@pytest.fixture
def cumulative():
    idx = pd.date_range("2020-01-31", periods=48, freq=perf.MONTH_END)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "strategy": rng.normal(0.008, 0.03, 48),
            "benchmark": rng.normal(0.005, 0.035, 48),
        },
        index=idx,
    )
    return perf.cumulative_returns(frame)


class TestApplyTheme:
    def test_returns_same_figure(self, cumulative):
        fig = px.line(cumulative)
        assert viz_plotly.apply_theme(fig) is fig, "その場で変更して返す"

    def test_surface_matches_matplotlib(self, cumulative):
        fig = viz_plotly.apply_theme(px.line(cumulative))
        surface = theme.palette()["surface"]
        assert fig.layout.paper_bgcolor == surface
        assert fig.layout.plot_bgcolor == surface

    def test_recolors_traces(self, cumulative):
        """layout.colorway だけでは px の色を上書きできないので trace を塗り替える。"""
        fig = viz_plotly.apply_theme(px.line(cumulative))
        expected = theme.categorical_colors(2)
        assert [trace.line.color for trace in fig.data] == expected

    def test_recolor_can_be_disabled(self, cumulative):
        fig = px.line(cumulative)
        original = [trace.line.color for trace in fig.data]
        viz_plotly.apply_theme(fig, recolor=False)
        assert [trace.line.color for trace in fig.data] == original

    def test_ordinal_uses_single_hue(self, cumulative):
        frame = pd.DataFrame({f"Q{i}": cumulative["strategy"] * i for i in range(1, 6)})
        fig = viz_plotly.apply_theme(px.line(frame), ordinal=True)
        assert [trace.line.color for trace in fig.data] == theme.ordinal_colors(5)

    def test_percent_axis(self, cumulative):
        fig = viz_plotly.apply_theme(px.line(cumulative), percent_axis="y", decimals=1)
        assert fig.layout.yaxis.tickformat == ".1%"
        assert fig.layout.xaxis.tickformat is None

    def test_percent_both_axes(self, cumulative):
        fig = viz_plotly.apply_theme(px.line(cumulative), percent_axis="both")
        assert fig.layout.xaxis.tickformat == ".0%"
        assert fig.layout.yaxis.tickformat == ".0%"

    def test_labels_and_title(self, cumulative):
        fig = viz_plotly.apply_theme(
            px.line(cumulative), title="Cumulative return", xlabel="", ylabel="Return"
        )
        assert fig.layout.title.text == "Cumulative return"
        assert fig.layout.title.x == 0.0, "タイトルは左寄せ"
        assert fig.layout.yaxis.title.text == "Return"

    def test_no_vertical_grid(self, cumulative):
        fig = viz_plotly.apply_theme(px.line(cumulative))
        assert fig.layout.xaxis.showgrid is False
        assert fig.layout.yaxis.gridcolor == theme.palette()["grid"]

    def test_zero_line(self, cumulative):
        fig = viz_plotly.apply_theme(px.line(cumulative), zero_line=True)
        assert len(fig.layout.shapes) == 1

    def test_dark_mode(self, cumulative):
        fig = viz_plotly.apply_theme(px.line(cumulative), mode="dark")
        assert fig.layout.paper_bgcolor == theme.palette("dark")["surface"]

    def test_invalid_percent_axis(self, cumulative):
        with pytest.raises(ValidationError, match="percent_axis"):
            viz_plotly.apply_theme(px.line(cumulative), percent_axis="diagonal")


class TestScales:
    def test_color_sequence_matches_theme(self):
        assert viz_plotly.color_sequence(3) == theme.categorical_colors(3)
        assert viz_plotly.color_sequence(5, ordinal=True) == theme.ordinal_colors(5)

    def test_color_sequence_caps_at_eight(self):
        assert len(viz_plotly.color_sequence(20)) == 8

    def test_sequential_scale_format(self):
        scale = viz_plotly.sequential_scale()
        assert scale[0][0] == 0.0 and scale[-1][0] == 1.0
        assert all(isinstance(color, str) for _, color in scale)

    def test_diverging_scale_has_neutral_midpoint(self):
        scale = viz_plotly.diverging_scale()
        midpoint = next(color for position, color in scale if position == pytest.approx(0.5))
        assert midpoint == theme.palette()["neutral"]

    def test_diverging_scale_follows_positive_hue(self):
        scale = viz_plotly.diverging_scale()
        assert scale[-1][1] == theme.POLES[theme.POSITIVE_HUE], "正側が設定どおりの色"


class TestIntegration:
    def test_works_with_imshow(self, cumulative):
        table = perf.monthly_table(cumulative["strategy"]).drop(columns="year_total")
        fig = px.imshow(
            table,
            color_continuous_scale=viz_plotly.diverging_scale(),
            color_continuous_midpoint=0,
        )
        assert viz_plotly.apply_theme(fig, title="Monthly") is fig

    def test_works_with_bar(self, cumulative):
        values = cumulative.iloc[-1]
        fig = viz_plotly.apply_theme(px.bar(values))
        assert fig.data[0].marker.color == theme.categorical_colors(1)[0]
