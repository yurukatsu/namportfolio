"""図。matplotlib のみを使い、すべての関数は Figure を返す（``show()`` はしない）。

matplotlib は optional 依存なので、無い環境では ``namportfolio`` 本体は動くが
このサブパッケージの import で分かりやすく失敗する。
"""

from __future__ import annotations

from ..core.errors import NamPortfolioError

try:
    import matplotlib  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise NamPortfolioError(
        "描画には matplotlib が必要です。\n"
        "  pip install 'namportfolio[viz]'  もしくは  pip install matplotlib"
    ) from exc

from . import plotly, theme
from .attribution import (
    plot_cumulative_effects,
    plot_effect_heatmap,
    plot_effects_by_segment,
    plot_waterfall,
)
from .holdings import (
    plot_allocation,
    plot_characteristics,
    plot_concentration,
    plot_contribution,
    plot_turnover,
)
from .performance import (
    plot_annual_returns,
    plot_cumulative_returns,
    plot_drawdown,
    plot_monthly_heatmap,
    plot_return_distribution,
    plot_rolling,
)
from .quantile import (
    plot_double_sort,
    plot_factor_decay,
    plot_ic,
    plot_ic_heatmap,
    plot_quantile_cumulative,
    plot_quantile_returns,
    plot_quantile_turnover,
    plot_transition_matrix,
)
from .risk import (
    plot_bias_statistic,
    plot_exposures,
    plot_factor_contribution,
    plot_risk_contribution,
    plot_risk_decomposition,
    plot_risk_forecast,
)
from .signals import (
    plot_coverage,
    plot_distribution,
    plot_distribution_stats,
    plot_explained_ratio,
    plot_signal_correlation,
    plot_signal_exposure,
)
from .stats import (
    plot_bootstrap,
    plot_regime,
    plot_subsample,
)

__all__ = [
    "theme",
    "plotly",
    # F4 リターン評価
    "plot_cumulative_returns",
    "plot_drawdown",
    "plot_monthly_heatmap",
    "plot_annual_returns",
    "plot_rolling",
    "plot_return_distribution",
    # F2 分位分析
    "plot_quantile_returns",
    "plot_quantile_cumulative",
    "plot_double_sort",
    "plot_ic",
    "plot_ic_heatmap",
    "plot_factor_decay",
    "plot_quantile_turnover",
    "plot_transition_matrix",
    # F1 シグナル診断
    "plot_coverage",
    "plot_distribution",
    "plot_distribution_stats",
    "plot_signal_correlation",
    "plot_signal_exposure",
    "plot_explained_ratio",
    # F3 保有分析
    "plot_allocation",
    "plot_concentration",
    "plot_contribution",
    "plot_characteristics",
    "plot_turnover",
    # F5 Brinson 帰属
    "plot_waterfall",
    "plot_effects_by_segment",
    "plot_cumulative_effects",
    "plot_effect_heatmap",
    # F6 / F7 Barra リスク
    "plot_exposures",
    "plot_risk_decomposition",
    "plot_risk_contribution",
    "plot_factor_contribution",
    "plot_risk_forecast",
    "plot_bias_statistic",
    # F8 統計的頑健性
    "plot_subsample",
    "plot_regime",
    "plot_bootstrap",
]
