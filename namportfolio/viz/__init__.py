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

from . import theme
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
    plot_factor_decay,
    plot_ic,
    plot_ic_heatmap,
    plot_quantile_cumulative,
    plot_quantile_returns,
    plot_quantile_turnover,
    plot_transition_matrix,
)
from .signals import (
    plot_coverage,
    plot_distribution,
    plot_distribution_stats,
    plot_signal_correlation,
)

__all__ = [
    "theme",
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
    # F3 保有分析
    "plot_allocation",
    "plot_concentration",
    "plot_contribution",
    "plot_characteristics",
    "plot_turnover",
]
