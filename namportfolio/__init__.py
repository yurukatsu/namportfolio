"""namportfolio — アクティブ運用戦略のパフォーマンス評価・可視化。"""

import importlib

from . import performance, quantile, stats
from .core import (
    NamPortfolioError,
    ValidationError,
    as_wide,
    infer_periods_per_year,
    set_columns,
    to_long,
)
from .performance import performance_summary

__version__ = "0.1.0"


def __getattr__(name: str):
    """``viz`` は matplotlib に依存するので、実際に触られるまで import しない。

    matplotlib が無い環境でも ``import namportfolio`` は成功し、``npf.viz`` に
    触れた時点で分かりやすいエラーになる。

    ``from . import viz`` は親モジュールの属性探索を経由してこの関数を再帰的に
    呼ぶため、``import_module`` でサブモジュールを直接読み込む。
    """
    if name == "viz":
        module = importlib.import_module(".viz", __name__)
        globals()["viz"] = module  # 次回以降は __getattr__ を通らない
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    # 機能モジュール（npf.performance.sharpe_ratio(...) のように使う）
    "performance",
    "quantile",
    "stats",
    "viz",
    # 基盤
    "set_columns",
    "as_wide",
    "to_long",
    "infer_periods_per_year",
    # 最頻出
    "performance_summary",
    # 例外
    "NamPortfolioError",
    "ValidationError",
]
