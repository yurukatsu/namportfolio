"""例外。細かく分けず、`except ValueError` でも捕まるようにしておく。"""

from __future__ import annotations

__all__ = ["NamPortfolioError", "ValidationError"]


class NamPortfolioError(Exception):
    """本パッケージが送出する例外の基底。"""


class ValidationError(NamPortfolioError, ValueError):
    """入力データが要件を満たさない。"""
