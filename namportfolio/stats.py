"""統計的有意性の検定。

いまは分位分析と performance が使う t 値だけ。サブサンプル分析・レジーム別分析・
多重検定補正は必要になった時点で足す。

IC やアクティブリターンの系列は自己相関を持つことが多く、素の t 値は有意性を
過大評価する。:func:`newey_west_tstat` が自己相関を補正した t 値を返す。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["t_statistic", "newey_west_tstat", "newey_west_lags"]


def _clean(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[~np.isnan(array)]


def t_statistic(values: Sequence[float] | pd.Series, *, mu: float = 0.0) -> float:
    """平均が ``mu`` と異なるかの t 値。

    系列に自己相関がある場合は過大評価になる。その場合は
    :func:`newey_west_tstat` を使う。
    """
    x = _clean(values)
    n = len(x)
    if n < 2:
        return np.nan
    std = x.std(ddof=1)
    if std == 0:
        return np.nan
    return float((x.mean() - mu) / (std / np.sqrt(n)))


def newey_west_lags(n_obs: int) -> int:
    """標本数から自動でラグ数を決める（Newey-West 1994 の経験則）。

    ``floor(4 * (n / 100) ** (2/9))``
    """
    if n_obs < 2:
        return 0
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def newey_west_tstat(
    values: Sequence[float] | pd.Series,
    *,
    lags: int | None = None,
    mu: float = 0.0,
) -> float:
    """自己相関に頑健な t 値（Newey-West、Bartlett カーネル）。

    Parameters
    ----------
    values :
        平均を検定したい系列（IC、アクティブリターンなど）。
    lags :
        考慮する自己相関のラグ数。``None`` なら :func:`newey_west_lags` で自動決定。
    mu :
        帰無仮説の平均。

    Returns
    -------
    float
        ラグ 0 なら通常の t 値と一致する。分散推定が非正になった場合は ``NaN``。
    """
    x = _clean(values)
    n = len(x)
    if n < 2:
        return np.nan
    if lags is None:
        lags = newey_west_lags(n)
    lags = min(lags, n - 1)

    resid = x - x.mean()
    variance = float(resid @ resid) / n  # γ0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)  # Bartlett
        cov = float(resid[lag:] @ resid[:-lag]) / n
        variance += 2.0 * weight * cov

    if variance <= 0:
        return np.nan
    return float((x.mean() - mu) / np.sqrt(variance / n))
