"""統計的頑健性の検証。

「その結果を信じてよいか」を確かめる。4 つの角度がある。

============== ============================================================
観点            関数
============== ============================================================
自己相関         :func:`newey_west_tstat`（素の t 値は有意性を過大評価する）
分布の仮定       :func:`bootstrap_ci`（正規性を仮定しない区間推定）
時間の安定性     :func:`subsample` / :func:`stability`
局面依存         :func:`regime_summary` / :func:`make_regimes`
試行回数         :func:`deflated_sharpe_ratio`（多数試して選んだバイアス）
============== ============================================================

正規分布の分位点には標準ライブラリの :class:`statistics.NormalDist` を使う
（scipy を要求しない）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import performance as perf
from .core.errors import ValidationError
from .core.frequency import resolve_periods_per_year, volatility_scale

__all__ = [
    # 有意性
    "t_statistic",
    "newey_west_tstat",
    "newey_west_lags",
    # ブートストラップ
    "bootstrap_distribution",
    "bootstrap_ci",
    "bootstrap_pvalue",
    # 安定性
    "subsample",
    "stability",
    # 局面
    "make_regimes",
    "regime_summary",
    # 多重検定
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
]

_NORMAL = NormalDist()

#: オイラー・マスケローニ定数（最大値の期待値に現れる）。
_EULER_MASCHERONI = 0.5772156649015329


def _clean(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    return array[~np.isnan(array)]


# --------------------------------------------------------------------------
# 有意性
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# ブートストラップ
# --------------------------------------------------------------------------


def bootstrap_distribution(
    values: pd.Series | Sequence[float],
    *,
    statistic: str = "mean",
    n_boot: int = 10_000,
    block_length: int | None = None,
    periods_per_year: float | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """ブロック・ブートストラップで統計量の標本分布を作る。

    **1 点ずつの復元抽出（IID ブートストラップ）は使わない。** リターンも IC も
    自己相関を持つので、バラバラに抽出すると依存構造が壊れて区間が狭く出る。
    連続した ``block_length`` 個をひと塊として抽出することで、系列相関を保つ。

    Parameters
    ----------
    statistic :
        ``"mean"`` / ``"sharpe_ratio"`` / ``"volatility"`` / ``"hit_rate"``。
    block_length :
        ブロックの長さ。``None`` なら経験則 :math:`\\lceil T^{1/3} \\rceil`。
        自己相関が強い系列では長めにする。
    n_boot :
        リサンプル回数。``n_boot × T`` の配列を作るので、日次データでは
        メモリを見ながら減らすこと。
    seed :
        乱数種。既定は固定（結果が再現する）。``None`` で毎回変わる。

    Returns
    -------
    np.ndarray
        長さ ``n_boot`` の統計量の配列。
    """
    if statistic not in _BOOTSTRAP_STATISTICS:
        raise ValidationError(
            f"statistic は {sorted(_BOOTSTRAP_STATISTICS)} のいずれかです: {statistic!r}"
        )
    if n_boot < 1:
        raise ValidationError(f"n_boot は 1 以上です: {n_boot}")

    x = _clean(values)
    n_obs = len(x)
    if n_obs < 2:
        return np.full(n_boot, np.nan)

    ppy = _resolve_ppy(values, periods_per_year, statistic)
    length = block_length or max(1, round(n_obs ** (1 / 3)))
    length = min(length, n_obs)

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n_obs / length)
    starts = rng.integers(0, n_obs - length + 1, size=(n_boot, n_blocks))
    offsets = np.arange(length)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n_obs]

    return _BOOTSTRAP_STATISTICS[statistic](x[indices], ppy)


def bootstrap_ci(
    values: pd.Series | Sequence[float],
    *,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_boot: int = 10_000,
    block_length: int | None = None,
    periods_per_year: float | None = None,
    seed: int | None = 0,
) -> tuple[float, float]:
    """ブートストラップによる信頼区間（パーセンタイル法）。

    区間が 0 をまたがなければ、その統計量は 0 と有意に異なる。

    Returns
    -------
    tuple[float, float]
        ``(下限, 上限)``。
    """
    if not 0.0 < confidence < 1.0:
        raise ValidationError(f"confidence は 0 と 1 の間です: {confidence}")

    distribution = bootstrap_distribution(
        values,
        statistic=statistic,
        n_boot=n_boot,
        block_length=block_length,
        periods_per_year=periods_per_year,
        seed=seed,
    )
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.nanpercentile(distribution, [alpha * 100, (1 - alpha) * 100])
    return float(lower), float(upper)


def bootstrap_pvalue(
    values: pd.Series | Sequence[float],
    *,
    statistic: str = "mean",
    null_value: float = 0.0,
    n_boot: int = 10_000,
    block_length: int | None = None,
    periods_per_year: float | None = None,
    seed: int | None = 0,
) -> float:
    """ブートストラップ分布のうち、帰無値の反対側に落ちた割合（両側）。

    観測された統計量が正なら「分布のうち ``null_value`` 以下の割合」を 2 倍する。
    """
    distribution = bootstrap_distribution(
        values,
        statistic=statistic,
        n_boot=n_boot,
        block_length=block_length,
        periods_per_year=periods_per_year,
        seed=seed,
    )
    valid = distribution[~np.isnan(distribution)]
    if len(valid) == 0:
        return np.nan

    centre = float(np.median(valid))
    tail = (valid <= null_value).mean() if centre > null_value else (valid >= null_value).mean()
    return float(min(1.0, 2.0 * tail))


# --------------------------------------------------------------------------
# 時間の安定性
# --------------------------------------------------------------------------


def subsample(
    values: pd.Series,
    *,
    n_splits: int = 2,
    statistic: str = "mean",
    periods_per_year: float | None = None,
) -> pd.DataFrame:
    """期間を等分し、区間ごとに統計量を出す。

    全期間で有意でも、特定の局面だけで稼いでいることがある。前半・後半
    （``n_splits=2``）が最も基本で、4 分割にすると崩れた時期が見える。

    Returns
    -------
    pd.DataFrame
        ``index`` は区間番号（1 始まり）、``columns=[start, end, n_periods,
        <statistic>, t_stat]``。
    """
    if n_splits < 2:
        raise ValidationError(f"n_splits は 2 以上です: {n_splits}")
    if statistic not in _SERIES_STATISTICS:
        raise ValidationError(
            f"statistic は {sorted(_SERIES_STATISTICS)} のいずれかです: {statistic!r}"
        )

    series = values.dropna()
    if len(series) < n_splits:
        raise ValidationError(f"{n_splits} 分割するには {n_splits} 点以上必要です。")

    ppy = resolve_periods_per_year(series, periods_per_year)
    rows = {}
    # np.array_split は Series を ndarray に変換してしまうので、位置で分けて iloc を取る
    for index, positions in enumerate(np.array_split(np.arange(len(series)), n_splits), start=1):
        chunk = series.iloc[positions]
        rows[index] = {
            "start": chunk.index[0],
            "end": chunk.index[-1],
            "n_periods": len(chunk),
            statistic: _SERIES_STATISTICS[statistic](chunk, ppy),
            "t_stat": t_statistic(chunk),
        }
    frame = pd.DataFrame(rows).T
    frame.index.name = "split"
    return frame


def stability(
    values: pd.Series,
    *,
    n_splits: int = 4,
    statistic: str = "mean",
    periods_per_year: float | None = None,
) -> pd.Series:
    """サブサンプル間で結果がどれだけ揃っているか。

    Returns
    -------
    pd.Series
        - ``sign_agreement`` 全期間と符号が一致した区間の割合
        - ``positive_ratio`` 統計量が正だった区間の割合
        - ``dispersion`` 区間ごとの標準偏差 ÷ |平均|（変動係数）
        - ``min`` / ``max`` 区間ごとの最小・最大
    """
    table = subsample(
        values, n_splits=n_splits, statistic=statistic, periods_per_year=periods_per_year
    )
    parts = pd.to_numeric(table[statistic]).dropna()
    ppy = resolve_periods_per_year(values.dropna(), periods_per_year)
    overall = _SERIES_STATISTICS[statistic](values.dropna(), ppy)

    mean = parts.mean()
    return pd.Series(
        {
            "sign_agreement": float((np.sign(parts) == np.sign(overall)).mean()),
            "positive_ratio": float((parts > 0).mean()),
            "dispersion": float(parts.std(ddof=1) / abs(mean)) if mean else np.nan,
            "min": float(parts.min()),
            "max": float(parts.max()),
        },
        name=statistic,
    )


# --------------------------------------------------------------------------
# 局面（レジーム）
# --------------------------------------------------------------------------


def make_regimes(
    reference: pd.Series,
    *,
    method: str = "direction",
    window: int = 12,
    n_quantiles: int = 2,
    labels: Sequence[str] | None = None,
) -> pd.Series:
    """よく使う局面分けを作る。

    レジーム定義は本来データ側の話なので、外から ``regimes`` を渡すのが基本。
    ここでは頻出のものだけ用意する。

    Parameters
    ----------
    reference :
        基準になる系列（ベンチマークリターン、金利、VIX など）。
    method :
        - ``"direction"``: 符号で ``up`` / ``down`` に分ける
        - ``"volatility"``: ``window`` 期のローリング標準偏差を分位で分ける
        - ``"quantile"``: ``reference`` 自体を分位で分ける
    n_quantiles :
        ``"volatility"`` / ``"quantile"`` での分位数。
    labels :
        分位に付ける名前。``None`` なら ``Q1`` … または ``low`` / ``high``。

    Returns
    -------
    pd.Series
        ``reference`` と同じ index のラベル。判定できない期間は ``NaN``。
    """
    if method not in _REGIME_METHODS:
        raise ValidationError(f"method は {sorted(_REGIME_METHODS)} のいずれかです: {method!r}")

    if method == "direction":
        return pd.Series(
            np.where(reference > 0, "up", "down"), index=reference.index, name="regime"
        ).where(reference.notna())

    base = reference.rolling(window).std() if method == "volatility" else reference
    if n_quantiles < 2:
        raise ValidationError(f"n_quantiles は 2 以上です: {n_quantiles}")

    names = list(labels) if labels is not None else _default_regime_labels(n_quantiles, method)
    if len(names) != n_quantiles:
        raise ValidationError(f"labels は {n_quantiles} 個必要です: {names}")

    ranks = base.rank(pct=True)
    bins = np.ceil(ranks * n_quantiles).clip(1, n_quantiles)
    return pd.Series(
        [names[int(b) - 1] if not np.isnan(b) else np.nan for b in bins],
        index=reference.index,
        name="regime",
    )


def regime_summary(
    values: pd.Series,
    regimes: pd.Series,
    *,
    statistic: str = "mean",
    periods_per_year: float | None = None,
    regime_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """局面ごとの統計量。

    「シグナルが効かなくなった」のか「効かない局面が続いた」のかを分ける。

    Returns
    -------
    pd.DataFrame
        ``index`` は局面ラベル、``columns=[n_periods, <statistic>, t_stat, share]``。
        ``share`` はその局面が占める期間の割合。
    """
    if statistic not in _SERIES_STATISTICS:
        raise ValidationError(
            f"statistic は {sorted(_SERIES_STATISTICS)} のいずれかです: {statistic!r}"
        )

    aligned, labels = values.align(regimes, join="inner")
    valid = aligned.notna() & labels.notna()
    aligned, labels = aligned[valid], labels[valid]
    if aligned.empty:
        raise ValidationError("値と局面ラベルの重なりがありません。")

    ppy = resolve_periods_per_year(aligned, periods_per_year)
    rows = {}
    for name, chunk in aligned.groupby(labels, observed=True):
        rows[name] = {
            "n_periods": len(chunk),
            statistic: _SERIES_STATISTICS[statistic](chunk, ppy),
            "t_stat": t_statistic(chunk),
            "share": len(chunk) / len(aligned),
        }

    frame = pd.DataFrame(rows).T
    if regime_order is not None:
        frame = frame.reindex(list(regime_order))
    frame.index.name = "regime"
    return frame


# --------------------------------------------------------------------------
# 多重検定
# --------------------------------------------------------------------------


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    *,
    benchmark_sharpe: float = 0.0,
    periods_per_year: float | None = None,
) -> float:
    """観測シャープが閾値を上回っている確率（Bailey & López de Prado 2014）。

    .. math::

        \\widehat{PSR}(SR^*) = \\Phi\\!\\left(
            \\frac{(\\hat{SR}-SR^*)\\sqrt{T-1}}
                 {\\sqrt{1-\\gamma_3\\hat{SR}+\\frac{\\gamma_4-1}{4}\\hat{SR}^2}}
        \\right)

    分母がシャープレシオの標準誤差で、**負に歪んで裾が重いほど大きくなる**
    （＝有意になりにくい）。同じシャープでも分布の形で評価が変わる。

    Parameters
    ----------
    benchmark_sharpe :
        比較する閾値。**年率**で渡す（内部で期間シャープに直す）。
    periods_per_year :
        年率化の係数。``None`` なら日付から推定。

    Returns
    -------
    float
        0〜1 の確率。0.95 を超えれば「閾値より高いと言ってよい」水準。
    """
    x = _clean(returns)
    n_obs = len(x)
    if n_obs < 3:
        return np.nan

    ppy = resolve_periods_per_year(returns, periods_per_year)
    std = x.std(ddof=1)
    if std == 0:
        return np.nan

    observed = x.mean() / std  # 期間シャープ
    threshold = benchmark_sharpe / volatility_scale(ppy)

    series = pd.Series(x)
    skewness = float(series.skew())
    # pandas の kurtosis は超過尖度（正規分布で 0）。式が要求するのは尖度そのもの
    kurtosis = float(series.kurtosis()) + 3.0

    variance = 1.0 - skewness * observed + (kurtosis - 1.0) / 4.0 * observed**2
    if variance <= 0:
        return np.nan
    return float(_NORMAL.cdf((observed - threshold) * np.sqrt(n_obs - 1) / np.sqrt(variance)))


def expected_max_sharpe(n_trials: int, *, variance: float) -> float:
    """``n_trials`` 回試したときに偶然得られる最大シャープの期待値。

    .. math::

        SR^* = \\sqrt{V}\\left[(1-\\gamma)\\,\\Phi^{-1}\\!\\left(1-\\tfrac{1}{N}\\right)
               + \\gamma\\,\\Phi^{-1}\\!\\left(1-\\tfrac{1}{Ne}\\right)\\right]

    Parameters
    ----------
    n_trials :
        独立な試行回数。
    variance :
        試行間のシャープレシオの分散（年率シャープを渡したなら年率ベース）。
    """
    if n_trials < 2:
        raise ValidationError(f"n_trials は 2 以上です: {n_trials}")
    if variance < 0:
        raise ValidationError(f"variance は非負です: {variance}")

    first = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    second = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(
        np.sqrt(variance) * ((1.0 - _EULER_MASCHERONI) * first + _EULER_MASCHERONI * second)
    )


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    trials: Sequence[float] | None = None,
    n_trials: int | None = None,
    trial_variance: float | None = None,
    periods_per_year: float | None = None,
) -> float:
    """試行回数を考慮したシャープレシオの有意性。

    100 通り試して最良を選べば、シャープ 1.5 程度は偶然でも出る。閾値を
    「その試行回数で偶然得られる最大シャープの期待値」に置き換えて評価する。

    Parameters
    ----------
    trials :
        試したすべての戦略の**年率**シャープ。渡せば試行回数と分散を導出する。
    n_trials, trial_variance :
        ``trials`` が無い場合に直接指定する。

    Returns
    -------
    float
        0〜1。**0.95 を超えれば、試行回数を考慮しても有意**と判断する。

    Notes
    -----
    **``n_trials`` は正直に申告しないと意味がない。** パラメータを変えて試した
    回数、捨てた戦略もすべて数に入れる。少なく申告すれば数字は良くなるが、
    それは自分を欺いているだけになる。
    """
    if trials is not None:
        values = _clean(trials)
        if len(values) < 2:
            raise ValidationError("trials は 2 つ以上の値が必要です。")
        n_trials = len(values)
        trial_variance = float(values.var(ddof=1))
    if n_trials is None or trial_variance is None:
        raise ValidationError(
            "trials を渡すか、n_trials と trial_variance の両方を指定してください。"
        )

    threshold = expected_max_sharpe(n_trials, variance=trial_variance)
    return probabilistic_sharpe_ratio(
        returns, benchmark_sharpe=threshold, periods_per_year=periods_per_year
    )


# --------------------------------------------------------------------------
# 内部ヘルパー
# --------------------------------------------------------------------------

#: ブートストラップで使う統計量。(n_boot, T) の行列を受けて長さ n_boot の配列を返す。
_BOOTSTRAP_STATISTICS = {
    "mean": lambda samples, ppy: samples.mean(axis=1),
    "volatility": lambda samples, ppy: samples.std(axis=1, ddof=1) * np.sqrt(ppy),
    "sharpe_ratio": lambda samples, ppy: (
        samples.mean(axis=1) / _safe(samples.std(axis=1, ddof=1)) * np.sqrt(ppy)
    ),
    "hit_rate": lambda samples, ppy: (samples > 0).mean(axis=1),
}

#: サブサンプル・局面分析で使う統計量。
_SERIES_STATISTICS = {
    "mean": lambda series, ppy: float(series.mean()),
    "annualized_return": lambda series, ppy: float(
        perf.annualized_return(series, periods_per_year=ppy)
    ),
    "volatility": lambda series, ppy: float(
        perf.annualized_volatility(series, periods_per_year=ppy)
    ),
    "sharpe_ratio": lambda series, ppy: float(perf.sharpe_ratio(series, periods_per_year=ppy)),
    "hit_rate": lambda series, ppy: float(perf.hit_rate(series)),
}

_REGIME_METHODS = {"direction", "volatility", "quantile"}

#: 年率化が要る統計量（頻度の推定が必要）。
_NEEDS_FREQUENCY = {"volatility", "sharpe_ratio", "annualized_return"}


def _safe(values: np.ndarray) -> np.ndarray:
    """0 除算を NaN にする。"""
    return np.where(values == 0, np.nan, values)


def _resolve_ppy(values, periods_per_year: float | None, statistic: str) -> float:
    """年率化が要る統計量のときだけ頻度を解決する。"""
    if statistic not in _NEEDS_FREQUENCY:
        return 1.0
    return resolve_periods_per_year(values, periods_per_year)


def _default_regime_labels(n_quantiles: int, method: str) -> list[str]:
    if n_quantiles == 2:
        return ["low", "high"] if method == "volatility" else ["bottom", "top"]
    if n_quantiles == 3:
        return ["low", "mid", "high"]
    return [f"Q{i}" for i in range(1, n_quantiles + 1)]
