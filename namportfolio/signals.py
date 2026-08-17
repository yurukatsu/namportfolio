"""シグナルの前処理と診断。

分位分析やポートフォリオ構築の前に、シグナルを整えて素性を確認する。

    df["value_z"] = npf.signals.standardize(df, factor="value")
    df["value_n"] = npf.signals.neutralize(df, factor="value_z", by=["sector", "log_mktcap"])

    npf.signals.coverage(df, factor="value")           # 有効銘柄数の推移
    npf.signals.distribution_summary(df, factor="value")  # 歪度・裾の広さ

前処理関数はすべて **``data`` と同じ index の Series** を返すので、そのまま列として
代入できる。処理の順序は呼び出し側が決める（Pipeline は作っていない）。典型的には
winsorize → standardize → neutralize の順。

.. rubric:: 横断面ごとに処理する

すべての前処理は日付ごと（``group`` を指定すればグループごと）の横断面で完結する。
過去のデータを参照しないので、先読みバイアスは入らない。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .core.config import resolve_columns
from .core.errors import ValidationError
from .core.panel import as_wide, require_columns
from .stats import newey_west_tstat, t_statistic

__all__ = [
    "winsorize",
    "clip_outliers",
    "standardize",
    "neutralize",
    "coverage",
    "distribution_summary",
    "signal_correlation",
    "rolling_signal_correlation",
    "factor_exposure",
    "explained_ratio",
    "exposure_summary",
]

_STANDARDIZE_METHODS = ("zscore", "rank")
_CORR_METHODS = ("spearman", "pearson")


# --------------------------------------------------------------------------
# 前処理
# --------------------------------------------------------------------------


def winsorize(
    data: pd.DataFrame,
    *,
    factor: str,
    lower: float = 0.01,
    upper: float = 0.99,
    group: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series:
    """分位で外れ値を丸める。

    値を捨てずに端に寄せる（``dropna`` と違って銘柄数が減らない）。

    Parameters
    ----------
    lower, upper :
        丸める分位。``0.01`` / ``0.99`` なら上下 1% を境界値で置き換える。

    Returns
    -------
    pd.Series
        ``data`` と同じ index。
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValidationError(f"0 <= lower < upper <= 1 が必要です: {lower}, {upper}")
    grouped = _grouped_factor(data, factor, group, date_col)
    low = grouped.transform("quantile", lower)
    high = grouped.transform("quantile", upper)
    return data[factor].clip(low, high).rename(f"{factor}_winsorized")


def clip_outliers(
    data: pd.DataFrame,
    *,
    factor: str,
    n_sigma: float = 3.0,
    group: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series:
    """平均 ± ``n_sigma`` × 標準偏差で丸める。

    分位基準（:func:`winsorize`）と違い、分布が正規に近い前提が要る。裾が重い
    シグナルでは平均と標準偏差自体が外れ値に引っ張られるので注意。
    """
    if n_sigma <= 0:
        raise ValidationError(f"n_sigma は正の値です: {n_sigma}")
    grouped = _grouped_factor(data, factor, group, date_col)
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return data[factor].clip(mean - n_sigma * std, mean + n_sigma * std).rename(f"{factor}_clipped")


def standardize(
    data: pd.DataFrame,
    *,
    factor: str,
    method: str = "zscore",
    group: str | None = None,
    weight: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series:
    """横断面を標準化する。

    Parameters
    ----------
    method :
        ``"zscore"`` は平均 0・標準偏差 1 に、``"rank"`` は順位を 0〜1 に写す。
        ``"rank"`` は分布の形を捨てるので外れ値に完全に頑健だが、値の大小の
        「幅」の情報も失う。
    weight :
        加重平均・加重標準偏差に使うカラム名（時価総額など）。``"zscore"`` のみ有効。
    """
    if method not in _STANDARDIZE_METHODS:
        raise ValidationError(f"method は {list(_STANDARDIZE_METHODS)} のいずれかです: {method!r}")
    date_col, _ = resolve_columns(date_col, id_col)

    if method == "rank":
        grouped = _grouped_factor(data, factor, group, date_col)
        return grouped.rank(pct=True).rename(f"{factor}_rank")

    if weight is None:
        grouped = _grouped_factor(data, factor, group, date_col)
        mean = grouped.transform("mean")
        std = grouped.transform("std")
        return ((data[factor] - mean) / std.replace(0, np.nan)).rename(f"{factor}_z")

    require_columns(data, [weight], context="standardize")
    keys = [date_col] + ([group] if group else [])
    result = pd.Series(np.nan, index=data.index, dtype=float, name=f"{factor}_z")
    for _, chunk in data.groupby(keys, sort=False, dropna=False):
        values = chunk[factor]
        weights = chunk[weight]
        valid = values.notna() & weights.notna() & (weights > 0)
        if valid.sum() < 2:
            continue
        w = weights[valid].to_numpy(dtype=float)
        x = values[valid].to_numpy(dtype=float)
        w = w / w.sum()
        mean = float(w @ x)
        std = float(np.sqrt(w @ (x - mean) ** 2))
        if std == 0:
            continue
        result.loc[values.index[valid]] = (x - mean) / std
    return result


def neutralize(
    data: pd.DataFrame,
    *,
    factor: str,
    by: str | Sequence[str],
    weight: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series:
    """指定した要因の影響を横断面回帰で取り除き、残差を返す。

    「このシグナルは業種の偏りやサイズの効果ではないか」を潰すために使う。

    Parameters
    ----------
    by :
        取り除く要因のカラム名。**カテゴリ列（業種など）はダミー化**され、
        数値列（log 時価総額など）はそのまま説明変数になる。
    weight :
        加重最小二乗に使うカラム名。大型株に合わせて中立化したい場合に指定する。

    Returns
    -------
    pd.Series
        回帰残差。定数項を含めるので平均は 0 付近になる。説明変数が欠損している
        銘柄や、観測数が説明変数の数に満たない日は ``NaN``。

    Notes
    -----
    カテゴリのダミーと定数項は完全に共線だが、``lstsq`` は最小ノルム解を返すので
    残差は一意に定まる（係数の解釈はしない前提）。
    """
    date_col, _ = resolve_columns(date_col, id_col)
    by = [by] if isinstance(by, str) else list(by)
    require_columns(data, [date_col, factor, *by], context="neutralize")
    if weight is not None:
        require_columns(data, [weight], context="neutralize")

    result = pd.Series(np.nan, index=data.index, dtype=float, name=f"{factor}_neutral")
    for _, chunk in data.groupby(date_col, sort=False, dropna=False):
        design = _design_matrix(chunk[by])
        valid = chunk[factor].notna() & design.notna().all(axis=1)
        if weight is not None:
            valid &= chunk[weight].notna() & (chunk[weight] > 0)
        if valid.sum() <= design.shape[1]:
            continue  # 自由度が無い

        y = chunk.loc[valid, factor].to_numpy(dtype=float)
        x = design.loc[valid].to_numpy(dtype=float)
        w = chunk.loc[valid, weight].to_numpy(dtype=float) if weight else None
        result.loc[chunk.index[valid]] = _ols_residual(y, x, w)
    return result


# --------------------------------------------------------------------------
# 診断
# --------------------------------------------------------------------------


def coverage(
    data: pd.DataFrame,
    *,
    factor: str,
    universe: pd.DataFrame | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """期間ごとの有効銘柄数と欠損率。

    シグナルが「いつから使えるか」「途中でデータが抜けていないか」を見る。
    分位分析の結果が特定期間だけおかしいとき、まずここを疑う。

    Parameters
    ----------
    universe :
        母集団の long DataFrame。渡すと被覆率（``coverage_rate``）を計算する。
        ``None`` なら ``data`` 自身の行数を母数にする。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns=[n_valid, n_total, missing_rate, coverage_rate]``。
    """
    date_col, id_col = resolve_columns(date_col, id_col)
    require_columns(data, [date_col, id_col, factor], context="coverage")

    dates = pd.to_datetime(data[date_col])
    grouped = data.groupby(dates, sort=True)[factor]
    n_valid = grouped.count()
    n_total = grouped.size()

    frame = pd.DataFrame(
        {
            "n_valid": n_valid,
            "n_total": n_total,
            "missing_rate": 1.0 - n_valid / n_total,
        }
    )
    if universe is None:
        frame["coverage_rate"] = n_valid / n_total
    else:
        require_columns(universe, [date_col, id_col], context="coverage")
        universe_size = universe.groupby(pd.to_datetime(universe[date_col]))[id_col].nunique()
        frame["coverage_rate"] = n_valid / universe_size.reindex(frame.index)
    frame.index.name = date_col
    return frame


def distribution_summary(
    data: pd.DataFrame,
    *,
    factor: str,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """期間ごとの分布の形。

    ``skew`` が大きく振れる、``kurtosis`` が跳ねる期間は外れ値が入っている可能性が
    高い。前処理の効き具合を前後で比べるのにも使う。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns=[mean, std, skew, kurtosis, min, p01, p50, p99, max]``。
    """
    date_col, _ = resolve_columns(date_col, id_col)
    require_columns(data, [date_col, factor], context="distribution_summary")

    grouped = data.groupby(pd.to_datetime(data[date_col]), sort=True)[factor]
    frame = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "std": grouped.std(),
            "skew": grouped.skew(),
            "kurtosis": grouped.apply(pd.Series.kurtosis),
            "min": grouped.min(),
            "p01": grouped.quantile(0.01),
            "p50": grouped.quantile(0.50),
            "p99": grouped.quantile(0.99),
            "max": grouped.max(),
        }
    )
    frame.index.name = date_col
    return frame


def factor_exposure(
    data: pd.DataFrame,
    *,
    factor: str,
    factors: Sequence[str],
    weight: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """シグナルを既存ファクターに回帰した係数（＝シグナルのファクター曝露）。

    **「このシグナルは結局モメンタムの焼き直しではないか」に答える。** IC が高くても
    既存のリスクファクターで説明できてしまうなら、新しい情報は持っていない。

    Parameters
    ----------
    factor :
        評価したいシグナルのカラム名。標準化済みの値を渡すと係数が比較しやすい。
    factors :
        説明変数にする既存ファクターのカラム名。カテゴリ列（業種など）を混ぜると
        自動でダミー化される。
    weight :
        加重最小二乗に使うカラム名。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns`` は説明変数（カテゴリ列は展開後の名前）。
        定数項は落として返す。

    See Also
    --------
    neutralize : 同じ回帰の**残差**（曝露を取り除いたシグナル）を返す。
    explained_ratio : 同じ回帰の決定係数。
    """
    coefficients, _ = _regress_by_date(
        data,
        factor=factor,
        factors=factors,
        weight=weight,
        date_col=date_col,
        id_col=id_col,
    )
    return coefficients


def explained_ratio(
    data: pd.DataFrame,
    *,
    factor: str,
    factors: Sequence[str],
    weight: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series:
    """シグナルの分散のうち、既存ファクターで説明される割合（決定係数）。

    1 に近いほど「既存ファクターの組み合わせで再現できてしまう」。0 に近ければ
    独自の情報を持っている。

    Returns
    -------
    pd.Series
        ``index=date`` の :math:`R^2`。
    """
    _, ratio = _regress_by_date(
        data,
        factor=factor,
        factors=factors,
        weight=weight,
        date_col=date_col,
        id_col=id_col,
    )
    return ratio


def exposure_summary(exposures: pd.DataFrame) -> pd.DataFrame:
    """ファクター曝露の時系列を要約する。

    曝露が期間を通じて**安定して**ゼロと異なるかを見る。ある月だけ大きく振れても
    シグナルの性質とは言えない。

    Returns
    -------
    pd.DataFrame
        ``index`` は説明変数、``columns=[mean, std, t_stat, t_stat_nw, hit_rate,
        n_periods]``。曝露も自己相関を持つので、判断には ``t_stat_nw`` を使う。
    """
    rows = {}
    for name in exposures.columns:
        series = exposures[name].dropna()
        rows[name] = {
            "mean": series.mean(),
            "std": series.std(ddof=1),
            "t_stat": t_statistic(series),
            "t_stat_nw": newey_west_tstat(series),
            "hit_rate": float((series > 0).mean()) if len(series) else np.nan,
            "n_periods": len(series),
        }
    return pd.DataFrame(rows).T


def signal_correlation(
    data: pd.DataFrame,
    *,
    factors: Sequence[str],
    method: str = "spearman",
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """シグナル間の横断面相関（期間平均）。

    相関が高いシグナルを両方使っても情報は増えない。合成する前に冗長性を見る。

    Returns
    -------
    pd.DataFrame
        ``factors`` × ``factors`` の相関行列。各期間で相関を取ってから平均する
        （全期間をプールすると、水準の時系列変動が相関に混ざる）。
    """
    _validate_corr(method, factors)
    date_col, _ = resolve_columns(date_col, id_col)
    require_columns(data, [date_col, *factors], context="signal_correlation")

    dates = pd.to_datetime(data[date_col])
    values = data[list(factors)]
    if method == "spearman":
        # pandas の corr(method="spearman") は scipy を要求する。
        # 横断面を順位に変換してから Pearson を取れば結果は同じ
        values = values.groupby(dates, sort=True).rank()

    per_date = values.groupby(dates, sort=True).corr()
    mean = per_date.groupby(level=-1, sort=False).mean()
    return mean.reindex(index=list(factors), columns=list(factors))


def rolling_signal_correlation(
    data: pd.DataFrame,
    *,
    factors: Sequence[str],
    method: str = "spearman",
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series:
    """2 つのシグナルの横断面相関の推移。

    シグナル同士の関係が時期によって変わるかを見る。``factors`` は 2 つだけ渡す。
    """
    _validate_corr(method, factors)
    if len(factors) != 2:
        raise ValidationError(f"factors は 2 つ渡してください: {list(factors)}")
    date_col, id_col = resolve_columns(date_col, id_col)

    left = as_wide(data, factors[0], date_col=date_col, id_col=id_col)
    right = as_wide(data, factors[1], date_col=date_col, id_col=id_col)
    left, right = left.align(right, join="inner")
    if method == "spearman":
        left = left.rank(axis=1)
        right = right.rank(axis=1)
    return left.corrwith(right, axis=1).rename(f"{factors[0]}~{factors[1]}")


# --------------------------------------------------------------------------
# 内部ヘルパー
# --------------------------------------------------------------------------


def _grouped_factor(data: pd.DataFrame, factor: str, group: str | None, date_col: str | None):
    """日付（＋グループ）で切った factor 列の GroupBy を返す。"""
    date_col, _ = resolve_columns(date_col, None)
    needed = [date_col, factor] + ([group] if group else [])
    require_columns(data, needed, context="signals")
    keys = [date_col] + ([group] if group else [])
    return data.groupby(keys, sort=False, dropna=False)[factor]


def _design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """説明変数行列。カテゴリ列はダミー化し、定数項を先頭に付ける。"""
    parts = []
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            parts.append(values.astype(float).to_frame(column))
        else:
            parts.append(pd.get_dummies(values, prefix=column, dtype=float))
    design = pd.concat(parts, axis=1)
    design.insert(0, "_const", 1.0)
    return design


def _ols_residual(y: np.ndarray, x: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """最小二乗（または加重最小二乗）の残差。"""
    return _ols(y, x, weights)[2]


def _ols(
    y: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """係数・決定係数・残差を返す。

    ``lstsq`` は最小ノルム解を返すので、カテゴリのダミーと定数項が共線でも残差と
    :math:`R^2` は一意に定まる（個々の係数の解釈はしない前提）。
    """
    if weights is None:
        weights = np.ones_like(y)
    root = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)

    residual = y - x @ beta
    total_weight = weights.sum()
    residual_ss = float(weights @ residual**2)
    mean = float(weights @ y / total_weight) if total_weight else np.nan
    total_ss = float(weights @ (y - mean) ** 2)
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else np.nan
    return beta, r_squared, residual


def _regress_by_date(
    data: pd.DataFrame,
    *,
    factor: str,
    factors: Sequence[str],
    weight: str | None,
    date_col: str | None,
    id_col: str | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """日付ごとに横断面回帰し、係数と決定係数を返す。"""
    date_col, _ = resolve_columns(date_col, id_col)
    factors = list(factors)
    require_columns(data, [date_col, factor, *factors], context="factor_exposure")
    if weight is not None:
        require_columns(data, [weight], context="factor_exposure")

    coefficients: dict = {}
    ratios: dict = {}
    columns: list[str] | None = None

    for date, chunk in data.groupby(pd.to_datetime(data[date_col]), sort=True):
        design = _design_matrix(chunk[factors])
        valid = chunk[factor].notna() & design.notna().all(axis=1)
        if weight is not None:
            valid &= chunk[weight].notna() & (chunk[weight] > 0)
        if valid.sum() <= design.shape[1]:
            continue  # 自由度が無い

        beta, r_squared, _ = _ols(
            chunk.loc[valid, factor].to_numpy(dtype=float),
            design.loc[valid].to_numpy(dtype=float),
            chunk.loc[valid, weight].to_numpy(dtype=float) if weight else None,
        )
        columns = list(design.columns)
        coefficients[date] = pd.Series(beta, index=columns)
        ratios[date] = r_squared

    frame = pd.DataFrame(coefficients).T
    if columns is not None:
        # 定数項は曝露として意味がないので落とす
        frame = frame.drop(columns=["_const"], errors="ignore")
    frame.index.name = date_col
    return frame.sort_index(), pd.Series(ratios, name="r_squared").sort_index()


def _validate_corr(method: str, factors: Sequence[str]) -> None:
    if method not in _CORR_METHODS:
        raise ValidationError(f"method は {list(_CORR_METHODS)} のいずれかです: {method!r}")
    if len(factors) < 2:
        raise ValidationError(f"factors は 2 つ以上必要です: {list(factors)}")
