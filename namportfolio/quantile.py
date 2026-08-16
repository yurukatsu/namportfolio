"""分位ポートフォリオ分析（シグナル評価）。

入力は **long 形式の DataFrame**（``date`` / ``bid`` カラム＋値カラム）で、
使うカラムを名前で指定する。

    q  = npf.quantile.quantile_returns(df, factor="value", forward_return="ret_20d")
    ic = npf.quantile.information_coefficient(df, factor="value", forward_return="ret_20d")

.. rubric:: 分位の向き

``Q1`` がファクター値の**最小**、``Qn`` が最大（``ascending=False`` で反転）。
「大きいほど良いシグナル」なら ``Qn - Q1`` がロング・ショートになる。

.. rubric:: 銘柄数が足りない日

その日の有効銘柄が ``min_assets`` に満たなければ分位を付けない（``NaN``）。
上場直後や祝日でユニバースが薄い日に、2〜3 銘柄で 5 分位を作らないため。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import performance as perf
from .core.config import resolve_columns
from .core.errors import ValidationError
from .core.frequency import resolve_periods_per_year, volatility_scale
from .core.panel import as_wide, require_columns
from .stats import newey_west_tstat, t_statistic

__all__ = [
    "assign_quantiles",
    "quantile_returns",
    "quantile_summary",
    "long_short_returns",
    "information_coefficient",
    "ic_summary",
    "factor_decay",
    "factor_autocorrelation",
    "quantile_turnover",
    "quantile_transition_matrix",
]

_CORR_METHODS = ("spearman", "pearson")


def assign_quantiles(
    data: pd.DataFrame,
    *,
    factor: str,
    n_quantiles: int = 5,
    group: str | None = None,
    ascending: bool = True,
    min_assets: int | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series:
    """各行に分位番号（1〜``n_quantiles``）を割り当てる。

    Parameters
    ----------
    data :
        long 形式の DataFrame。
    factor :
        分位分けに使う値のカラム名。
    group :
        指定すると**グループ内で**分位を作る（業種中立など）。
    ascending :
        ``True`` なら ``Q1`` が最小値。
    min_assets :
        その日（グループ）の有効銘柄数がこれ未満なら分位を付けない。
        ``None`` なら ``n_quantiles`` と同じ。

    Returns
    -------
    pd.Series
        ``data`` と同じ index を持つ float の Series（欠損は ``NaN``）。
        ``data["quantile"] = assign_quantiles(...)`` のように使える。
    """
    if n_quantiles < 2:
        raise ValidationError(f"n_quantiles は 2 以上です: {n_quantiles}")
    date_col, _ = resolve_columns(date_col, id_col)
    needed = [date_col, factor] + ([group] if group else [])
    require_columns(data, needed, context="assign_quantiles")

    if min_assets is None:
        min_assets = n_quantiles
    grouper = [date_col] + ([group] if group else [])
    grouped = data.groupby(grouper, sort=False, dropna=False)[factor]

    # 順位ベースで割る。qcut は同値が多いと境界が重複して失敗するが、
    # rank(method="first") なら同値も順に振り分けられて分位のサイズが揃う。
    ranks = grouped.rank(method="first", ascending=ascending, pct=True)
    counts = grouped.transform("count")
    quantiles = np.ceil(ranks * n_quantiles).clip(1, n_quantiles)
    return quantiles.where(counts >= min_assets).rename("quantile")


def quantile_returns(
    data: pd.DataFrame,
    *,
    factor: str,
    forward_return: str,
    n_quantiles: int = 5,
    group: str | None = None,
    weight: str | None = None,
    ascending: bool = True,
    min_assets: int | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """分位ごとの期間リターン。

    Parameters
    ----------
    forward_return :
        フォワードリターンのカラム名。
    weight :
        加重平均に使うカラム名（時価総額など）。``None`` なら等ウェイト。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns=["Q1", ..., "Qn"]``。ある日にその分位の銘柄が
        1 つも無ければ ``NaN``。
    """
    date_col, id_col = resolve_columns(date_col, id_col)
    needed = [date_col, factor, forward_return] + ([weight] if weight else [])
    require_columns(data, needed, context="quantile_returns")

    quantiles = assign_quantiles(
        data,
        factor=factor,
        n_quantiles=n_quantiles,
        group=group,
        ascending=ascending,
        min_assets=min_assets,
        date_col=date_col,
        id_col=id_col,
    )
    frame = pd.DataFrame(
        {
            "_date": pd.to_datetime(data[date_col]),
            "_quantile": quantiles,
            "_ret": data[forward_return],
        }
    )
    if weight is not None:
        frame["_weight"] = data[weight]
    frame = frame.dropna()

    if weight is None:
        grouped = frame.groupby(["_date", "_quantile"])["_ret"].mean()
    else:
        frame["_weighted"] = frame["_ret"] * frame["_weight"]
        sums = frame.groupby(["_date", "_quantile"])[["_weighted", "_weight"]].sum()
        grouped = sums["_weighted"] / sums["_weight"].replace(0, np.nan)

    wide = grouped.unstack("_quantile")
    wide.columns = [_label(q) for q in wide.columns]
    wide.index.name = date_col
    # その分位が 1 日も現れなかった場合に列が欠けるので、全分位を揃える
    return wide.reindex(columns=quantile_labels(n_quantiles)).sort_index()


def quantile_summary(
    returns: pd.DataFrame,
    *,
    periods_per_year: float | None = None,
) -> pd.DataFrame:
    """分位別リターンの要約統計。

    Parameters
    ----------
    returns :
        :func:`quantile_returns` の出力。

    Returns
    -------
    pd.DataFrame
        ``index`` は分位ラベル。``t_stat`` は素の t 値、``t_stat_nw`` は
        自己相関を補正した t 値（分位リターンは自己相関を持ちやすい）。
    """
    ppy = resolve_periods_per_year(returns, periods_per_year)
    rows = {}
    for label in returns.columns:
        series = returns[label].dropna()
        rows[label] = {
            "mean": series.mean(),
            "std": series.std(ddof=1),
            "annualized_return": perf.annualized_return(series, periods_per_year=ppy),
            "annualized_volatility": perf.annualized_volatility(series, periods_per_year=ppy),
            "sharpe_ratio": perf.sharpe_ratio(series, periods_per_year=ppy),
            "max_drawdown": perf.max_drawdown(series),
            "hit_rate": perf.hit_rate(series),
            "t_stat": t_statistic(series),
            "t_stat_nw": newey_west_tstat(series),
            "n_periods": len(series),
        }
    return pd.DataFrame(rows).T


def long_short_returns(
    returns: pd.DataFrame,
    *,
    long: str | None = None,
    short: str | None = None,
) -> pd.Series:
    """ロング・ショートのスプレッド。

    既定は「最上位分位 − 最下位分位」。``ascending=False`` で分位を作った場合や
    特定の分位を指定したい場合は ``long`` / ``short`` にラベルを渡す。
    """
    labels = list(returns.columns)
    if not labels:
        raise ValidationError("分位リターンが空です。")
    long = long or labels[-1]
    short = short or labels[0]
    require_columns(returns, [long, short], context="long_short_returns")
    return (returns[long] - returns[short]).rename(f"{long}-{short}")


def information_coefficient(
    data: pd.DataFrame,
    *,
    factor: str,
    forward_return: str,
    method: str = "spearman",
    group: str | None = None,
    min_assets: int = 5,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series | pd.DataFrame:
    """期間ごとの情報係数（IC）。

    Parameters
    ----------
    method :
        ``"spearman"``（順位相関、既定）または ``"pearson"``。順位相関は外れ値に
        頑健なので、シグナルの評価では通常こちらを使う。
    group :
        指定するとグループごとの IC を列に持つ DataFrame を返す。
    min_assets :
        その期間の有効銘柄数がこれ未満なら ``NaN``（少数銘柄の相関は不安定）。

    Returns
    -------
    pd.Series | pd.DataFrame
        ``index=date``。``group`` を指定した場合は ``columns=group``。
    """
    if method not in _CORR_METHODS:
        raise ValidationError(f"method は {list(_CORR_METHODS)} のいずれかです: {method!r}")
    date_col, id_col = resolve_columns(date_col, id_col)

    if group is not None:
        return _ic_by_group(
            data,
            factor=factor,
            forward_return=forward_return,
            method=method,
            group=group,
            min_assets=min_assets,
            date_col=date_col,
        )

    factor_wide = as_wide(data, factor, date_col=date_col, id_col=id_col)
    return_wide = as_wide(data, forward_return, date_col=date_col, id_col=id_col)
    factor_wide, return_wide = factor_wide.align(return_wide, join="inner")

    counts = (factor_wide.notna() & return_wide.notna()).sum(axis=1)
    if method == "spearman":
        # 行ごとに順位へ変換してから Pearson を取れば Spearman と一致する
        factor_wide = factor_wide.rank(axis=1)
        return_wide = return_wide.rank(axis=1)

    ic = factor_wide.corrwith(return_wide, axis=1)
    return ic.where(counts >= min_assets).rename("ic")


def ic_summary(
    ic: pd.Series | pd.DataFrame,
    *,
    periods_per_year: float | None = None,
) -> pd.Series | pd.DataFrame:
    """IC 系列の要約。

    Returns
    -------
    pd.Series | pd.DataFrame
        ``icir`` は期間ベースの IC / IC標準偏差、``icir_annualized`` はそれを
        年率化した値。``t_stat_nw`` は自己相関を補正した t 値。
    """
    if isinstance(ic, pd.DataFrame):
        return pd.DataFrame(
            {col: ic_summary(ic[col], periods_per_year=periods_per_year) for col in ic.columns}
        )

    values = ic.dropna()
    if len(values) < 2:
        return pd.Series(dtype=float)

    mean = float(values.mean())
    std = float(values.std(ddof=1))
    icir = mean / std if std else np.nan
    ppy = resolve_periods_per_year(values, periods_per_year)
    return pd.Series(
        {
            "mean": mean,
            "std": std,
            "icir": icir,
            "icir_annualized": icir * volatility_scale(ppy) if std else np.nan,
            "t_stat": t_statistic(values),
            "t_stat_nw": newey_west_tstat(values),
            "hit_rate": float((values > 0).mean()),
            "n_periods": len(values),
        },
        name=ic.name,
    )


def factor_decay(
    data: pd.DataFrame,
    *,
    factor: str,
    forward_returns: Sequence[str],
    method: str = "spearman",
    min_assets: int = 5,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """保有期間を変えたときの IC の減衰。

    Parameters
    ----------
    forward_returns :
        ホライズンの異なるフォワードリターンのカラム名（短い順に渡す）。

    Returns
    -------
    pd.DataFrame
        ``index`` はカラム名（ホライズン）、``columns`` は :func:`ic_summary` の項目。
    """
    if not len(forward_returns):
        raise ValidationError("forward_returns が空です。")
    rows = {
        column: ic_summary(
            information_coefficient(
                data,
                factor=factor,
                forward_return=column,
                method=method,
                min_assets=min_assets,
                date_col=date_col,
                id_col=id_col,
            )
        )
        for column in forward_returns
    }
    return pd.DataFrame(rows).T


def factor_autocorrelation(
    data: pd.DataFrame,
    *,
    factor: str,
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    method: str = "spearman",
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series:
    """シグナル自体の持続性。

    各時点で「今のシグナル」と「``lag`` 期前のシグナル」の横断面相関を取り、
    期間平均を返す。1 に近いほどシグナルが動かない＝売買が少なくて済む。
    """
    if method not in _CORR_METHODS:
        raise ValidationError(f"method は {list(_CORR_METHODS)} のいずれかです: {method!r}")
    wide = as_wide(data, factor, date_col=date_col, id_col=id_col)
    if method == "spearman":
        wide = wide.rank(axis=1)
    return pd.Series(
        {lag: float(wide.corrwith(wide.shift(lag), axis=1).mean()) for lag in lags},
        name="autocorrelation",
    ).rename_axis("lag")


def quantile_turnover(
    data: pd.DataFrame,
    *,
    factor: str,
    n_quantiles: int = 5,
    group: str | None = None,
    ascending: bool = True,
    min_assets: int | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """分位ごとの入れ替わり率。

    「前期その分位にいた銘柄のうち、今期は抜けた割合」を名前ベースで数える
    （ウェイトの増減は見ない）。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns=["Q1", ..., "Qn"]``。最初の期間は ``NaN``。
    """
    membership = _quantile_panel(
        data,
        factor=factor,
        n_quantiles=n_quantiles,
        group=group,
        ascending=ascending,
        min_assets=min_assets,
        date_col=date_col,
        id_col=id_col,
    )

    result = {}
    for q in range(1, n_quantiles + 1):
        current = membership == q
        previous = current.shift(1)
        n_previous = previous.sum(axis=1)
        left = (previous & ~current).sum(axis=1)
        turnover = left / n_previous.replace(0, np.nan)
        turnover.iloc[0] = np.nan  # 前期が無いので定義できない
        result[_label(q)] = turnover
    return pd.DataFrame(result)


def quantile_transition_matrix(
    data: pd.DataFrame,
    *,
    factor: str,
    n_quantiles: int = 5,
    group: str | None = None,
    ascending: bool = True,
    min_assets: int | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """分位間の遷移確率。

    Returns
    -------
    pd.DataFrame
        ``index`` が前期の分位、``columns`` が今期の分位。各行の合計が 1。
        対角が大きいほどシグナルが安定している。
    """
    membership = _quantile_panel(
        data,
        factor=factor,
        n_quantiles=n_quantiles,
        group=group,
        ascending=ascending,
        min_assets=min_assets,
        date_col=date_col,
        id_col=id_col,
    )
    pairs = pd.DataFrame(
        {
            "from": membership.shift(1).to_numpy().ravel(),
            "to": membership.to_numpy().ravel(),
        }
    ).dropna()
    if pairs.empty:
        raise ValidationError("遷移を数えられる期間がありません（2 期間以上必要です）。")

    matrix = pd.crosstab(pairs["from"], pairs["to"], normalize="index")
    labels = quantile_labels(n_quantiles)
    matrix.index = [_label(q) for q in matrix.index]
    matrix.columns = [_label(q) for q in matrix.columns]
    return matrix.reindex(index=labels, columns=labels)


def quantile_labels(n_quantiles: int) -> list[str]:
    """``["Q1", ..., "Qn"]``。"""
    return [f"Q{i}" for i in range(1, n_quantiles + 1)]


# --------------------------------------------------------------------------
# 内部ヘルパー
# --------------------------------------------------------------------------


def _label(quantile: float) -> str:
    return f"Q{int(quantile)}"


def _quantile_panel(data: pd.DataFrame, *, factor: str, date_col, id_col, **kwargs) -> pd.DataFrame:
    """(date × bid) の分位番号パネルを作る。"""
    date_col, id_col = resolve_columns(date_col, id_col)
    quantiles = assign_quantiles(data, factor=factor, date_col=date_col, id_col=id_col, **kwargs)
    frame = pd.DataFrame(
        {
            date_col: pd.to_datetime(data[date_col]),
            id_col: data[id_col],
            "_quantile": quantiles,
        }
    )
    return as_wide(frame, "_quantile", date_col=date_col, id_col=id_col)


def _ic_by_group(
    data: pd.DataFrame,
    *,
    factor: str,
    forward_return: str,
    method: str,
    group: str,
    min_assets: int,
    date_col: str,
) -> pd.DataFrame:
    """グループごとの IC。グループ数は業種程度を想定しているので素直に回す。"""
    require_columns(
        data, [date_col, group, factor, forward_return], context="information_coefficient"
    )
    frame = data[[date_col, group, factor, forward_return]].dropna().copy()
    frame[date_col] = pd.to_datetime(frame[date_col])

    def _corr(chunk: pd.DataFrame) -> float:
        if len(chunk) < min_assets:
            return np.nan
        return chunk[factor].corr(chunk[forward_return], method=method)

    ic = frame.groupby([date_col, group]).apply(_corr, include_groups=False)
    return ic.unstack(group).sort_index()
