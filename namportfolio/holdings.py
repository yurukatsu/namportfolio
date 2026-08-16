"""保有ベース分析。

入力は **long 形式の DataFrame**（``date`` / ``bid`` カラム＋ウェイト列）。
ベンチマークウェイトや銘柄属性も同じ DataFrame の列として持たせる。

    df   # date, bid, weight, bench_weight, ret_1m, sector, per

    npf.holdings.concentration(df, weight="weight")
    npf.holdings.contribution(df, forward_return="ret_1m", by="sector")
    npf.holdings.turnover(df, weight="weight")

.. rubric:: ウェイトの欠損は 0

行が無い／値が欠損している銘柄は「保有していない」とみなす。保有銘柄だけを
行に持つ long データをそのまま渡せる。

.. rubric:: ウェイトのドリフト

ターンオーバーや売買の集計は、**渡されたウェイトのスナップショット同士の差**を
見る。期中の株価変動によるウェイト変化（ドリフト）と実際の売買を区別できないので、
リバランス直後のウェイトを渡すか、値を目安として読むこと。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .core.config import resolve_columns
from .core.errors import ValidationError
from .core.panel import as_wide, require_columns

__all__ = [
    "concentration",
    "allocation",
    "top_holdings",
    "characteristics",
    "contribution",
    "top_contributors",
    "turnover",
    "trades",
    "average_holding_period",
]

_TURNOVER_METHODS = ("one_way", "two_way")


# --------------------------------------------------------------------------
# 保有構造
# --------------------------------------------------------------------------


def concentration(
    data: pd.DataFrame,
    *,
    weight: str = "weight",
    top: Sequence[int] = (10, 20),
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """銘柄数と集中度の推移。

    Parameters
    ----------
    top :
        上位何銘柄のウェイト合計を出すか。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns=[n_holdings, hhi, effective_n, top10_share, ...]``。

        - ``hhi`` はウェイトの二乗和。分散が効いているほど小さい
        - ``effective_n`` は ``1 / hhi``。「実質何銘柄に賭けているか」
        - ``top*_share`` はウェイト上位の合計（ロング側の集中を見る）
    """
    weights = _weight_panel(data, weight, date_col, id_col)
    squared = (weights**2).sum(axis=1)

    frame = pd.DataFrame(
        {
            "n_holdings": (weights != 0).sum(axis=1),
            "hhi": squared,
            "effective_n": 1.0 / squared.replace(0, np.nan),
        }
    )
    descending = np.sort(weights.to_numpy(), axis=1)[:, ::-1]
    for k in top:
        frame[f"top{k}_share"] = descending[:, :k].sum(axis=1)
    return frame


def allocation(
    data: pd.DataFrame,
    *,
    by: str,
    weight: str = "weight",
    benchmark_weight: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """セグメント別の配分。

    Parameters
    ----------
    by :
        セグメントのカラム名（業種、サイズ区分、国など）。
    benchmark_weight :
        指定すると**アクティブ配分**（ポート − ベンチ）を返す。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns`` はセグメント値。
    """
    date_col, _ = resolve_columns(date_col, id_col)
    needed = [date_col, by, weight] + ([benchmark_weight] if benchmark_weight else [])
    require_columns(data, needed, context="allocation")

    dates = pd.to_datetime(data[date_col])
    portfolio = data.groupby([dates, data[by]], observed=True)[weight].sum().unstack(1)
    if benchmark_weight is None:
        result = portfolio
    else:
        benchmark = (
            data.groupby([dates, data[by]], observed=True)[benchmark_weight].sum().unstack(1)
        )
        result = portfolio.sub(benchmark, fill_value=0.0)
    result.index.name = date_col
    return result.sort_index()


def top_holdings(
    data: pd.DataFrame,
    *,
    weight: str = "weight",
    n: int = 10,
    at: object | None = None,
    benchmark_weight: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """ある時点の上位保有銘柄。

    Parameters
    ----------
    at :
        対象日。``None`` なら最終日。
    benchmark_weight :
        指定すると ``active`` 列（ポート − ベンチ）を付け、**アクティブウェイト順**に
        並べる。指定しなければポートのウェイト順。

    Returns
    -------
    pd.DataFrame
        ``index=bid``。上位・下位を両端から ``n`` 件ずつ返す（アクティブ指定時は
        オーバーウェイトとアンダーウェイトの両端）。
    """
    date_col, id_col = resolve_columns(date_col, id_col)
    needed = [date_col, id_col, weight] + ([benchmark_weight] if benchmark_weight else [])
    require_columns(data, needed, context="top_holdings")

    dates = pd.to_datetime(data[date_col])
    target = dates.max() if at is None else pd.Timestamp(at)
    snapshot = data.loc[dates == target].set_index(id_col)
    if snapshot.empty:
        raise ValidationError(f"{target:%Y-%m-%d} のデータがありません。")

    columns = [weight]
    if benchmark_weight is not None:
        snapshot = snapshot.assign(
            active=snapshot[weight].fillna(0.0) - snapshot[benchmark_weight].fillna(0.0)
        )
        columns = [weight, benchmark_weight, "active"]
        key = "active"
    else:
        key = weight

    ordered = snapshot[columns].sort_values(key, ascending=False)
    if len(ordered) <= 2 * n:
        return ordered
    return pd.concat([ordered.head(n), ordered.tail(n)])


def characteristics(
    data: pd.DataFrame,
    *,
    columns: Sequence[str],
    weight: str = "weight",
    benchmark_weight: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """ポートフォリオ特性の加重平均（PER、時価総額、ROE など）。

    属性が欠損している銘柄は除外し、**残りのウェイトで再正規化**する。
    欠損銘柄を 0 として扱うと、特性値が保有比率に応じて薄まってしまう。

    Parameters
    ----------
    columns :
        加重平均する属性のカラム名。
    benchmark_weight :
        指定すると ``差分`` を返す（ポート − ベンチ）。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns`` は属性名。
    """
    date_col, _ = resolve_columns(date_col, id_col)
    needed = [date_col, weight, *columns] + ([benchmark_weight] if benchmark_weight else [])
    require_columns(data, needed, context="characteristics")

    dates = pd.to_datetime(data[date_col])
    result = {}
    for column in columns:
        portfolio = _weighted_mean(data, column, weight, dates)
        if benchmark_weight is None:
            result[column] = portfolio
        else:
            result[column] = portfolio - _weighted_mean(data, column, benchmark_weight, dates)

    frame = pd.DataFrame(result)
    frame.index.name = date_col
    return frame.sort_index()


# --------------------------------------------------------------------------
# 寄与度
# --------------------------------------------------------------------------


def contribution(
    data: pd.DataFrame,
    *,
    forward_return: str,
    weight: str = "weight",
    by: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """期間ごとのリターン寄与（``ウェイト × リターン``）。

    「先月なぜ勝った／負けたか」に最短で答える。行方向に合計すればその期間の
    ポートフォリオリターンになる。

    Parameters
    ----------
    by :
        指定するとセグメント別に合計する。``None`` なら銘柄別。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns`` は銘柄またはセグメント。

    Notes
    -----
    **期間をまたぐ合計は近似**。複数期間の寄与を足し上げても、複利の効果があるため
    累積リターンとは厳密には一致しない。厳密な期間リンクが要るときは Brinson 帰属の
    リンキング（F5）を使う。
    """
    date_col, id_col = resolve_columns(date_col, id_col)
    require_columns(data, [date_col, id_col, weight, forward_return], context="contribution")

    frame = pd.DataFrame(
        {
            date_col: pd.to_datetime(data[date_col]),
            id_col: data[id_col],
            "_contribution": data[weight] * data[forward_return],
        }
    )
    if by is None:
        return as_wide(frame, "_contribution", date_col=date_col, id_col=id_col)

    require_columns(data, [by], context="contribution")
    grouped = (
        frame.assign(_segment=data[by].to_numpy())
        .groupby([date_col, "_segment"], observed=True)["_contribution"]
        .sum()
        .unstack("_segment")
    )
    grouped.index.name = date_col
    return grouped.sort_index()


def top_contributors(
    data: pd.DataFrame,
    *,
    forward_return: str,
    weight: str = "weight",
    n: int = 10,
    by: str | None = None,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """全期間の寄与を合計し、上位・下位を返す。

    Returns
    -------
    pd.DataFrame
        ``index`` は銘柄（または ``by`` のセグメント）、``columns=[contribution,
        n_periods, side]``。``side`` は ``"top"`` / ``"bottom"``。
    """
    per_period = contribution(
        data,
        forward_return=forward_return,
        weight=weight,
        by=by,
        date_col=date_col,
        id_col=id_col,
    )
    total = per_period.sum(axis=0, min_count=1).dropna().sort_values(ascending=False)
    counts = per_period.notna().sum(axis=0)

    if len(total) <= 2 * n:
        selected = total
        side = np.where(selected >= 0, "top", "bottom")
    else:
        selected = pd.concat([total.head(n), total.tail(n)])
        side = ["top"] * n + ["bottom"] * n

    return pd.DataFrame(
        {
            "contribution": selected,
            "n_periods": counts.reindex(selected.index),
            "side": side,
        }
    )


# --------------------------------------------------------------------------
# 売買
# --------------------------------------------------------------------------


def turnover(
    data: pd.DataFrame,
    *,
    weight: str = "weight",
    method: str = "one_way",
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series:
    """ターンオーバー。

    Parameters
    ----------
    method :
        ``"one_way"``（既定）は ``Σ|Δw| / 2``＝「入れ替わった割合」。
        ``"two_way"`` は ``Σ|Δw|``＝売り買い合計の売買金額比率。

    Returns
    -------
    pd.Series
        ``index=date``。最初の期間は前期が無いので ``NaN``。
    """
    if method not in _TURNOVER_METHODS:
        raise ValidationError(f"method は {list(_TURNOVER_METHODS)} のいずれかです: {method!r}")
    weights = _weight_panel(data, weight, date_col, id_col)
    changed = (weights - weights.shift(1)).abs().sum(axis=1)
    if method == "one_way":
        changed = changed / 2.0
    changed.iloc[0] = np.nan
    return changed.rename(f"turnover_{method}")


def trades(
    data: pd.DataFrame,
    *,
    weight: str = "weight",
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """期間ごとの新規・売却の件数と売買ウェイト。

    Returns
    -------
    pd.DataFrame
        ``index=date``、``columns=[n_new, n_closed, weight_bought, weight_sold]``。
        ``weight_bought`` / ``weight_sold`` は既存銘柄の買い増し・売り減らしも含む。
    """
    weights = _weight_panel(data, weight, date_col, id_col)
    held = weights != 0
    previously = held.shift(1, fill_value=False)
    change = weights - weights.shift(1)

    frame = pd.DataFrame(
        {
            "n_new": (held & ~previously).sum(axis=1),
            "n_closed": (~held & previously).sum(axis=1),
            "weight_bought": change.clip(lower=0).sum(axis=1),
            "weight_sold": -change.clip(upper=0).sum(axis=1),
        }
    )
    frame.iloc[0] = np.nan
    return frame


def average_holding_period(
    data: pd.DataFrame,
    *,
    weight: str = "weight",
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series:
    """銘柄ごとの平均保有期間（データ点数）。

    「保有していた期間数 ÷ 保有を開始した回数」。一度買って持ち続けた銘柄は
    全期間数に、出たり入ったりした銘柄は短い値になる。

    Returns
    -------
    pd.Series
        ``index=bid``。一度も保有していない銘柄は含まない。
    """
    weights = _weight_panel(data, weight, date_col, id_col)
    held = weights != 0
    spells = (held & ~held.shift(1, fill_value=False)).sum(axis=0)
    periods = held.sum(axis=0)
    return (periods / spells.replace(0, np.nan)).dropna().rename("avg_holding_period")


# --------------------------------------------------------------------------
# 内部ヘルパー
# --------------------------------------------------------------------------


def _weight_panel(
    data: pd.DataFrame,
    weight: str,
    date_col: str | None,
    id_col: str | None,
) -> pd.DataFrame:
    """(date × bid) のウェイト行列。欠損は「保有していない」= 0。"""
    return as_wide(data, weight, date_col=date_col, id_col=id_col).fillna(0.0)


def _weighted_mean(
    data: pd.DataFrame,
    value: str,
    weight: str,
    dates: pd.Series,
) -> pd.Series:
    """属性の加重平均。属性が欠損している銘柄はウェイトごと除外する。"""
    valid = data[value].notna() & data[weight].notna()
    subset = data.loc[valid]
    keys = dates.loc[valid]
    numerator = (subset[value] * subset[weight]).groupby(keys).sum()
    denominator = subset[weight].groupby(keys).sum()
    return numerator / denominator.replace(0, np.nan)
