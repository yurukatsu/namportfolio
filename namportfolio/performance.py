"""ポートフォリオ・リターン評価。

すべて ``DatetimeIndex`` を持つリターン系列（単純リターン）を受け取る。
Series を想定しているが、多くの関数は DataFrame でも列ごとに動く。

.. rubric:: 型の規約

入力は :data:`ReturnsLike`（``pd.Series | pd.DataFrame``）。スカラー指標を返す関数の
戻り値は :data:`Metric`（``float | pd.Series``）で、**Series を渡せば ``float``、
DataFrame を渡せば列名を index に持つ ``pd.Series``** が返る。系列を返す関数
（``cumulative_returns`` / ``drawdown`` / ``rolling_*`` など）は入力と同じ型を返す。

``drawdown_table`` と ``monthly_table`` は Series 専用で ``pd.DataFrame`` を返す。

.. rubric:: 欠損の扱い

``mean`` / ``std`` / ``prod`` は pandas の既定どおり欠損を無視する。累積計算
（``cumprod``）だけは欠損があると以降すべてが欠損になるため、**リターン 0 として
扱って累積を継続する**（データがない期間は動かない、という定義）。

.. rubric:: 引数の規約

``risk_free`` は**年率**のスカラーで渡す。内部で期間率に変換する。
``periods_per_year`` は ``None`` なら日付から推定する。
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd

from .core.errors import ValidationError
from .core.frequency import resolve_periods_per_year, volatility_scale

__all__ = [
    # 型エイリアス
    "ReturnsLike",
    "Metric",
    # 基本
    "cumulative_returns",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    # レシオ
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    # ドローダウン
    "drawdown",
    "max_drawdown",
    "drawdown_table",
    # 分布
    "value_at_risk",
    "conditional_value_at_risk",
    "hit_rate",
    "win_loss_ratio",
    # ベンチマーク相対
    "active_returns",
    "tracking_error",
    "information_ratio",
    "beta",
    "alpha",
    "capture_ratio",
    # 期間集計
    "aggregate_returns",
    "monthly_table",
    # ローリング
    "rolling_volatility",
    "rolling_sharpe",
    "rolling_beta",
    "rolling_tracking_error",
    "rolling_information_ratio",
    # まとめ
    "performance_summary",
    # 頻度エイリアス
    "MONTH_END",
    "QUARTER_END",
    "YEAR_END",
]

#: リターン系列。``index`` は ``DatetimeIndex``、値は単純リターン。
#: Series なら単一戦略、DataFrame なら列ごとに複数戦略を表す。
ReturnsLike: TypeAlias = pd.Series | pd.DataFrame

#: スカラー指標の戻り値。入力が Series なら ``float``、
#: DataFrame なら列名を index に持つ ``pd.Series``。
Metric: TypeAlias = float | pd.Series

# pandas 2.2 で頻度文字列が "M" → "ME" に変わり、3.0 で旧表記は削除された。
# 社内の古い pandas でも動くよう分岐する。
_PANDAS_VERSION = tuple(int(part) for part in pd.__version__.split(".")[:2])
_MODERN_FREQ = _PANDAS_VERSION >= (2, 2)

MONTH_END = "ME" if _MODERN_FREQ else "M"
QUARTER_END = "QE" if _MODERN_FREQ else "Q"
YEAR_END = "YE" if _MODERN_FREQ else "A"


# --------------------------------------------------------------------------
# 基本
# --------------------------------------------------------------------------


def cumulative_returns(returns: ReturnsLike, *, log: bool = False) -> ReturnsLike:
    """累積リターン系列を返す。

    Parameters
    ----------
    returns :
        期間リターン。欠損はリターン 0 として扱う。
    log :
        ``True`` なら対数累積（``log(1+r)`` の累積和）を返す。プロットで
        縦軸を対数にしたい場合に使う。

    Returns
    -------
    ReturnsLike
        入力と同じ型・同じ index。期首を 0 とした累積リターン（``0.21`` なら +21%）。
    """
    filled = returns.fillna(0.0)
    if log:
        return np.log1p(filled).cumsum()
    return (1.0 + filled).cumprod() - 1.0


def total_return(returns: ReturnsLike) -> Metric:
    """全期間の累積リターン。"""
    return (1.0 + returns).prod() - 1.0


def annualized_return(
    returns: ReturnsLike,
    *,
    periods_per_year: float | None = None,
) -> Metric:
    """年率リターン（幾何平均、CAGR）。

    期間数は欠損を除いた有効データ数で数える。
    """
    ppy = resolve_periods_per_year(returns, periods_per_year)
    n = returns.count()
    growth = (1.0 + returns).prod()
    if isinstance(n, pd.Series):  # DataFrame 入力
        return growth ** (ppy / n.replace(0, np.nan)) - 1.0
    if n == 0:
        return np.nan
    return growth ** (ppy / n) - 1.0


def annualized_volatility(
    returns: ReturnsLike,
    *,
    periods_per_year: float | None = None,
    ddof: int = 1,
) -> Metric:
    """年率ボラティリティ。"""
    ppy = resolve_periods_per_year(returns, periods_per_year)
    return returns.std(ddof=ddof) * volatility_scale(ppy)


# --------------------------------------------------------------------------
# レシオ
# --------------------------------------------------------------------------


def sharpe_ratio(
    returns: ReturnsLike,
    *,
    risk_free: float = 0.0,
    periods_per_year: float | None = None,
    ddof: int = 1,
) -> Metric:
    """シャープレシオ（算術ベース）。

    ``mean(r - rf) / std(r - rf) * sqrt(P)``。
    """
    ppy = resolve_periods_per_year(returns, periods_per_year)
    excess = returns - _periodic_rate(risk_free, ppy)
    vol = excess.std(ddof=ddof)
    return excess.mean() / _nan_if_zero(vol) * volatility_scale(ppy)


def sortino_ratio(
    returns: ReturnsLike,
    *,
    risk_free: float = 0.0,
    target: float = 0.0,
    periods_per_year: float | None = None,
) -> Metric:
    """ソルティノレシオ。

    下方偏差は ``target`` を下回った分の二乗平均平方根。平均は**全有効期間**で
    取る（下振れ期間だけで割る流派もあるが、期間数によって値が跳ねるため採らない）。
    """
    ppy = resolve_periods_per_year(returns, periods_per_year)
    excess = returns - _periodic_rate(risk_free, ppy)
    downside = (excess - target).clip(upper=0.0)
    downside_dev = np.sqrt((downside**2).mean())
    return excess.mean() / _nan_if_zero(downside_dev) * volatility_scale(ppy)


def calmar_ratio(returns: ReturnsLike, *, periods_per_year: float | None = None) -> Metric:
    """カルマーレシオ（年率リターン / 最大ドローダウンの絶対値）。"""
    mdd = max_drawdown(returns)
    ann = annualized_return(returns, periods_per_year=periods_per_year)
    return ann / _nan_if_zero(abs(mdd))


# --------------------------------------------------------------------------
# ドローダウン
# --------------------------------------------------------------------------


def drawdown(returns: ReturnsLike) -> ReturnsLike:
    """アンダーウォーター系列（各時点の高値からの下落率、0 以下）。"""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(returns: ReturnsLike) -> Metric:
    """最大ドローダウン（負の値）。"""
    return drawdown(returns).min()


def drawdown_table(returns: pd.Series, *, top: int = 5) -> pd.DataFrame:
    """深い順にドローダウン期間を一覧する。

    Parameters
    ----------
    returns :
        Series のみ（DataFrame 非対応）。
    top :
        返す件数。

    Returns
    -------
    pd.DataFrame
        ``peak`` 直近高値の日付 / ``valley`` 底の日付 / ``recovery`` 高値を回復した日付
        （未回復なら ``NaT``）/ ``max_drawdown`` / ``length`` peak〜recovery の期間数
        （未回復なら peak〜最終日）/ ``recovery_length`` valley〜recovery の期間数。
        期間数はカレンダー日数ではなくデータ点数。
    """
    if isinstance(returns, pd.DataFrame):
        raise ValidationError("drawdown_table は Series のみ対応です。列ごとに呼んでください。")

    columns = ["peak", "valley", "recovery", "max_drawdown", "length", "recovery_length"]
    underwater = drawdown(returns)
    is_under = underwater < 0
    if not is_under.any():
        return pd.DataFrame(columns=columns)

    index = underwater.index
    # 連続する水面下の区間に ID を振る
    segment_id = (is_under != is_under.shift(fill_value=False)).cumsum()

    records = []
    for _, segment in underwater[is_under].groupby(segment_id[is_under]):
        first_pos = index.get_loc(segment.index[0])
        last_pos = index.get_loc(segment.index[-1])
        recovered = last_pos + 1 < len(index)

        peak_date = index[first_pos - 1] if first_pos > 0 else segment.index[0]
        recovery_date = index[last_pos + 1] if recovered else pd.NaT
        valley_date = segment.idxmin()
        valley_pos = index.get_loc(valley_date)
        end_pos = last_pos + 1 if recovered else last_pos

        records.append(
            {
                "peak": peak_date,
                "valley": valley_date,
                "recovery": recovery_date,
                "max_drawdown": segment.min(),
                "length": end_pos - max(first_pos - 1, 0),
                "recovery_length": (end_pos - valley_pos) if recovered else np.nan,
            }
        )

    table = pd.DataFrame.from_records(records, columns=columns)
    return table.sort_values("max_drawdown").head(top).reset_index(drop=True)


# --------------------------------------------------------------------------
# 分布
# --------------------------------------------------------------------------


def value_at_risk(returns: ReturnsLike, *, level: float = 0.05) -> Metric:
    """ヒストリカル VaR。``level`` 分位点をそのまま返す（損失は負の値）。"""
    return returns.quantile(level)


def conditional_value_at_risk(returns: ReturnsLike, *, level: float = 0.05) -> Metric:
    """条件付き VaR（期待ショートフォール）。VaR 以下の平均。"""
    var = returns.quantile(level)
    if isinstance(returns, pd.DataFrame):
        return pd.Series(
            {col: returns[col][returns[col] <= var[col]].mean() for col in returns.columns}
        )
    return returns[returns <= var].mean()


def hit_rate(returns: ReturnsLike) -> Metric:
    """プラスだった期間の割合。"""
    return (returns > 0).sum() / returns.count()


def win_loss_ratio(returns: ReturnsLike) -> Metric:
    """平均勝ち幅 / 平均負け幅の絶対値。"""
    wins = returns[returns > 0].mean()
    losses = returns[returns < 0].mean()
    return wins / _nan_if_zero(abs(losses))


# --------------------------------------------------------------------------
# ベンチマーク相対
# --------------------------------------------------------------------------


def active_returns(returns: ReturnsLike, benchmark: ReturnsLike) -> ReturnsLike:
    """アクティブリターン（単純差分）。日付は内側結合で揃える。

    ``benchmark`` は Series、または 1 列の DataFrame。
    """
    aligned, bench = _align(returns, benchmark)
    if isinstance(aligned, pd.DataFrame):
        # DataFrame - Series は既定で列方向に合わせるため、明示的に日付方向で引く
        return aligned.sub(bench, axis=0)
    return aligned - bench


def tracking_error(
    returns: ReturnsLike,
    benchmark: ReturnsLike,
    *,
    periods_per_year: float | None = None,
    ddof: int = 1,
) -> Metric:
    """トラッキングエラー（年率）。"""
    active = active_returns(returns, benchmark)
    ppy = resolve_periods_per_year(active, periods_per_year)
    return active.std(ddof=ddof) * volatility_scale(ppy)


def information_ratio(
    returns: ReturnsLike,
    benchmark: ReturnsLike,
    *,
    periods_per_year: float | None = None,
    ddof: int = 1,
) -> Metric:
    """情報比（アクティブリターン年率 / トラッキングエラー）。"""
    active = active_returns(returns, benchmark)
    ppy = resolve_periods_per_year(active, periods_per_year)
    te = active.std(ddof=ddof) * volatility_scale(ppy)
    return active.mean() * ppy / _nan_if_zero(te)


def beta(returns: ReturnsLike, benchmark: ReturnsLike) -> Metric:
    """ベンチマークに対するベータ（``cov / var``）。"""
    aligned, bench = _align(returns, benchmark)
    variance = bench.var()
    if isinstance(aligned, pd.DataFrame):
        return aligned.apply(lambda col: col.cov(bench)) / _nan_if_zero(variance)
    return aligned.cov(bench) / _nan_if_zero(variance)


def alpha(
    returns: ReturnsLike,
    benchmark: ReturnsLike,
    *,
    risk_free: float = 0.0,
    periods_per_year: float | None = None,
) -> Metric:
    """ジェンセンのアルファ（年率）。

    ``ann(r) - [rf + beta * (ann(b) - rf)]``
    """
    aligned, bench = _align(returns, benchmark)
    ppy = resolve_periods_per_year(aligned, periods_per_year)
    b = beta(aligned, bench)
    port = annualized_return(aligned, periods_per_year=ppy)
    market = annualized_return(bench, periods_per_year=ppy)
    return port - (risk_free + b * (market - risk_free))


def capture_ratio(
    returns: ReturnsLike,
    benchmark: ReturnsLike,
    *,
    side: str = "up",
) -> Metric:
    """アップ／ダウンキャプチャレシオ。

    ベンチマークが正（``side="up"``）または負（``"down"``）だった期間だけを取り出し、
    ポートフォリオの累積リターンをベンチマークの累積リターンで割る。
    """
    if side not in ("up", "down"):
        raise ValidationError(f"side は 'up' か 'down' です: {side!r}")
    aligned, bench = _align(returns, benchmark)
    mask = bench > 0 if side == "up" else bench < 0
    if not mask.any():
        return np.nan
    port_growth = (1.0 + aligned[mask]).prod() - 1.0
    bench_growth = (1.0 + bench[mask]).prod() - 1.0
    return port_growth / _nan_if_zero(bench_growth)


# --------------------------------------------------------------------------
# 期間集計
# --------------------------------------------------------------------------


def aggregate_returns(returns: ReturnsLike, freq: str = MONTH_END) -> ReturnsLike:
    """期間リターンを幾何的に集約する（月次・四半期・年次）。

    ``freq`` には :data:`MONTH_END` / :data:`QUARTER_END` / :data:`YEAR_END` を渡す
    （pandas のバージョン差を吸収した頻度文字列）。
    """
    return (1.0 + returns.fillna(0.0)).resample(freq).prod() - 1.0


def monthly_table(returns: pd.Series) -> pd.DataFrame:
    """年 × 月のリターン表。末尾に年間リターンの列を付ける。"""
    if isinstance(returns, pd.DataFrame):
        raise ValidationError("monthly_table は Series のみ対応です。")

    monthly = aggregate_returns(returns, MONTH_END)
    table = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.to_numpy(),
        }
    ).pivot(index="year", columns="month", values="ret")
    table.columns = [f"{m:02d}" for m in table.columns]
    table["year_total"] = aggregate_returns(returns, YEAR_END).to_numpy()
    return table


# --------------------------------------------------------------------------
# ローリング
# --------------------------------------------------------------------------


def rolling_volatility(
    returns: ReturnsLike,
    window: int,
    *,
    periods_per_year: float | None = None,
) -> ReturnsLike:
    """ローリング年率ボラティリティ。"""
    ppy = resolve_periods_per_year(returns, periods_per_year)
    return returns.rolling(window).std() * volatility_scale(ppy)


def rolling_sharpe(
    returns: ReturnsLike,
    window: int,
    *,
    risk_free: float = 0.0,
    periods_per_year: float | None = None,
) -> ReturnsLike:
    """ローリングシャープレシオ。"""
    ppy = resolve_periods_per_year(returns, periods_per_year)
    excess = returns - _periodic_rate(risk_free, ppy)
    rolled = excess.rolling(window)
    return rolled.mean() / rolled.std() * volatility_scale(ppy)


def rolling_beta(returns: ReturnsLike, benchmark: ReturnsLike, window: int) -> ReturnsLike:
    """ローリングベータ。"""
    aligned, bench = _align(returns, benchmark)
    var = bench.rolling(window).var()
    if isinstance(aligned, pd.DataFrame):
        cov = aligned.apply(lambda col: col.rolling(window).cov(bench))
        return cov.div(var, axis=0)
    return aligned.rolling(window).cov(bench) / var


def rolling_tracking_error(
    returns: ReturnsLike,
    benchmark: ReturnsLike,
    window: int,
    *,
    periods_per_year: float | None = None,
) -> ReturnsLike:
    """ローリング年率トラッキングエラー。"""
    active = active_returns(returns, benchmark)
    ppy = resolve_periods_per_year(active, periods_per_year)
    return active.rolling(window).std() * volatility_scale(ppy)


def rolling_information_ratio(
    returns: ReturnsLike,
    benchmark: ReturnsLike,
    window: int,
    *,
    periods_per_year: float | None = None,
) -> ReturnsLike:
    """ローリング情報比。"""
    active = active_returns(returns, benchmark)
    ppy = resolve_periods_per_year(active, periods_per_year)
    rolled = active.rolling(window)
    return rolled.mean() * ppy / (rolled.std() * volatility_scale(ppy))


# --------------------------------------------------------------------------
# まとめ
# --------------------------------------------------------------------------


def performance_summary(
    returns: ReturnsLike,
    benchmark: ReturnsLike | None = None,
    *,
    risk_free: float = 0.0,
    periods_per_year: float | None = None,
    var_level: float = 0.05,
) -> pd.Series | pd.DataFrame:
    """主要指標を 1 本にまとめる。

    Parameters
    ----------
    returns :
        Series なら Series を、DataFrame なら列ごとに計算した DataFrame（index=指標名）を返す。
    benchmark :
        指定すると相対指標（active_return / tracking_error / information_ratio /
        beta / alpha / up_capture / down_capture）を追加する。

    Returns
    -------
    pd.Series | pd.DataFrame
        index は英語の指標名。表示用のラベル付けは呼び出し側で行う。
    """
    if isinstance(returns, pd.DataFrame):
        return pd.DataFrame(
            {
                col: performance_summary(
                    returns[col],
                    benchmark,
                    risk_free=risk_free,
                    periods_per_year=periods_per_year,
                    var_level=var_level,
                )
                for col in returns.columns
            }
        )

    ppy = resolve_periods_per_year(returns, periods_per_year)
    metrics = {
        "total_return": total_return(returns),
        "annualized_return": annualized_return(returns, periods_per_year=ppy),
        "annualized_volatility": annualized_volatility(returns, periods_per_year=ppy),
        "sharpe_ratio": sharpe_ratio(returns, risk_free=risk_free, periods_per_year=ppy),
        "sortino_ratio": sortino_ratio(returns, risk_free=risk_free, periods_per_year=ppy),
        "calmar_ratio": calmar_ratio(returns, periods_per_year=ppy),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "win_loss_ratio": win_loss_ratio(returns),
        "skew": returns.skew(),
        "kurtosis": returns.kurtosis(),
        f"var_{var_level:g}": value_at_risk(returns, level=var_level),
        f"cvar_{var_level:g}": conditional_value_at_risk(returns, level=var_level),
        "n_periods": returns.count(),
        "periods_per_year": ppy,
    }

    if benchmark is not None:
        active = active_returns(returns, benchmark)
        metrics |= {
            "active_return": annualized_return(active, periods_per_year=ppy),
            "tracking_error": tracking_error(returns, benchmark, periods_per_year=ppy),
            "information_ratio": information_ratio(returns, benchmark, periods_per_year=ppy),
            "beta": beta(returns, benchmark),
            "alpha": alpha(returns, benchmark, risk_free=risk_free, periods_per_year=ppy),
            "up_capture": capture_ratio(returns, benchmark, side="up"),
            "down_capture": capture_ratio(returns, benchmark, side="down"),
        }

    return pd.Series(metrics, name=getattr(returns, "name", None))


# --------------------------------------------------------------------------
# 内部ヘルパー
# --------------------------------------------------------------------------


def _periodic_rate(annual_rate: float, periods_per_year: float) -> float:
    """年率を期間率に変換する（幾何）。"""
    if annual_rate == 0.0:
        return 0.0
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


def _nan_if_zero(value: Metric) -> Metric:
    """0 除算を NaN にする。負の分母（あり得ない）はそのまま通す。"""
    if isinstance(value, pd.Series):
        return value.replace(0, np.nan)
    return np.nan if value == 0 else value


def _align(returns: ReturnsLike, benchmark: ReturnsLike) -> tuple[ReturnsLike, pd.Series]:
    """リターンとベンチマークの日付を内側結合で揃える。

    ``benchmark`` が 1 列の DataFrame なら Series に落とす。
    """
    if not isinstance(benchmark, (pd.Series, pd.DataFrame)):
        raise ValidationError(
            f"benchmark は Series か DataFrame を渡してください: {type(benchmark).__name__}"
        )
    if isinstance(benchmark, pd.DataFrame):
        if benchmark.shape[1] != 1:
            raise ValidationError("benchmark の DataFrame は 1 列にしてください。")
        benchmark = benchmark.iloc[:, 0]
    if isinstance(returns, pd.DataFrame):
        aligned, bench = returns.align(benchmark, join="inner", axis=0)
        return aligned, bench
    aligned, bench = returns.align(benchmark, join="inner")
    return aligned, bench
