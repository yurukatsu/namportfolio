"""long ⇄ wide 変換。

分析関数の入口で :func:`as_wide` を呼べば、以降は入力形式を意識せず
``index=date`` / ``columns=bid`` の行列として扱える。

    def rank_ic(factor, fwd_ret):
        f = as_wide(factor, "factor")
        r = as_wide(fwd_ret, "ret")
        f, r = f.align(r, join="inner")
        ...

日付・銘柄の突き合わせは ``DataFrame.align`` で済むため専用関数は置かない。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from .config import resolve_columns
from .errors import ValidationError

__all__ = ["as_wide", "to_long", "require_columns", "check_duplicates"]


def as_wide(
    data: pd.DataFrame | pd.Series,
    value: str | None = None,
    *,
    date_col: str | None = None,
    id_col: str | None = None,
) -> pd.DataFrame:
    """パネルを wide 形式（``index=date``, ``columns=bid``）に変換する。

    受け取る形は 3 種類。既に wide ならそのまま返す。

    ==================================== ==================
    入力                                  処理
    ==================================== ==================
    ``date`` / ``bid`` カラムを持つ long   ``pivot``
    ``MultiIndex (date, bid)``            ``unstack``
    ``DatetimeIndex`` を持つ wide          そのまま
    ==================================== ==================

    Parameters
    ----------
    data :
        変換対象。
    value :
        値として使うカラム名。``None`` なら候補が 1 つに定まる場合のみ自動で選ぶ。
    date_col, id_col :
        キーカラム名。``None`` なら :func:`~namportfolio.core.config.set_columns`
        で設定した既定を使う。

    Returns
    -------
    pd.DataFrame
        日付昇順にソート済み。``pivot`` の結果に生じる欠損は埋めない
        （ユニバース変動で穴が空くのは正常な状態で、扱いは分析ごとに異なるため）。
    """
    date_col, id_col = resolve_columns(date_col, id_col)

    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.MultiIndex):
            raise ValidationError(
                "Series を wide に変換するには MultiIndex (date, bid) が必要です。"
                " long 形式の DataFrame か wide 形式の DataFrame を渡してください。"
            )
        wide = _unstack(data, id_col)

    elif isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex):
            col = _resolve_value(data, value, exclude=())
            wide = _unstack(data[col], id_col)
        elif date_col in data.columns and id_col in data.columns:
            col = _resolve_value(data, value, exclude=(date_col, id_col))
            check_duplicates(data, [date_col, id_col])
            frame = data[[date_col, id_col, col]].copy()
            frame[date_col] = pd.to_datetime(frame[date_col])
            wide = frame.pivot(index=date_col, columns=id_col, values=col)
        else:
            wide = _ensure_datetime_index(data, date_col, id_col)

    else:
        raise ValidationError(f"DataFrame か Series を渡してください: {type(data).__name__}")

    wide = wide.sort_index()
    wide.index.name = date_col
    wide.columns.name = id_col
    return wide


def to_long(
    wide: pd.DataFrame,
    value: str = "value",
    *,
    date_col: str | None = None,
    id_col: str | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """wide 形式を long 形式に戻す。

    Parameters
    ----------
    wide :
        ``index=date``, ``columns=bid`` の DataFrame。
    value :
        値カラムにつける名前。
    dropna :
        欠損セルを落とすか。既定は ``True``（wide は直積なので穴が多い）。
    """
    date_col, id_col = resolve_columns(date_col, id_col)
    out = (
        wide.rename_axis(index=date_col, columns=id_col)
        .reset_index()
        .melt(id_vars=date_col, var_name=id_col, value_name=value)
    )
    if dropna:
        out = out.dropna(subset=[value])
    return out.sort_values([date_col, id_col], ignore_index=True)


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    context: str | None = None,
) -> None:
    """必須カラムの存在を確認する。

    ``KeyError: 'sector'`` より原因を特定しやすいメッセージを出す。
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        prefix = f"{context}: " if context else ""
        raise ValidationError(
            f"{prefix}必須カラムが見つかりません: {missing} (存在するカラム: {list(df.columns)})"
        )


def check_duplicates(df: pd.DataFrame, subset: Sequence[str]) -> None:
    """``subset`` の組み合わせが一意か確認する。

    重複したまま ``pivot`` すると ``ValueError: Index contains duplicate entries``
    という原因の分かりにくいエラーになるため、事前に具体例つきで知らせる。
    """
    mask = df.duplicated(subset=list(subset), keep=False)
    n_dup = int(mask.sum())
    if n_dup:
        examples = df.loc[mask, list(subset)].drop_duplicates().head(3).to_dict("records")
        raise ValidationError(f"{list(subset)} が重複しています（{n_dup} 件）例: {examples}")


def _unstack(series: pd.Series, id_col: str) -> pd.DataFrame:
    """MultiIndex Series を wide 化する。銘柄レベルは名前で探し、無ければ最終レベル。"""
    names = list(series.index.names)
    level = id_col if id_col in names else -1
    return series.unstack(level=level)


def _resolve_value(df: pd.DataFrame, value: str | None, exclude: Sequence[str]) -> str:
    """値カラムを決める。``value`` 未指定なら候補が 1 つに定まる場合のみ自動選択。"""
    if value is not None:
        require_columns(df, [value], context="as_wide")
        return value
    candidates = [c for c in df.columns if c not in exclude]
    if len(candidates) == 1:
        return candidates[0]
    raise ValidationError(f"value を指定してください。候補: {candidates}")


def _ensure_datetime_index(df: pd.DataFrame, date_col: str, id_col: str) -> pd.DataFrame:
    """wide とみなした DataFrame の index を DatetimeIndex に揃える。"""
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    try:
        return df.set_axis(pd.to_datetime(df.index), axis=0)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"wide 形式とみなしましたが index を日付として解釈できません。"
            f" long 形式で渡すなら '{date_col}' と '{id_col}' カラムが必要です"
            f" (存在するカラム: {list(df.columns)[:10]})"
        ) from exc
