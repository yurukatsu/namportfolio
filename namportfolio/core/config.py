"""キーカラム名の既定値。

社内データのカラム名に合わせて、セッション開始時に一度変更する想定。

    >>> import namportfolio as npf
    >>> npf.set_columns(date="trade_date", id="barra_id")

.. note::
   他モジュールから ``from .config import DATE_COL`` と取り込むと
   :func:`set_columns` による変更が反映されない。必ず
   :func:`resolve_columns` を経由すること。
"""

from __future__ import annotations

__all__ = ["DATE_COL", "ID_COL", "set_columns", "resolve_columns"]

DATE_COL = "date"
ID_COL = "bid"


def set_columns(*, date: str | None = None, id: str | None = None) -> None:
    """キーカラム名の既定を変更する。``None`` の側は変更しない。"""
    global DATE_COL, ID_COL
    if date is not None:
        DATE_COL = date
    if id is not None:
        ID_COL = id


def resolve_columns(
    date_col: str | None = None,
    id_col: str | None = None,
) -> tuple[str, str]:
    """明示指定があればそれを、なければ既定を返す。"""
    return (
        DATE_COL if date_col is None else date_col,
        ID_COL if id_col is None else id_col,
    )
