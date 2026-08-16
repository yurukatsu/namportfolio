import pytest

from namportfolio.core import config
from namportfolio.core.config import resolve_columns, set_columns


@pytest.fixture(autouse=True)
def restore_columns():
    saved = (config.DATE_COL, config.ID_COL)
    yield
    set_columns(date=saved[0], id=saved[1])


def test_defaults():
    assert resolve_columns() == ("date", "bid")


def test_set_columns_partial_update():
    set_columns(date="trade_date")
    assert resolve_columns() == ("trade_date", "bid"), "指定しなかった側は維持される"


def test_explicit_wins_over_default():
    set_columns(date="trade_date", id="barra_id")
    assert resolve_columns(date_col="dt") == ("dt", "barra_id")
    assert resolve_columns("a", "b") == ("a", "b")
