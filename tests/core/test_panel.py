import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from namportfolio.core.errors import ValidationError
from namportfolio.core.panel import as_wide, check_duplicates, require_columns, to_long


@pytest.fixture
def long_df():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"]),
            "bid": ["JP1301", "JP1332", "JP1301", "JP1332"],
            "factor": [1.24, -0.31, 0.88, 0.45],
            "ret": [0.012, -0.003, 0.021, -0.008],
        }
    )


class TestAsWide:
    def test_from_long_columns(self, long_df):
        w = as_wide(long_df, "factor")
        assert list(w.index) == list(pd.to_datetime(["2024-01-31", "2024-02-29"]))
        assert list(w.columns) == ["JP1301", "JP1332"]
        assert w.index.name == "date"
        assert w.columns.name == "bid"
        assert w.loc["2024-02-29", "JP1332"] == 0.45

    def test_from_long_multiindex(self, long_df):
        mi = long_df.set_index(["date", "bid"])
        assert_frame_equal(as_wide(mi, "factor"), as_wide(long_df, "factor"))

    def test_from_multiindex_series(self, long_df):
        s = long_df.set_index(["date", "bid"])["factor"]
        assert_frame_equal(as_wide(s), as_wide(long_df, "factor"))

    def test_wide_passthrough(self, long_df):
        w = as_wide(long_df, "factor")
        assert_frame_equal(as_wide(w), w)

    def test_wide_with_string_index_is_converted(self):
        w = pd.DataFrame({"JP1301": [1.0, 2.0]}, index=["2024-01-31", "2024-02-29"])
        assert isinstance(as_wide(w).index, pd.DatetimeIndex)

    def test_string_dates_are_converted(self, long_df):
        long_df["date"] = long_df["date"].dt.strftime("%Y-%m-%d")
        assert isinstance(as_wide(long_df, "factor").index, pd.DatetimeIndex)

    def test_sorted_by_date(self, long_df):
        shuffled = long_df.iloc[[3, 0, 2, 1]].reset_index(drop=True)
        assert as_wide(shuffled, "factor").index.is_monotonic_increasing

    def test_missing_cells_are_kept_as_nan(self, long_df):
        """ユニバース変動で空くセルは埋めない。"""
        w = as_wide(long_df.drop(index=3), "factor")
        assert np.isnan(w.loc["2024-02-29", "JP1332"])

    def test_value_auto_selected_when_unambiguous(self, long_df):
        one_value = long_df[["date", "bid", "factor"]]
        assert_frame_equal(as_wide(one_value), as_wide(long_df, "factor"))

    def test_value_required_when_ambiguous(self, long_df):
        with pytest.raises(ValidationError, match="候補"):
            as_wide(long_df)

    def test_unknown_value_column(self, long_df):
        with pytest.raises(ValidationError, match="必須カラム"):
            as_wide(long_df, "nonexistent")

    def test_duplicate_keys_rejected(self, long_df):
        dup = pd.concat([long_df, long_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValidationError, match="重複"):
            as_wide(dup, "factor")

    def test_custom_column_names(self):
        df = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-31"]),
                "barra_id": ["JP1301"],
                "factor": [1.0],
            }
        )
        w = as_wide(df, "factor", date_col="trade_date", id_col="barra_id")
        assert w.index.name == "trade_date"
        assert w.columns.name == "barra_id"

    def test_plain_series_rejected(self):
        s = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
        with pytest.raises(ValidationError, match="MultiIndex"):
            as_wide(s)

    def test_undecodable_index_gives_actionable_message(self):
        w = pd.DataFrame({"a": [1.0]}, index=["not-a-date"])
        with pytest.raises(ValidationError, match="date"):
            as_wide(w)


class TestToLong:
    def test_roundtrip(self, long_df):
        wide = as_wide(long_df, "factor")
        back = to_long(wide, "factor")
        assert_frame_equal(back, long_df[["date", "bid", "factor"]], check_like=True)

    def test_dropna_default(self, long_df):
        wide = as_wide(long_df.drop(index=3), "factor")
        assert len(to_long(wide, "factor")) == 3
        assert len(to_long(wide, "factor", dropna=False)) == 4


class TestValidation:
    def test_require_columns_lists_missing_and_available(self, long_df):
        with pytest.raises(ValidationError) as exc:
            require_columns(long_df, ["sector", "mktcap"])
        assert "sector" in str(exc.value)
        assert "factor" in str(exc.value)

    def test_check_duplicates_counts_all_rows(self, long_df):
        dup = pd.concat([long_df, long_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValidationError, match="2 件"):
            check_duplicates(dup, ["date", "bid"])
