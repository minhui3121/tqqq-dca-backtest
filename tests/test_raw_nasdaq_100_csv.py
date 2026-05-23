from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import unittest

import pandas as pd


RAW_CSV_PATH = Path("data/raw/nasdaq_100_yahoo.csv")


def round_half_up(value: float, places: int = 2) -> float:
    quantizer = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quantizer, rounding=ROUND_HALF_UP))


class TestRawNasdaq100Csv(unittest.TestCase):
    def setUp(self) -> None:
        if not RAW_CSV_PATH.exists():
            self.fail(f"Missing raw CSV at {RAW_CSV_PATH}")

        self.frame = pd.read_csv(RAW_CSV_PATH)
        required_columns = ["date", "Open", "Close"]
        self.assertEqual(list(self.frame.columns), required_columns)

    def assert_row_matches(self, date: str, expected_open: float, expected_close: float) -> None:
        row = self.frame.loc[self.frame["date"] == date]
        self.assertFalse(row.empty, f"Missing row for {date}")

        actual_open = round_half_up(float(row.iloc[0]["Open"]), 2)
        actual_close = round_half_up(float(row.iloc[0]["Close"]), 2)

        self.assertEqual(actual_open, expected_open, f"Open mismatch for {date}")
        self.assertEqual(actual_close, expected_close, f"Close mismatch for {date}")

    def test_expected_rows_match_yahoo_values(self) -> None:
        self.assert_row_matches("1985-10-01", 110.62, 112.14)
        self.assert_row_matches("2015-07-28", 4539.47, 4560.23)
        self.assert_row_matches("2025-12-31", 25464.71, 25249.85)

    def test_first_and_last_row_dates(self) -> None:
        self.assertEqual(self.frame.iloc[0]["date"], "1985-10-01")
        self.assertEqual(self.frame.iloc[-1]["date"], "2025-12-31")


if __name__ == "__main__":
    unittest.main()
