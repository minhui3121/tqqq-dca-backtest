from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import unittest

import pandas as pd


PROCESSED_FILES = {
    "0%": Path("data/processed/tqqq_backfilled_0%.csv"),
    "5%": Path("data/processed/tqqq_backfilled_5%.csv"),
    "10%": Path("data/processed/tqqq_backfilled_10%.csv"),
}

COMMON_ROWS = {
    "2010-02-11": (0.20, 0.22),
    "2020-03-30": (5.70, 6.13),
    "2025-12-31": (54.13, 52.72),
}

ENGINEERED_FIRST_ROW = {
    "0%": ("2010-02-10", 0.1947626439984103, 0.2070531895684367),
    "5%": ("2010-02-10", 0.19480059993657192, 0.2070935407256098),
    "10%": ("2010-02-10", 0.19484061617217055, 0.20713608219580715),
}


def round_half_up(value: float, places: int = 2) -> float:
    quantizer = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quantizer, rounding=ROUND_HALF_UP))


class TestProcessedTqqqBackfilledCsv(unittest.TestCase):
    def load_frame(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            self.fail(f"Missing processed CSV at {path}")

        frame = pd.read_csv(path)
        self.assertEqual(list(frame.columns), ["date", "open", "close"])
        return frame

    def assert_row_matches(self, frame: pd.DataFrame, date: str, expected_open: float, expected_close: float) -> None:
        row = frame.loc[frame["date"] == date]
        self.assertFalse(row.empty, f"Missing row for {date}")

        actual_open = round_half_up(float(row.iloc[0]["open"]), 2)
        actual_close = round_half_up(float(row.iloc[0]["close"]), 2)

        self.assertEqual(actual_open, expected_open, f"Open mismatch for {date}")
        self.assertEqual(actual_close, expected_close, f"Close mismatch for {date}")

    def assert_row_matches_exact(self, frame: pd.DataFrame, date: str, expected_open: float, expected_close: float) -> None:
        row = frame.loc[frame["date"] == date]
        self.assertFalse(row.empty, f"Missing row for {date}")

        actual_open = float(row.iloc[0]["open"])
        actual_close = float(row.iloc[0]["close"])

        self.assertAlmostEqual(actual_open, expected_open, places=15, msg=f"Open mismatch for {date}")
        self.assertAlmostEqual(actual_close, expected_close, places=15, msg=f"Close mismatch for {date}")

    def test_common_raw_match_rows_are_preserved(self) -> None:
        for label, path in PROCESSED_FILES.items():
            with self.subTest(fee=label):
                frame = self.load_frame(path)
                for date, (expected_open, expected_close) in COMMON_ROWS.items():
                    self.assert_row_matches(frame, date, expected_open, expected_close)

    def test_first_engineered_date_matches_known_values(self) -> None:
        for label, path in PROCESSED_FILES.items():
            with self.subTest(fee=label):
                frame = self.load_frame(path)
                date, expected_open, expected_close = ENGINEERED_FIRST_ROW[label]
                self.assert_row_matches_exact(frame, date, expected_open, expected_close)


if __name__ == "__main__":
    unittest.main()
