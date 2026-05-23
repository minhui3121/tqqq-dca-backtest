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

ENGINEERED_ROWS = {
    # Engineered row: 2010-02-10, derived from raw rows
    #   NDX 2010-02-09 close = 1753.8399658203125
    #   NDX 2010-02-10 open  = 1752.4599609375
    #   NDX 2010-02-10 close = 1749.760009765625
    #   NDX 2010-02-11 open  = 1747.550048828125
    #   NDX 2010-02-11 close = 1775.739990234375
    #   TQQQ 2010-02-11 open = 0.20343799889087677
    #
    # Formulas:
    #   close_factor(t+1) = 1 + 3 * (ndx_close[t+1] / ndx_close[t] - 1) - daily_fee
    #   close[t] = tqqq_open[t+1] / close_factor(t+1)
    #   overnight_factor(t) = 1 + 3 * (ndx_open[t] / ndx_close[t-1] - 1) - daily_fee
    #   open[t] = close[t-1] * overnight_factor(t)
    #
    # 0% fee
    #   daily_fee = 0.0
    #   close_factor(2010-02-11) = 1.0445432179106033
    #   close[2010-02-10] = 0.20705318956843666
    #   overnight_factor(2010-02-10) = 0.9976394570034212
    #   open[2010-02-10] = 0.20801615489423345
    #
    # 5% fee
    #   daily_fee = 0.0002035241051570047
    #   close_factor(2010-02-11) = 1.0443396938054463
    #   close[2010-02-10] = 0.20709354072560973
    #   overnight_factor(2010-02-10) = 0.9974359328982642
    #   open[2010-02-10] = 0.20805689115999584
    #
    # 10% fee
    #   daily_fee = 0.0004180098938665333
    #   close_factor(2010-02-11) = 1.044125208016737
    #   close[2010-02-10] = 0.2071360821958071
    #   overnight_factor(2010-02-10) = 0.9972214471095546
    #   open[2010-02-10] = 0.20809983873061952
    "0%": [
        ("2010-02-08", 0.205121975058992, 0.2018891961799838),
        ("2010-02-09", 0.2093602034802477, 0.208508347814395),
        ("2010-02-10", 0.2080161548942335, 0.2070531895684367),
    ],
    "5%": [
        ("2010-02-08", 0.20524512490099941, 0.2020097442842208),
        ("2010-02-09", 0.20944409867291794, 0.20859173436377207),
        ("2010-02-10", 0.2080568911599959, 0.2070935407256098),
    ],
    "10%": [
        ("2010-02-08", 0.205375014251666, 0.20213688893549767),
        ("2010-02-09", 0.20953256700642656, 0.20867966622037337),
        ("2010-02-10", 0.20809983873061957, 0.20713608219580715),
    ],
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

    def test_engineered_dates_match_known_values(self) -> None:
        for label, path in PROCESSED_FILES.items():
            with self.subTest(fee=label):
                frame = self.load_frame(path)
                for date, expected_open, expected_close in ENGINEERED_ROWS[label]:
                    self.assert_row_matches_exact(frame, date, expected_open, expected_close)


if __name__ == "__main__":
    unittest.main()
