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
    # The values below were produced by the backfill using the overnight +
    # intraday decomposition of Nasdaq moves.  For transparency we show the
    # derivation for the engineered row on 2010-02-10 (the day immediately
    # before the first actual TQQQ row on 2010-02-11).
    #
    # Notation and formulas used:
    # - ndx_close_pos  = Nasdaq close on day t (2010-02-10)
    # - ndx_open_next  = Nasdaq open on day t+1 (2010-02-11)
    # - ndx_open_pos   = Nasdaq open on day t (2010-02-10)
    # - tqqq_open_next = actual TQQQ open on day t+1 (anchor = 2010-02-11)
    # - daily_fee      = 1 - (1 - ANNUAL_FEE)^(1/252) (converted from percent)
    #
    # Step 1 (overnight): recover previous-day TQQQ close from the next-day
    # open using the overnight leveraged factor
    #   overnight_return = ndx_open_next / ndx_close_pos - 1
    #   overnight_factor = 1 + 3 * overnight_return - daily_fee
    #   model_close_prev = tqqq_open_next / overnight_factor
    #
    # Step 2 (intraday): recover previous-day TQQQ open from that recovered close
    #   intraday_return = ndx_close_pos / ndx_open_pos - 1
    #   intraday_factor = 1 + 3 * intraday_return - daily_fee
    #   model_open_prev = model_close_prev / intraday_factor
    #
    # Concrete numbers used (from `data/raw/nasdaq_100_yahoo.csv` and
    # `data/raw/tqqq_yahoo.csv`):
    #   ndx_close (2010-02-10 close) = 1749.760009765625
    #   ndx_open_next (2010-02-11 open) = 1747.550048828125
    #   ndx_open (2010-02-10 open) = 1752.4599609375
    #   tqqq_open_next (2010-02-11 actual open) = 0.20343799889087677
    #
    # Using these values the backfill yields (rounded here for readability):
    # - ANNUAL_FEE = 0%  -> daily_fee = 0.0
    #   overnight_return = 1747.550048828125/1749.760009765625 - 1
    #                    ≈ -0.001262931
    #   overnight_factor ≈ 1 + 3*(-0.001262931) - 0 = 0.996211207
    #   model_close_prev = 0.20343799889087677 / 0.996211207
    #                    = 0.20421176217532735
    #   intraday_return = 1749.760009765625/1752.4599609375 - 1
    #                    ≈ -0.001529102
    #   intraday_factor ≈ 1 + 3*(-0.001529102) - 0 = 0.995412694
    #   model_open_prev = 0.20421176217532735 / 0.995412694
    #                  = 0.20516000981797294
    #
    # - ANNUAL_FEE = 5%  -> daily_fee = 0.0002035241051569680881677597
    #   overnight_return = -0.0012630080269099403559388795
    #   overnight_factor = 0.9960074518141132108440156018
    #   model_close_prev = 0.2042534907950114369001589469
    #   intraday_return = -0.0015406635427097734824763627
    #   intraday_factor = 0.9951744852667137114644031522
    #   model_open_prev = 0.2052438982499336033946819253
    #
    # - ANNUAL_FEE = 10% -> daily_fee = 0.0004180098938665269585106995
    #   overnight_return = -0.0012630080269099403559388795
    #   overnight_factor = 0.9957929660254036519736726620
    #   model_close_prev = 0.2042974853526800949335125416
    #   intraday_return = -0.0015406635427097734824763627
    #   intraday_factor = 0.9949599994780041525940602124
    #   model_open_prev = 0.2053323605570703691428423812
    "0%": ("2010-02-10", 0.20516000981797294, 0.20421176217532735),
    "5%": ("2010-02-10", 0.20524389824993358, 0.20425349079501143),
    "10%": ("2010-02-10", 0.20533236055707035, 0.20429748535268008),
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
