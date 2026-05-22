from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate buying TQQQ at each month-end using the processed backfilled price series."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/tqqq_backfilled.csv",
        help="Processed price CSV produced by scripts/download_tqqq_backfill.py.",
    )
    parser.add_argument(
        "--investment-amount",
        type=float,
        default=500.0,
        help="Dollar amount to invest at each month end (default: 500).",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional start date in YYYY-MM-DD format. Defaults to the first available date.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional end date in YYYY-MM-DD format. Defaults to the last available date.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/backtests/monthly_tqqq_dca.csv",
        help="CSV file to write the monthly investment ledger to.",
    )
    return parser.parse_args()


def load_prices(path: Path, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required_columns = {"date", "tqqq_modeled_close"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    frame = frame.sort_values("date").reset_index(drop=True)
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        frame = frame.loc[frame["date"] >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        frame = frame.loc[frame["date"] <= end_ts]

    if frame.empty:
        raise ValueError("No rows remain after applying the requested date range.")

    frame["price"] = pd.to_numeric(frame["tqqq_modeled_close"], errors="coerce")
    frame = frame.dropna(subset=["price"])
    return frame


def build_month_end_ledger(prices: pd.DataFrame, investment_amount: float) -> pd.DataFrame:
    month_end_rows = prices.groupby(prices["date"].dt.to_period("M"), sort=True).tail(1).copy()
    month_end_rows = month_end_rows.sort_values("date").reset_index(drop=True)

    month_end_rows["investment_amount"] = investment_amount
    month_end_rows["shares_bought"] = month_end_rows["investment_amount"] / month_end_rows["price"]
    month_end_rows["total_shares"] = month_end_rows["shares_bought"].cumsum()
    month_end_rows["total_invested"] = investment_amount * (month_end_rows.index + 1)
    month_end_rows["portfolio_value"] = month_end_rows["total_shares"] * month_end_rows["price"]
    month_end_rows["unrealized_gain"] = month_end_rows["portfolio_value"] - month_end_rows["total_invested"]
    month_end_rows["portfolio_multiple"] = month_end_rows["portfolio_value"] / month_end_rows["total_invested"]
    month_end_rows["portfolio_value_change"] = month_end_rows["portfolio_value"].diff()

    start_date = month_end_rows.loc[0, "date"]
    elapsed_years = (month_end_rows["date"] - start_date).dt.days / 365.2425
    month_end_rows["portfolio_peak"] = month_end_rows["portfolio_value"].cummax()
    month_end_rows["drawdown"] = (month_end_rows["portfolio_value"] / month_end_rows["portfolio_peak"]) - 1.0
    month_end_rows["max_drawdown_to_date"] = month_end_rows["drawdown"].cummin()
    growth_ratio = month_end_rows["portfolio_value"] / month_end_rows["total_invested"]
    month_end_rows["cagr"] = np.where(
        elapsed_years > 0,
        growth_ratio ** (1.0 / elapsed_years) - 1.0,
        np.nan,
    )

    ledger = month_end_rows[
        [
            "date",
            "price",
            "investment_amount",
            "shares_bought",
            "total_shares",
            "total_invested",
            "portfolio_value",
            "portfolio_value_change",
            "portfolio_peak",
            "drawdown",
            "max_drawdown_to_date",
            "cagr",
            "unrealized_gain",
            "portfolio_multiple",
            "tqqq_source",
        ]
    ].copy()
    ledger["date"] = ledger["date"].dt.date.astype(str)
    ledger = ledger.rename(columns={"date": "month_end_date"})
    return ledger


def write_output(ledger: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    prices = load_prices(input_path, args.start_date, args.end_date)
    ledger = build_month_end_ledger(prices, args.investment_amount)
    write_output(ledger, output_path)

    final_row = ledger.iloc[-1]
    print(f"Saved monthly DCA ledger to {output_path}")
    print(f"Backtest window: {ledger.iloc[0]['month_end_date']} through {ledger.iloc[-1]['month_end_date']}")
    print(
        f"Final portfolio value: {final_row['portfolio_value']:.2f} after investing {final_row['total_invested']:.2f}"
    )
    print(f"Final portfolio multiple: {final_row['portfolio_multiple']:.4f}x")
    print(f"Final CAGR: {final_row['cagr']:.4%}")
    print(f"Max drawdown: {ledger['drawdown'].min():.4%}")


if __name__ == "__main__":
    main()
