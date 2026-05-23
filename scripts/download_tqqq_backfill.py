from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf


@dataclass(frozen=True)
class Symbols:
    leverage: str = "TQQQ"
    benchmark: str = "^NDX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download TQQQ and Nasdaq-100 data from Yahoo Finance, then backfill "
            "TQQQ prices before inception using the leveraged daily return formula."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where raw and processed CSV files will be saved.",
    )
    parser.add_argument(
        "--daily-fee",
        type=float,
        default=0.0005,
        help="Daily fee to subtract from the leveraged daily return (default: 0.0005).",
    )
    parser.add_argument(
        "--start-date",
        default="1985-10-01",
        help="Earliest Yahoo Finance date to request for the Nasdaq-100 series.",
    )
    parser.add_argument(
        "--end-date",
        default="2026-01-01",
        help="End date in YYYY-MM-DD format. Defaults to 2026-01-01.",
    )
    return parser.parse_args()


def download_history(symbol: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    frame = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )

    if frame.empty:
        raise RuntimeError(f"No data returned for {symbol}.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index.name = "date"
    return frame.sort_index()


def pick_price_columns(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "Open" not in frame.columns or "Close" not in frame.columns:
        raise KeyError("Both 'Open' and 'Close' must be available in the downloaded data.")

    return frame["Open"].astype(float), frame["Close"].astype(float)


def build_backfilled_series(
    benchmark: pd.DataFrame,
    leverage: pd.DataFrame,
    daily_fee: float,
) -> pd.DataFrame:
    _, benchmark_close = pick_price_columns(benchmark)
    leverage_open, leverage_close = pick_price_columns(leverage)

    benchmark_close = benchmark_close.rename("ndx_close")
    leverage_open = leverage_open.rename("tqqq_actual_open")
    leverage_close = leverage_close.rename("tqqq_actual_close")

    merged = pd.concat([benchmark_close, leverage_open, leverage_close], axis=1, join="outer").sort_index()
    merged["ndx_daily_return"] = merged["ndx_close"].pct_change()
    merged["leveraged_growth_factor"] = 1.0 + (merged["ndx_daily_return"] * 3.0) - daily_fee

    merged["tqqq_modeled_open"] = np.nan
    merged["tqqq_modeled_close"] = np.nan

    anchor_date = merged["tqqq_actual_close"].first_valid_index()
    if anchor_date is None:
        raise RuntimeError("Could not find any actual TQQQ rows to anchor the backfill.")

    anchor_pos = merged.index.get_loc(anchor_date)
    if isinstance(anchor_pos, slice):
        anchor_pos = anchor_pos.start

    merged.iloc[anchor_pos:, merged.columns.get_loc("tqqq_modeled_open")] = merged.iloc[anchor_pos:][
        "tqqq_actual_open"
    ].to_numpy()
    merged.iloc[anchor_pos:, merged.columns.get_loc("tqqq_modeled_close")] = merged.iloc[anchor_pos:][
        "tqqq_actual_close"
    ].to_numpy()

    dates = merged.index.to_list()
    model_open_values = merged["tqqq_modeled_open"].astype(float).to_numpy(copy=True)
    model_close_values = merged["tqqq_modeled_close"].astype(float).to_numpy(copy=True)
    factor_values = merged["leveraged_growth_factor"].astype(float).to_numpy(copy=True)

    for position in range(anchor_pos - 1, -1, -1):
        next_position = position + 1
        factor = factor_values[next_position]
        if pd.isna(factor):
            raise RuntimeError(
                f"Missing Nasdaq-100 return for {dates[next_position].date()} while backfilling TQQQ."
            )
        if factor <= 0:
            raise RuntimeError(
                f"Non-positive leveraged growth factor {factor} on {dates[next_position].date()}."
            )

        model_open_values[position] = model_open_values[next_position] / factor
        model_close_values[position] = model_close_values[next_position] / factor

    merged["tqqq_modeled_open"] = model_open_values
    merged["tqqq_modeled_close"] = model_close_values

    output = merged.reset_index()
    output["date"] = output["date"].dt.date.astype(str)
    return output[
        [
            "date",
            "tqqq_modeled_open",
            "tqqq_modeled_close",
        ]
    ].rename(columns={"tqqq_modeled_open": "open", "tqqq_modeled_close": "close"})


def write_csvs(output_dir: Path, benchmark: pd.DataFrame, leverage: pd.DataFrame, combined: pd.DataFrame) -> None:
    raw_dir = output_dir / "raw"
    processed_dir = output_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_columns = ["Open", "Close"]
    benchmark.reindex(columns=raw_columns).to_csv(raw_dir / "nasdaq_100_yahoo.csv", index_label="date")
    leverage.reindex(columns=raw_columns).to_csv(raw_dir / "tqqq_yahoo.csv", index_label="date")
    combined.to_csv(processed_dir / "tqqq_backfilled.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    symbols = Symbols()
    benchmark = download_history(symbols.benchmark, args.start_date, args.end_date)
    leverage = download_history(symbols.leverage, "2010-02-11", args.end_date)
    combined = build_backfilled_series(benchmark, leverage, args.daily_fee)
    write_csvs(output_dir, benchmark, leverage, combined)

    earliest = combined.iloc[0]["date"]
    latest = combined.iloc[-1]["date"]
    print(f"Saved raw benchmark data to {output_dir / 'raw' / 'nasdaq_100_yahoo.csv'}")
    print(f"Saved raw TQQQ data to {output_dir / 'raw' / 'tqqq_yahoo.csv'}")
    print(f"Saved backfilled series to {output_dir / 'processed' / 'tqqq_backfilled.csv'}")
    print(f"Backfilled series covers {earliest} through {latest}.")


if __name__ == "__main__":
    main()
