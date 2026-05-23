"""Plot processed TQQQ backfilled CSVs.

This script loads the processed CSVs produced by the backfill pipeline
and creates comparison charts (close price over time for fee variants).

Usage: python scripts/plot_processed.py
Outputs: outputs/plots/tqqq_close_comparison.png
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
# Use non-interactive backend so plotting works on CI / headless systems
matplotlib.use("Agg")

import pandas as pd


def find_processed_files(base: Path) -> dict[str, Path]:
    # Expect files named like tqqq_backfilled_0%.csv etc
    processed_dir = base / "data" / "processed"
    patterns = ["*_0%.csv", "*_5%.csv", "*_10%.csv"]
    found = {}
    for pat in patterns:
        for p in processed_dir.glob(pat):
            # infer label from filename
            name = p.stem
            if name.endswith("_0%"):
                found.setdefault("0%", p)
            elif name.endswith("_5%"):
                found.setdefault("5%", p)
            elif name.endswith("_10%"):
                found.setdefault("10%", p)
    return found


def find_ndx_raw(base: Path) -> Path | None:
    p = base / "data" / "raw" / "nasdaq_100_yahoo.csv"
    return p if p.exists() else None


def load_close_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"]) 
    if "date" not in df.columns:
        raise SystemExit(f"Missing 'date' column in {path}")
    df = df.set_index("date").sort_index()
    if "close" not in df.columns:
        raise SystemExit(f"Missing 'close' column in {path}")
    return df["close"].astype(float)


def plot_comparison(base: Path, files: dict[str, Path]) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        print("matplotlib is required: pip install matplotlib", file=sys.stderr)
        raise

    series = {}
    for label, path in sorted(files.items()):
        series[label] = load_close_series(path)

    # combine
    df = pd.DataFrame(series)

    outdir = base / "outputs" / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "tqqq_close_comparison.png"

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(ax=ax, title="TQQQ backfilled close comparison (fee variants)")
    ax.set_xlabel("Date")
    ax.set_ylabel("TQQQ Close (model scale)")
    ax.grid(True, linestyle=":")

    # If NDX raw exists, plot on a secondary y-axis to avoid overlap
    ndx_path = find_ndx_raw(base)
    if ndx_path:
        try:
            # Read flexibly: accept 'date' or 'Date' as the date column
            sample = pd.read_csv(ndx_path, nrows=1)
            date_col = None
            if 'date' in sample.columns:
                date_col = 'date'
            elif 'Date' in sample.columns:
                date_col = 'Date'

            if date_col:
                ndx = pd.read_csv(ndx_path, parse_dates=[date_col])
                ndx = ndx.set_index(date_col)
            else:
                ndx = pd.read_csv(ndx_path)
                # try to interpret first column as date index
                first = ndx.columns[0]
                ndx[first] = pd.to_datetime(ndx[first], errors='coerce')
                ndx = ndx.set_index(first)

            # prefer 'Close' or 'close'
            if "Close" in ndx.columns:
                ndx_close = ndx["Close"].rename("NDX Close")
            elif "close" in ndx.columns:
                ndx_close = ndx["close"].rename("NDX Close")
            else:
                ndx_close = ndx.iloc[:, 0].rename("NDX Close")

            # align to the model dates, interpolate missing values
            ndx_close = ndx_close.reindex(df.index).interpolate()
            ax2 = ax.twinx()
            ndx_close.plot(ax=ax2, color="gray", alpha=0.6, linewidth=1, label="NDX Close")
            ax2.set_ylabel("NDX Close")
            ax2.legend(loc="upper right")
            outpath = outdir / "tqqq_close_comparison_with_ndx.png"
        except Exception as exc:
            print(f"Warning: failed to read/plot NDX data ({exc}); continuing without it")

    plt.legend(title="ANNUAL_FEE", loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Wrote chart: {outpath}")
    return outpath


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    files = find_processed_files(base)
    if not files:
        print("No processed files found in data/processed. Run the backfill pipeline first.")
        raise SystemExit(2)
    plot_comparison(base, files)


if __name__ == "__main__":
    main()
