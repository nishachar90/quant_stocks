from pathlib import Path

import pandas as pd
import yfinance as yf


CACHE_DIR = Path("data/cache")


def get_cache_path(
    symbol,
    start,
    end,
    auto_adjust,
):
    adjustment = "adjusted" if auto_adjust else "raw"

    filename = (
        f"{symbol}"
        f"__{start}"
        f"__{end}"
        f"__{adjustment}.csv"
    )

    return CACHE_DIR / filename


def download_stock_data(
    symbol,
    start,
    end,
    auto_adjust=True,
):
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = get_cache_path(
        symbol,
        start,
        end,
        auto_adjust,
    )

    # --------------------------------------------------
    # Check local cache
    # --------------------------------------------------

    if cache_path.exists():
        return pd.read_csv(
            cache_path,
            index_col=0,
            parse_dates=True,
        )

    # --------------------------------------------------
    # Download from Yahoo Finance
    # --------------------------------------------------

    ticker = yf.Ticker(symbol)

    data = ticker.history(
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        timeout=30,
    )

    # --------------------------------------------------
    # Save downloaded data
    # --------------------------------------------------

    data.to_csv(cache_path)

    return data

