from pathlib import Path
import io

import pandas as pd
import requests


NIFTY50_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"

DATA_DIR = Path(__file__).parent / "data"
NIFTY50_FILE = DATA_DIR / "nifty50.csv"


def update_nifty50() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
    }

    response = requests.get(
        NIFTY50_URL,
        headers=headers,
        timeout=15,
    )

    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))

    if "Symbol" not in df.columns:
        raise ValueError("NSE NIFTY 50 file does not contain a Symbol column.")

    symbols = df["Symbol"].dropna().str.strip()

    if len(symbols) != 50:
        raise ValueError(
            f"Expected 50 NIFTY 50 stocks, but found {len(symbols)}."
        )

    if symbols.duplicated().any():
        raise ValueError("NIFTY 50 universe contains duplicate symbols.")

    if (symbols == "").any():
        raise ValueError("NIFTY 50 universe contains blank symbols.")

    df.to_csv(NIFTY50_FILE, index=False)

    print(f"Updated NIFTY 50 universe: {len(symbols)} stocks")


def load_universe(name: str) -> list[str]:
    if name != "NIFTY50":
        raise ValueError(f"Unknown universe: {name}")

    if not NIFTY50_FILE.exists():
        raise FileNotFoundError(
            "NIFTY 50 universe file not found. "
            "Run update_nifty50() first."
        )

    df = pd.read_csv(NIFTY50_FILE)

    if "Symbol" not in df.columns:
        raise ValueError("Local NIFTY 50 file does not contain a Symbol column.")

    symbols = df["Symbol"].dropna().str.strip()

    if len(symbols) != 50:
        raise ValueError(
            f"Expected 50 NIFTY 50 stocks, but found {len(symbols)}."
        )

    if symbols.duplicated().any():
        raise ValueError("Local NIFTY 50 universe contains duplicate symbols.")

    if (symbols == "").any():
        raise ValueError("Local NIFTY 50 universe contains blank symbols.")

    return [f"{symbol}.NS" for symbol in symbols]
