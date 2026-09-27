from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/analysis/nifty50_analysis.csv")
OUTPUT_FILE = Path("data/reference/nifty50_classification.csv")


CLASSIFICATION = {
    "ADANIENT.NS": ("Industrials", "Conglomerate / Infrastructure"),
    "ADANIPORTS.NS": ("Services", "Ports & Logistics"),
    "APOLLOHOSP.NS": ("Healthcare", "Hospitals"),
    "ASIANPAINT.NS": ("Consumer Durables", "Paints"),
    "AXISBANK.NS": ("Financial Services", "Bank"),
    "BAJAJ-AUTO.NS": ("Automobile & Auto Components", "Automobiles"),
    "BAJFINANCE.NS": ("Financial Services", "NBFC"),
    "BAJAJFINSV.NS": ("Financial Services", "Diversified Financial Services"),
    "BEL.NS": ("Capital Goods", "Defence Electronics"),
    "BHARTIARTL.NS": ("Telecommunication", "Telecom"),
    "CIPLA.NS": ("Healthcare", "Pharmaceuticals"),
    "COALINDIA.NS": ("Oil, Gas & Consumable Fuels", "Coal"),
    "DRREDDY.NS": ("Healthcare", "Pharmaceuticals"),
    "EICHERMOT.NS": ("Automobile & Auto Components", "Automobiles"),
    "ETERNAL.NS": ("Consumer Services", "Internet / Consumer Services"),
    "GRASIM.NS": ("Construction Materials", "Diversified Materials"),
    "HCLTECH.NS": ("Information Technology", "IT Services"),
    "HDFCBANK.NS": ("Financial Services", "Bank"),
    "HDFCLIFE.NS": ("Financial Services", "Life Insurance"),
    "HINDALCO.NS": ("Metals & Mining", "Non-Ferrous Metals"),
    "HINDUNILVR.NS": ("Fast Moving Consumer Goods", "FMCG"),
    "ICICIBANK.NS": ("Financial Services", "Bank"),
    "ITC.NS": ("Fast Moving Consumer Goods", "FMCG"),
    "INFY.NS": ("Information Technology", "IT Services"),
    "INDIGO.NS": ("Services", "Airline"),
    "JSWSTEEL.NS": ("Metals & Mining", "Steel"),
    "JIOFIN.NS": ("Financial Services", "Financial Services"),
    "KOTAKBANK.NS": ("Financial Services", "Bank"),
    "LT.NS": ("Construction", "Engineering & Construction"),
    "M&M.NS": ("Automobile & Auto Components", "Automobiles"),
    "MARUTI.NS": ("Automobile & Auto Components", "Automobiles"),
    "MAXHEALTH.NS": ("Healthcare", "Hospitals"),
    "NESTLEIND.NS": ("Fast Moving Consumer Goods", "FMCG"),
    "NTPC.NS": ("Power", "Power Generation"),
    "ONGC.NS": ("Oil, Gas & Consumable Fuels", "Oil & Gas"),
    "POWERGRID.NS": ("Power", "Power Transmission"),
    "RELIANCE.NS": ("Oil, Gas & Consumable Fuels", "Integrated Energy"),
    "SBILIFE.NS": ("Financial Services", "Life Insurance"),
    "SHRIRAMFIN.NS": ("Financial Services", "NBFC"),
    "SBIN.NS": ("Financial Services", "Bank"),
    "SUNPHARMA.NS": ("Healthcare", "Pharmaceuticals"),
    "TCS.NS": ("Information Technology", "IT Services"),
    "TATACONSUM.NS": ("Fast Moving Consumer Goods", "FMCG"),
    "TMPV.NS": ("Automobile & Auto Components", "Automobiles"),
    "TATASTEEL.NS": ("Metals & Mining", "Steel"),
    "TECHM.NS": ("Information Technology", "IT Services"),
    "TITAN.NS": ("Consumer Durables", "Consumer Products"),
    "TRENT.NS": ("Consumer Services", "Retail"),
    "ULTRACEMCO.NS": ("Construction Materials", "Cement"),
    "WIPRO.NS": ("Information Technology", "IT Services"),
}


def main():
    stocks = pd.read_csv(INPUT_FILE)

    symbols = stocks["symbol"].tolist()

    missing = sorted(set(symbols) - set(CLASSIFICATION))
    extra = sorted(set(CLASSIFICATION) - set(symbols))

    if missing:
        raise ValueError(f"Missing classifications: {missing}")

    if extra:
        raise ValueError(f"Unexpected classifications: {extra}")

    classification_df = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "sector": CLASSIFICATION[symbol][0],
                "business_type": CLASSIFICATION[symbol][1],
            }
            for symbol in symbols
        ]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    classification_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Created: {OUTPUT_FILE}")
    print(f"Stocks classified: {len(classification_df)}")


if __name__ == "__main__":
    main()
