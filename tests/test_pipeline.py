from datetime import datetime, timedelta
from pathlib import Path

from src.config import (
    MARKET_SYMBOL,
    ANALYSIS_YEARS,
)

from src.universe.stocks import (
    update_nifty50,
    load_universe,
)

from src.analysis.multi_stock import (
    analyze_stock_universe,
    results_to_dataframe,
)

from src.analysis.screener import filter_stocks


def get_analysis_dates(years):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    return (
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )


def main():
    start_date, end_date = get_analysis_dates(
        ANALYSIS_YEARS
    )

    print("=" * 60)
    print("MULTI-STOCK ANALYSIS PIPELINE TEST")
    print("=" * 60)

    # --------------------------------------------------
    # Update NIFTY 50 universe
    # --------------------------------------------------

    print("\nUPDATING NIFTY 50 UNIVERSE")

    update_nifty50()

    stocks = load_universe("NIFTY50")

    print(
        "Number of stocks:",
        len(stocks)
    )

    print(
        "Analysis period:",
        ANALYSIS_YEARS,
        "years"
    )

    print(
        "Start date:",
        start_date
    )

    print(
        "End date:",
        end_date
    )

    # --------------------------------------------------
    # Stock analysis
    # --------------------------------------------------

    print("\nANALYSING STOCKS")

    print(
        len(stocks),
        "STOCKS BEING ANALYSED"
    )

    print(
        "STOCK NAMES:",
        ", ".join(stocks)
    )

    results, failed_stocks = analyze_stock_universe(
        stocks,
        MARKET_SYMBOL,
        start_date,
        end_date,
    )

    print(
        "\nStocks successfully analysed:",
        len(results)
    )

    print(
        "Stocks failed:",
        len(failed_stocks)
    )

    if failed_stocks:
        print("\nFAILED STOCKS")

        for failure in failed_stocks:
            print(
                failure["symbol"],
                "->",
                failure["error"],
            )

    # --------------------------------------------------
    # Convert results to DataFrame
    # --------------------------------------------------

    analysis_df = results_to_dataframe(results)

    print("\nANALYSIS DATAFRAME")

    print(analysis_df)

    print(
        "\nDataFrame shape:",
        analysis_df.shape
    )

    # --------------------------------------------------
    # Save analysis dataset
    # --------------------------------------------------

    output_dir = Path("data") / "analysis"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_dir / "nifty50_analysis.csv"

    analysis_df.to_csv(
        output_file,
        index=False,
    )

    print(
        "\nAnalysis dataset saved to:",
        output_file
    )

    # --------------------------------------------------
    # Screening
    # --------------------------------------------------

    print("\nSCREENING")

    screened_df = filter_stocks(
        analysis_df,
        "roe",
        minimum=0.15,
    )

    print("\nROE >= 15%")

    print(screened_df)

    print(
        "\nStocks passing screen:",
        len(screened_df)
    )

    print("\n" + "=" * 60)
    print("MULTI-STOCK PIPELINE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
