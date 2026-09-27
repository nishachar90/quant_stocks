import matplotlib.pyplot as plt
import yfinance as yf

# 1. Define the Indian Stock Tickers (NSE)
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

print("Fetching live data from NSE...")
# 2. Download the latest 1 day of historical data
data = yf.download(tickers, period="1d", threads=False)

# 3. Extract the closing prices for each stock
# We access the 'Close' column and look at the last available row
latest_prices = data["Close"].iloc[-1]

# Clean up ticker names for the chart labels (removing the '.NS' suffix)
clean_labels = [ticker.replace(".NS", "") for ticker in tickers]
prices_list = [latest_prices[ticker] for ticker in tickers]

# 4. Print the fetched live prices to the terminal
print("\n--- Live NSE Closing Prices (₹) ---")
for stock, price in zip(clean_labels, prices_list):
    print(f"{stock}: ₹{price:,.2f}")

# 5. Generate the Matplotlib Chart
plt.figure(figsize=(10, 5))
bars = plt.bar(clean_labels, prices_list, color="teal", edgecolor="black")

# Add price text tags on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 10,
        f"₹{height:,.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.title("Live Closing Prices of Top Indian Blue-Chip Stocks", fontsize=14, fontweight="bold")
plt.xlabel("Company Ticker (NSE)", fontsize=12)
plt.ylabel("Stock Price (₹)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)

print("\nRendering chart...")
plt.tight_layout()
plt.show()