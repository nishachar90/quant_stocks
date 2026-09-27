import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# 1. Define tickers and time period
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
period = "1y"  # Fetch 1 full year of historical data

print(f"Fetching {period} of historical data from NSE...")
# Download data sequentially to avoid cache locks
raw_data = yf.download(tickers, period=period, threads=False)

# Extract only the Closing prices
close_prices = raw_data["Close"]

print("\n--- Data Summary Loaded ---")
print(f"Total trading days analyzed: {len(close_prices)}")

# 2. Quantitative Metric: Calculate Volatility (Risk)
# Calculate daily percentage change, then find the standard deviation (annualized)
daily_returns = close_prices.pct_change()
annualized_volatility = daily_returns.std() * np.sqrt(252) * 100

print("\n--- Annualized Volatility (Risk Metric) ---")
for ticker in tickers:
    clean_name = ticker.replace(".NS", "")
    print(f"{clean_name}: {annualized_volatility[ticker]:.2f}%")

# 3. Trend Indicator: Calculate Moving Averages for a primary stock
# Let's focus on RELIANCE for the detailed trend plot
target_stock = "RELIANCE.NS"
clean_target = target_stock.replace(".NS", "")

df_target = pd.DataFrame({"Close": close_prices[target_stock]})
df_target["50_Day_SMA"] = df_target["Close"].rolling(window=50).mean()

# 4. Generate a Professional Historical Trend Plot
plt.figure(figsize=(12, 6))

# Plot active stock price vs its moving average
plt.plot(df_target.index, df_target["Close"], label=f"{clean_target} Closing Price", color="darkblue", linewidth=1.5)
plt.plot(df_target.index, df_target["50_Day_SMA"], label="50-Day Moving Average (Trend Line)", color="orange", linestyle="--", linewidth=2)

# Chart aesthetics
plt.title(f"{clean_target} Share Price Trend & 50-Day SMA (Past 1 Year)", fontsize=14, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (₹)", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left", fontsize=10)

plt.tight_layout()
print(f"\nRendering historical trend graph for {clean_target}...")
plt.show()
