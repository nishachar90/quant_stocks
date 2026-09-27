import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# 1. Configuration & Live Data Ingestion
tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
TOTAL_CAPITAL = 100000

print("Pulling historical market data from NSE...")
raw_data = yf.download(tickers, period="2y", threads=False)  # 2 years for stable long-term moving averages
close_prices = raw_data["Close"]

# 2. Part A: Automated Momentum Signal Scanner
print("\n" + "="*45)
print("     AUTOMATED ALGORITHMIC SIGNAL SCANNER     ")
print("="*45)

signals = {}
for ticker in tickers:
    clean_name = ticker.replace(".NS", "")
    
    # Calculate short-term (50-day) and long-term (200-day) moving averages
    sma_50 = close_prices[ticker].rolling(window=50).mean().iloc[-1]
    sma_200 = close_prices[ticker].rolling(window=200).mean().iloc[-1]
    current_price = close_prices[ticker].iloc[-1]
    
    # Algorithmic condition logic
    if sma_50 > sma_200:
        signal = "🟢 BULLISH (Golden Cross)"
    else:
        signal = "🔴 BEARISH (Death Cross)"
        
    signals[clean_name] = {"Price": current_price, "SMA50": sma_50, "SMA200": sma_200, "Signal": signal}
    print(f"{clean_name:<12} | Price: ₹{current_price:<8,.2f} | 50MA: ₹{sma_50:<8,.2f} | Status: {signal}")

# 3. Part B: Risk-Managed Portfolio Optimization (Inverse-Volatility Allocation)
# Calculate daily returns and risk (volatility) profile
daily_returns = close_prices.pct_change()
volatility = daily_returns.std() * np.sqrt(252)  # Annualized standard deviation

# Inverse Volatility Strategy: Allocate more money to steadier stocks, less to wild ones
inverse_vol = 1.0 / volatility
allocation_weights = inverse_vol / inverse_vol.sum()

print("\n" + "="*45)
print(f"   PORTFOLIO STRATEGY (Capital: ₹{TOTAL_CAPITAL:,})   ")
print("="*45)

portfolio_data = []
for ticker in tickers:
    clean_name = ticker.replace(".NS", "")
    weight = allocation_weights[ticker]
    cash_alloc = TOTAL_CAPITAL * weight
    current_price = signals[clean_name]["Price"]
    shares_to_buy = int(cash_alloc // current_price)
    
    portfolio_data.append({
        "Stock": clean_name,
        "Weight %": weight * 100,
        "Cash Allocated": cash_alloc,
        "Shares to Purchase": shares_to_buy
    })
    
    print(f"{clean_name:<12} | Allocation: {weight*100:>5.1f}% | Capital: ₹{cash_alloc:<8,.2f} | Buy: {shares_to_buy} shares")

# 4. Generate Strategy Visualizations
df_port = pd.DataFrame(portfolio_data)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Asset Allocation Breakdown
colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c']
ax1.pie(df_port["Weight %"], labels=df_port["Stock"], autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'w'})
ax1.set_title("Risk-Managed Asset Allocation (%)", fontsize=12, fontweight="bold")

# Subplot 2: Capital Deployment Model
bars = ax2.bar(df_port["Stock"], df_port["Cash Allocated"], color="darkslate%s" % "gray", edgecolor="black")
ax2.set_title("Capital Distribution Per Asset (₹)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Capital Investment (₹)")
ax2.grid(axis='y', linestyle=':', alpha=0.6)

# Tag exact cash value inside the bar chart
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height/2, f"₹{int(height):,}", ha='center', va='center', color='white', fontweight='bold')

plt.suptitle("Quantitative Portfolio Execution Architecture", fontsize=14, fontweight="bold")
plt.tight_layout()
print("\nLaunching visualization windows...")
plt.show()
