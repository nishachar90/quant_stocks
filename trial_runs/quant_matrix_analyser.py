import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# 1. Define the Multi-Dimensional Stock Map based on your notebook
stock_universe = {
    "RELIANCE.NS": {
        "Name": "Reliance Industries",
        "Size": "Large Cap",
        "Sector": "Energy / Consumer",
        "Model": "Capital-Intensive / Cyclical Mix",
        "Ownership": "Private Promoter",
        "Focus": "Domestic-Facing"
    },
    "TCS.NS": {
        "Name": "TCS",
        "Size": "Large Cap",
        "Sector": "IT Services",
        "Model": "Asset-Light / Defensive",
        "Ownership": "Private Promoter",
        "Focus": "Export-Oriented"
    },
    "HDFCBANK.NS": {
        "Name": "HDFC Bank",
        "Size": "Large Cap",
        "Sector": "Financials",
        "Model": "Financial Intermediation",
        "Ownership": "Widely Held",
        "Focus": "Domestic-Facing"
    },
    "COALINDIA.NS": {
        "Name": "Coal India",
        "Size": "Large Cap",
        "Sector": "Metals & Mining",
        "Model": "Capital-Intensive / Cyclical",
        "Ownership": "PSU (Government)",
        "Focus": "Domestic-Facing"
    },
    "BSE.NS": {
        "Name": "BSE Limited",
        "Size": "Mid Cap",
        "Sector": "Financials",
        "Model": "Asset-Light / Growth",
        "Ownership": "Widely Held",
        "Focus": "Domestic-Facing"
    }
}

tickers = list(stock_universe.keys())

print("📥 Ingesting 3 years of historical structural data from NSE...")
raw_data = yf.download(tickers, period="3y", threads=False) # 3 years captures market cycles well
close_prices = raw_data["Close"]

# 2. Compute Advanced Quant Metrics (Risk and Return Profiles)
daily_returns = close_prices.pct_change()

# Calculate Annualized Return (Geometric Mean approximation)
annualized_returns = (1 + daily_returns.mean()) ** 252 - 1

# Calculate Annualized Risk (Volatility)
annualized_volatility = daily_returns.std() * np.sqrt(252)

# Calculate Sharpe Ratio (Assuming a 6.5% Risk-Free Rate for India/RBI bonds)
RISK_FREE_RATE = 0.065
sharpe_ratios = (annualized_returns - RISK_FREE_RATE) / annualized_volatility

# 3. Compile the Complete Structural Matrix DataFrame
processed_rows = []
for ticker in tickers:
    meta = stock_universe[ticker]
    processed_rows.append({
        "Ticker": ticker.replace(".NS", ""),
        "Company Name": meta["Name"],
        "Size": meta["Size"],
        "Sector": meta["Sector"],
        "Model": meta["Model"],
        "Ownership": meta["Ownership"],
        "Focus": meta["Focus"],
        "Ann. Return (%)": annualized_returns[ticker] * 100,
        "Volatility (%)": annualized_volatility[ticker] * 100,
        "Sharpe Ratio": sharpe_ratios[ticker]
    })

df_matrix = pd.DataFrame(processed_rows)

print("\n" + "="*85)
print("                       THE QUANTITATIVE RETAIL INVESTOR MATRIX                 ")
print("="*85)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df_matrix.to_string(index=False, formatters={
    "Ann. Return (%)": "{:.2f}%".format,
    "Volatility (%)": "{:.2f}%".format,
    "Sharpe Ratio": "{:.2f}".format
}))
print("="*85)

# 4. Generate the Risk vs. Return Factor Matrix Visual
plt.figure(figsize=(10, 6))

# Plot each category combination explicitly on a matrix
for i, row in df_matrix.iterrows():
    # Choose color based on ownership type
    color = "red" if "PSU" in row["Ownership"] else ("blue" if "Private" in row["Ownership"] else "green")
    
    plt.scatter(row["Volatility (%)"], row["Ann. Return (%)"], s=250, color=color, alpha=0.7, edgecolors='black')
    
    # Text annotation overlaying the architectural characteristics
    label_text = f"{row['Ticker']}\n({row['Size']}-{row['Focus']})"
    plt.text(row["Volatility (%)"] + 0.4, row["Ann. Return (%)"] - 0.2, label_text, fontsize=9, fontweight='bold')

plt.title("Factor Matrix: Risk vs. Return Profiles by Business Archetype", fontsize=14, fontweight="bold")
plt.xlabel("Historical Volatility / Risk (%) — Higher means wilder swings", fontsize=11)
plt.ylabel("Annualized Returns (%) — Performance track record", fontsize=11)
plt.axhline(RISK_FREE_RATE * 100, color='gray', linestyle='--', alpha=0.7, label="India Risk-Free Rate (6.5%)")
plt.grid(True, linestyle=":", alpha=0.5)

print("\n🚀 Launching Factor Mapping matrix visualization...")
plt.tight_layout()
plt.show()
