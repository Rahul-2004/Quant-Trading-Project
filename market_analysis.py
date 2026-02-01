import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_market():
    # Define tickers
    tickers = {
        "USD Index": "DX-Y.NYB",
        "Gold": "GC=F",
        "US Treasury (TLT)": "TLT"
    }

    print("Fetching data...")
    # Download data for the last 5 years
    data = yf.download(list(tickers.values()), period="5y")["Close"]
    
    # Rename columns for easier access
    data.columns = [key for key, value in tickers.items() for col in data.columns if value == col]
    # Simple rename based on order might be risky if yfinance changes order, let's map properly
    # Re-download individually to be safe or map carefully. 
    # yfinance multi-download returns columns as (Price, Ticker). 
    # Let's do it safely:
    
    df = pd.DataFrame()
    for name, ticker in tickers.items():
        try:
            # Download individually to ensure correct mapping
            ticker_data = yf.download(ticker, period="5y", progress=False)
            if not ticker_data.empty:
                df[name] = ticker_data["Close"]
        except Exception as e:
            print(f"Error fetching {name}: {e}")

    # Drop NaN values to ensure alignment
    df.dropna(inplace=True)

    # 1. Normalized Growth (Base 100)
    normalized_df = (df / df.iloc[0]) * 100

    # Setup the plot
    fig = plt.figure(figsize=(15, 12))
    plt.style.use('bmh') # Use a nice style
    
    # Subplot 1: Relative Performance (Growth)
    ax1 = fig.add_subplot(3, 1, 1)
    for col in df.columns:
        ax1.plot(normalized_df.index, normalized_df[col], label=col, linewidth=2)
    ax1.set_title("5-Year Relative Growth (Base = 100)")
    ax1.set_ylabel("Normalized Price")
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: USD vs Treasury (Scatter with Trendline)
    ax2 = fig.add_subplot(3, 2, 3)
    sns.regplot(x=df["USD Index"], y=df["US Treasury (TLT)"], ax=ax2, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    ax2.set_title("USD Index vs US Treasury (TLT)")
    ax2.set_xlabel("USD Index Price")
    ax2.set_ylabel("Treasury ETF (TLT) Price")

    # Subplot 3: Rolling Correlation (Gold Independence)
    # We calculate rolling 6-month (approx 126 trading days) correlation
    ax3 = fig.add_subplot(3, 2, 4)
    rolling_corr_usd = df["Gold"].rolling(window=126).corr(df["USD Index"])
    rolling_corr_tlt = df["Gold"].rolling(window=126).corr(df["US Treasury (TLT)"])
    
    ax3.plot(rolling_corr_usd.index, rolling_corr_usd, label="Gold vs USD", color='purple')
    ax3.plot(rolling_corr_tlt.index, rolling_corr_tlt, label="Gold vs Treasuries", color='orange')
    ax3.axhline(0, color='black', linestyle='--', linewidth=1)
    ax3.set_title("Gold's Rolling Correlation (126-Day)")
    ax3.set_ylabel("Correlation Coefficient")
    ax3.legend()

    # Subplot 4: Rolling Correlation (USD vs Treasuries) - To show the specific interaction requested
    ax4 = fig.add_subplot(3, 1, 3)
    rolling_corr_usd_tlt = df["US Treasury (TLT)"].rolling(window=126).corr(df["USD Index"])
    ax4.plot(rolling_corr_usd_tlt.index, rolling_corr_usd_tlt, label="USD vs Treasuries", color='green')
    ax4.axhline(0, color='black', linestyle='--', linewidth=1)
    ax4.set_title("Rolling Correlation: USD vs Treasuries")
    ax4.set_ylabel("Correlation")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("market_relationships.png")
    print("Plot saved to market_relationships.png")

    # Correlation Matrix
    print("\n--- Overall Correlation Matrix (5 Years) ---")
    print(df.corr())

if __name__ == "__main__":
    analyze_market()
