import yfinance as yf
import pandas as pd

# Download historical data for Bitcoin
btc_data = yf.download('BTC-USD', start='2015-01-01', end='2024-11-7', interval='1d')

# Save to CSV
btc_data.to_csv('bitcoin_historical_yfinance.csv')
print("Bitcoin historical data saved to 'bitcoin_historical_yfinance.csv'")
