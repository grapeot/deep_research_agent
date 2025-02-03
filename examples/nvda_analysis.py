import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define the time period: one month from today (2025-02-03) back to 2025-01-03
end_date = '2025-02-03'
start_date = '2025-01-03'

# Fetch NVDA data
ticker = 'NVDA'
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    print('No data fetched.')
    exit()

# Determine which price column to use: Prefer 'Adj Close' if available, otherwise 'Close'
price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'

# Calculate daily returns
data['Daily Change'] = data[price_column].pct_change() * 100

# Identify big moves: daily move > 5% in absolute value
big_moves = data[abs(data['Daily Change']) > 5]

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, data[price_column], label=f'NVDA {price_column}')

# Mark sudden moves
for date, row in big_moves.iterrows():
    plt.axvline(x=date, color='red', linestyle='--', alpha=0.5)
    # Convert daily change value to float to ensure proper formatting
    daily_change = float(row['Daily Change'])
    plt.text(date, row[price_column], f"{daily_change:.1f}%", color='red', fontsize=8, rotation=90, verticalalignment='bottom')

plt.title('NVDA Stock Price Trend (Jan 3, 2025 - Feb 3, 2025)')
plt.xlabel('Date')
plt.ylabel(f'{price_column} (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig('nvda_trend.png')
plt.close()

# Print summary
print('NVDA stock trend analysis for the period {} to {}'.format(start_date, end_date))
print('Number of big moves (>5% change):', len(big_moves))
if not big_moves.empty:
    print(big_moves[[price_column, 'Daily Change']])
else:
    print('No big moves detected.')
