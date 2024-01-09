import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close'].to_frame()
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

def calculate_daily_returns(stock_prices):
    return stock_prices.pct_change().dropna()

def calculate_bollinger_bands(stock_prices, window=20, num_std=2):
    rolling_mean = stock_prices.rolling(window=window).mean()
    rolling_std = stock_prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_macd(stock_prices, short_window=12, long_window=26, signal_window=9):
    short_ema = stock_prices.ewm(span=short_window, adjust=False).mean()
    long_ema = stock_prices.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

def calculate_rsi(daily_returns, window=14):
    delta = daily_returns.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(stock_prices, window=14):
    if not isinstance(stock_prices, pd.DataFrame):
        print("Error: ATR calculation requires a DataFrame.")
        return None

    required_columns = ['High', 'Low', 'Close']

    if not all(col in stock_prices.columns for col in required_columns):
        print(f"Error: Required columns {required_columns} not found in DataFrame.")
        return None

    high_low = stock_prices['High'] - stock_prices['Low']
    high_close = abs(stock_prices['High'] - stock_prices['Close'].shift(1))
    low_close = abs(stock_prices['Low'] - stock_prices['Close'].shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = true_range.max(axis=1)

    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_volatility(daily_returns, window=252):
    return daily_returns.rolling(window=window).std() * (252 ** 0.5)

def stock_analyst_recommendation(average_daily_return, volatility):
    if average_daily_return > 0 and volatility < 0.02:
        return "Strong Buy"
    elif average_daily_return > 0 and volatility >= 0.02:
        return "Buy"
    elif average_daily_return < 0 and volatility < 0.02:
        return "Hold"
    else:
        return "Sell"

def plot_stock_prices(stock_prices, ticker):
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(12, 24), sharex=True)

    initial_price = stock_prices.iloc[0]
    current_price = stock_prices.iloc[-1]

    # Calculate percentage change
    percentage_change = ((current_price - initial_price) / initial_price) * 100

    # Plot Stock Prices
    axes[0].plot(stock_prices.index, stock_prices, color='blue', label='Stock Prices')
    axes[0].set_title(f'{ticker} Stock Prices Over Time')
    axes[0].set_ylabel('Stock Price')
    axes[0].legend()
    axes[0].grid(True)

    # Display initial and current prices
    axes[0].annotate(f'Initial Price: ${initial_price:.2f}', xy=(0.02, 0.9), xycoords='axes fraction', fontsize=10)
    axes[0].annotate(f'Current Price: ${current_price:.2f}', xy=(0.02, 0.85), xycoords='axes fraction', fontsize=10)
    axes[0].annotate(f'Percentage Change: {percentage_change:.2f}%', xy=(0.02, 0.80), xycoords='axes fraction', fontsize=10)

    # Plot Daily Returns
    daily_returns = calculate_daily_returns(stock_prices)
    axes[1].plot(daily_returns, color='green', label='Daily Returns')
    axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
    axes[1].set_title(f'{ticker} Daily Returns')
    axes[1].set_ylabel('Daily Return')
    axes[1].legend()
    axes[1].grid(True)

    # Display average daily return and volatility
    average_daily_return = daily_returns.mean()
    volatility = daily_returns.std()
    axes[1].annotate(f'Avg Daily Return: {average_daily_return:.4f}', xy=(0.02, 0.9), xycoords='axes fraction', fontsize=10)
    axes[1].annotate(f'Volatility: {volatility:.4f}', xy=(0.02, 0.85), xycoords='axes fraction', fontsize=10)

    # Plot Bollinger Bands
    upper_band, lower_band = calculate_bollinger_bands(stock_prices)
    axes[2].plot(stock_prices.index, stock_prices, label='Stock Prices', color='blue')
    axes[2].plot(upper_band.index, upper_band, label='Upper Bollinger Band', color='red', linestyle='--')
    axes[2].plot(lower_band.index, lower_band, label='Lower Bollinger Band', color='red', linestyle='--')
    axes[2].set_title(f'{ticker} Bollinger Bands')
    axes[2].set_ylabel('Price')
    axes[2].legend()
    axes[2].grid(True)

    # Plot MACD and Signal Line
    macd, signal_line = calculate_macd(stock_prices)
    axes[3].plot(stock_prices.index, macd, label='MACD', color='blue')
    axes[3].plot(stock_prices.index, signal_line, label='Signal Line', color='orange')
    axes[3].axhline(0, color='black', linestyle='-', linewidth=1)
    axes[3].set_title(f'{ticker} MACD and Signal Line')
    axes[3].set_xlabel('Date')
    axes[3].legend()
    axes[3].grid(True)

    # Display stock analyst recommendation
    recommendation = stock_analyst_recommendation(average_daily_return, volatility)
    axes[4].annotate(f'Recommendation: {recommendation}', xy=(0.02, 0.9), xycoords='axes fraction', fontsize=10)
    axes[4].axis('off')

    # Plot ATR
    atr = calculate_atr(stock_prices)
    axes[5].plot(atr, color='purple', label='ATR')
    axes[5].set_title(f'{ticker} Average True Range (ATR)')   
    axes[5].set_ylabel('ATR')
    axes[5].legend()
    axes[5].grid(True)

    # Plot Volatility
    vol = calculate_volatility(daily_returns)
    axes[6].plot(vol, color='brown', label='Volatility')
    axes[6].set_title(f'{ticker} Volatility')
    axes[6].set_xlabel('Date')
    axes[6].set_ylabel('Volatility')
    axes[6].legend()
    axes[6].grid(True)

    # Display stock analyst recommendation
    recommendation = stock_analyst_recommendation(average_daily_return, volatility)
    plt.suptitle(f'Stock Analyst Recommendation: {recommendation}', fontsize=14, y=0.92)

    plt.tight_layout()
    plt.show()

def main():
    stock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'DIS', 'JPM', 'TSLA', 'NFLX', 'NVDA', 'WMT', 'IBM', 'GE', 'CSCO']
    start_date = '2000-01-01'
    end_date = '2024-01-09'

    for ticker in stock_tickers:
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        if stock_data is not None:
            plot_stock_prices(stock_data, ticker)

if __name__ == "__main__":
    main()
