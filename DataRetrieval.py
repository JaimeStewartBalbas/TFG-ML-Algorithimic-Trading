import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import threading,time
class HistoricalDataRetrieval(object):
    def __init__(self, name, ticker, days=365):
        self.ticker = ticker
        self.name = name
        self.days = days
        self.stock = yf.download(self.ticker, period=str(days) + 'd')

    def get_data(self):
        return self.stock

    def update_period(self,days):
        return HistoricalDataRetrieval(self.name, self.ticker, days)

    def plot_data(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.stock.index, self.stock['Close'], label=self.ticker)
        plt.title('Stock Price of ' + self.ticker + ' of last ' + str(self.days))
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()


def real_time_stock(symbol):
    while True:
        # Retrieve intraday stock data for the current day
        stock_data = yf.download(symbol, period='1d', interval='1m')

        # Plot the intraday price
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data.index, stock_data['Close'], label=symbol)
        plt.title('Intraday Stock Price of ' + symbol)
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.show()

        # Wait for some time before updating
        time.sleep(10)  # Update every minute

if __name__ == '__main__':
    last_year_data = HistoricalDataRetrieval('Ibex 35', '^IBEX', 365)
    last_year_data.plot_data()

    last_month_data = HistoricalDataRetrieval('Ibex 35', '^IBEX', 30)
    last_month_data.plot_data()

    last_week_data = HistoricalDataRetrieval('Ibex 35', '^IBEX', 7)
    last_week_data.plot_data()

    real_time_data = threading.Thread(target=real_time_stock,args=(last_year_data.ticker))



