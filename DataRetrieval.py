import matplotlib.pyplot as plt
import yfinance as yf
import time
class HistoricalDataRetrieval(object):
    def __init__(self, name, ticker, days=365):
        self.ticker = ticker
        self.name = name
        self.days = days
        self.stock = self.retrieve_data()

    def retrieve_data(self):
        # Retrieve stock data
        stock_data = yf.download(self.ticker, period=str(self.days) + 'd')

        # Resample data to exclude weekends
        stock_data = stock_data.resample('B').ffill()

        return stock_data

    def plot_data(self):
        # Plot the closing price
        self.stock['Close'].plot(title='Stock Price of ' + self.name + ' for the last ' + str(self.days) + 'days.')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.show()


# def real_time_stock(symbol):
#     while True:
#         try:
#             # Retrieve intraday stock data for the current day
#             stock_data = yf.download(symbol, period='1d', interval='1m')
#
#             # Plot the intraday price
#             plt.figure(figsize=(10, 6))
#             plt.plot(stock_data.index, stock_data['Close'], label=symbol)
#             plt.title('Intraday Stock Price of ' + symbol)
#             plt.xlabel('Time')
#             plt.ylabel('Price (USD)')
#             plt.legend()
#             plt.grid(True)
#             plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
#             plt.tight_layout()  # Adjust layout to prevent clipping of labels
#             plt.show()
#
#             # Wait for some time before updating
#             time.sleep(10)  # Update every 10 seconds
#
#         except Exception as e:
#             print(e)
#             break








