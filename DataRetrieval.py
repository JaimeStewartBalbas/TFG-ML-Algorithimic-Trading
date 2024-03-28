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







