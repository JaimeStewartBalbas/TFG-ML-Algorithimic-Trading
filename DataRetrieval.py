import matplotlib.pyplot as plt
import yfinance as yf
import time
from datetime import datetime
import pandas as pd
class HistoricalDataRetrieval(object):
    def __init__(self, name, ticker, start_date, end_date):
        self.ticker = ticker
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self._days = (datetime.strptime(end_date,"%Y-%m-%d") - datetime.strptime(start_date,"%Y-%m-%d")).days

    # We create read-only attribute for the stock.
    @property
    def stock(self):
        # Retrieve stock data
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date,progress=False)
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data = stock_data.resample('B').ffill()
        return stock_data

    def plot_data(self):
        # Plot the closing price
        self.stock['Close'][-30:].plot(title='Stock Price of ' + self.name + ' for the last ' + str(30) + 'days.')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.show()







