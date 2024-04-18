import unittest
from datetime import datetime
from src.DataRetrieval import HistoricalDataRetrieval

class TestHistoricalDataRetrieval(unittest.TestCase):
    def setUp(self):
        # Define test parameters
        self.ticker = '^IBEX'  # Example ticker symbol
        self.start_date = '2023-01-01'
        self.end_date = '2023-01-30'
        self.historical_data_retriever = HistoricalDataRetrieval('Ibex 35', self.ticker, self.start_date, self.end_date)

    def test_data_fetch_accuracy(self):
        # Test if data is fetched accurately
        data = self.historical_data_retriever.stock
        # Assert that data is not empty
        self.assertIsNotNone(data)
        #There should be 20 stock days in this timeframe
        self.assertEqual(data.shape[0], 20)

    def test_data_latency(self):
        # Test for minimal latency in data retrieval
        start_time = datetime.now()
        self.historical_data_retriever.stock
        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds()
        # Assert that latency is minimal (e.g., less than 1 second)
        self.assertLessEqual(latency, 1)

    def test_plot_data(self):
        # Test if plot_data method executes without errors
        try:
            self.historical_data_retriever.plot_data()
        except Exception as e:
            self.fail(f"plot_data method raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
