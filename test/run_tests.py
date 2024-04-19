import unittest
import warnings
from test.TestDataRetrieval import TestHistoricalDataRetrieval
from test.TestRealtimeRetrieval import TestRealTimeCandlestick
from test.TestModelTrainer import TestModelTrainer
from test.TestPredictionGeneration import TestPredictor
from test.TestTradeExecution import TestTrader
from test.TestAnalytics import TestAnalytics

def suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestHistoricalDataRetrieval))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictor))
    suite.addTests(loader.loadTestsFromTestCase(TestTrader))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalytics))

    suite.addTests(loader.loadTestsFromTestCase(TestRealTimeCandlestick))

    return suite

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="oneDNN custom operations are on.*", category=UserWarning)
    warnings.filterwarnings("ignore", message="The tostring_rgb function was deprecated in Matplotlib*", category=UserWarning)
    runner = unittest.TextTestRunner()
    runner.run(suite())
