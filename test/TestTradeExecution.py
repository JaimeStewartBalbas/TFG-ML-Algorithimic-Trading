import unittest
from src.TradeExecution import Trader
import os

class TestTrader(unittest.TestCase):
    def setUp(self):
        self.predictions = [100, 102, 105, 110, 115, 112, 108, 109, 111, 115]  # Example closing stock prices
        self.filepath = "../operations/actions.json"
        self.trader = Trader(self.predictions, self.filepath)

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_maximize_profit(self):
        expected_profit = 22
        profit = self.trader.maximize_profit()
        self.assertEqual(profit, expected_profit, "Profit calculation incorrect")


    def test_actions_file_created(self):
        self.trader.maximize_profit()
        self.assertTrue(os.path.exists(self.filepath), "Actions file not created")





if __name__ == "__main__":
    unittest.main()
