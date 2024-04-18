import unittest
import tkinter as tk
from src.RealtimeRetrieval import RealTimeCandlestick


class TestRealTimeCandlestick(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.real_time_candlestick = RealTimeCandlestick(self.root)

    def test_create_widgets(self):
        try:
            self.real_time_candlestick.create_widgets()
        except Exception as e:
            self.fail(f"Error creating GUI components: {e}")

    def test_update_graph(self):
        try:
            self.real_time_candlestick.update_graph()
        except Exception as e:
            self.fail(f"Error updating graph: {e}")

    def test_run(self):
        try:
            self.real_time_candlestick.run()
        except Exception as e:
            self.fail(f"Error running main loop: {e}")


if __name__ == '__main__':
    unittest.main()
