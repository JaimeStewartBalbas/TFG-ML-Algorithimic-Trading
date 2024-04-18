import unittest
from src.PredictionGeneration import Predictor

class TestPredictor(unittest.TestCase):
    def setUp(self):
        self.model_path = "../models/SVM.enc"
        self.window_size = 30
        self.model_id = 0
        self.ticker = "^IBEX"
        self.start_date = "2023-01-01"
        self.end_date = "2023-12-31"
        self.num_days = 30
        self.predictor = Predictor(self.model_path, self.window_size, self.model_id, self.ticker, self.start_date, self.end_date)

    def test_load_model(self):
        # Test to see if model is loaded correctly
        self.predictor.load_model()
        self.assertIsNotNone(self.predictor.model, "Model not loaded successfully")

    def test_predict_future(self):
        # test to see if predictions match number of days.
        self.predictor.load_model()
        future_predictions = self.predictor.predict_future(self.num_days)
        self.assertEqual(len(future_predictions), self.num_days, "Number of predicted days does not match")

    def test_plot_values(self):
        # Test to validate the behaviour of the plot values method.
        actual_values = [100, 102, 105, 110, 115, 112, 108, 109, 111, 115]  # Example actual values
        future_predictions = [98, 100, 104, 108, 112, 116, 120, 124, 128, 132]  # Example predicted values
        try:
            self.predictor.plot_values(future_predictions, actual_values)
        except Exception as e:
            self.fail(f"plot_values method raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
