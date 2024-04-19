import unittest
from src.Analytics import Analytics
import json
import numpy as np

class TestAnalytics(unittest.TestCase):
    def setUp(self):
        self.predictions = [100, 102, 105, 110, 115, 112, 108, 109, 111, 115]  # Example predictions
        self.actual_values = [110, 100, 106, 108, 118, 110, 107, 108, 112, 118]  # Example actual values
        self.filepath = "test_actions.json"  # Example actions file path
        self.analytics = Analytics(self.predictions, self.actual_values, self.filepath)

    def tearDown(self):
        pass

    def test_read_actions_from_file(self):
        # Create a test actions file
        actions = [{"operation": "buy", "Day": 1}, {"operation": "sell", "Day": 2}]
        with open(self.filepath, "w") as json_file:
            json.dump(actions, json_file)

        self.analytics.read_actions_from_file()
        self.assertEqual(self.analytics.actions, actions, "Actions not read correctly from file")

    def test_calculate_actual_gains(self):
        expected_actual_gains = 10  # Example expected actual gains
        actual_gains = self.analytics.calculate_actual_gains()
        self.assertEqual(actual_gains, expected_actual_gains, "Actual gains calculation incorrect")

    def test_calculate_percentage_error(self):
        expected_percentage_error = 2.37
        percentage_error = self.analytics.calculate_percentage_error()
        self.assertAlmostEqual(round(percentage_error,2), expected_percentage_error, places=5,
                               msg="Percentage error calculation incorrect")

    def test_calculate_average_prediction(self):
        expected_average_prediction = 108.7
        average_prediction = self.analytics.calculate_average_prediction()
        self.assertEqual(average_prediction, expected_average_prediction, "Average prediction calculation incorrect")

    def test_calculate_average_actual_value(self):
        expected_average_actual_value = 109.7
        average_actual_value = self.analytics.calculate_average_actual_value()
        self.assertEqual(average_actual_value, expected_average_actual_value,
                         "Average actual value calculation incorrect")

    def test_calculate_rmse(self):
        expected_rmse = 3.66
        rmse = self.analytics.calculate_rmse()
        self.assertAlmostEqual(round(rmse,2), expected_rmse, places=5, msg="RMSE calculation incorrect")

    def test_calculate_directional_accuracy(self):
        expected_directional_accuracy =88.89
        directional_accuracy = self.analytics.calculate_directional_accuracy()
        self.assertAlmostEqual(round(directional_accuracy,2), expected_directional_accuracy, places=1,
                               msg="Directional accuracy calculation incorrect")

    def test_calculate_predictions_std(self):
        expected_predictions_std = 4.82  # Example expected predictions standard deviation
        predictions_std = self.analytics.calculate_predictions_std()
        self.assertAlmostEqual(round(predictions_std,2), expected_predictions_std, places=5,
                               msg="Predictions standard deviation calculation incorrect")

    def test_calculate_actual_values_std(self):
        expected_actual_values_std = 5.14
        actual_values_std = self.analytics.calculate_actual_values_std()
        self.assertAlmostEqual(round(actual_values_std,2), expected_actual_values_std, places=5,
                               msg="Actual values standard deviation calculation incorrect")

    def test_calculate_correlation_coefficient(self):
        expected_correlation_coefficient = 0.75
        correlation_coefficient = self.analytics.calculate_correlation_coefficient()
        self.assertAlmostEqual(round(correlation_coefficient,2), expected_correlation_coefficient, places=5,
                               msg="Correlation coefficient calculation incorrect")

    def test_calculate_mape(self):
        expected_mape = 2.35
        mape = self.analytics.calculate_mape()
        self.assertAlmostEqual(round(mape,2), expected_mape, places=5, msg="MAPE calculation incorrect")

    def test_calculate_volatility(self):
        expected_predicted_volatility = 3.06
        expected_actual_volatility = 6.31  # Example expected actual volatility
        predicted_volatility, actual_volatility = self.analytics.calculate_volatility()
        self.assertAlmostEqual(round(predicted_volatility,2), expected_predicted_volatility, places=5,
                               msg="Predicted volatility calculation incorrect")
        self.assertAlmostEqual(round(actual_volatility,2), expected_actual_volatility, places=5,
                               msg="Actual volatility calculation incorrect")

    def test_calculate_lag(self):
        expected_lag = -1.0  # Example expected lag
        lag = self.analytics.calculate_lag()
        self.assertAlmostEqual(lag, expected_lag, places=5, msg="Lag calculation incorrect")

if __name__ == "__main__":
    unittest.main()
