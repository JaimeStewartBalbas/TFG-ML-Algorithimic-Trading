import unittest
from src.Analytics import Analytics
import json

class TestAnalytics(unittest.TestCase):
    def setUp(self):
        self.predictions = [110, 102, 105, 110, 115, 112, 108, 109, 111, 115]
        self.actual_values = [120, 100, 106, 108, 118, 110, 107, 108, 112, 118]
        self.filepath = "../operations/actions.json"
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
        expected_actual_gains = 10
        actual_gains = self.analytics.calculate_actual_gains()
        self.assertEqual(actual_gains, expected_actual_gains, "Actual gains calculation incorrect")

    def test_calculate_percentage_error(self):
        expected_percentage_error = 2.3486901535682025
        percentage_error = self.analytics.calculate_percentage_error()
        self.assertAlmostEqual(percentage_error, expected_percentage_error, places=5,
                               msg="Percentage error calculation incorrect")

    def test_calculate_average_prediction(self):
        expected_average_prediction = 109.7
        average_prediction = self.analytics.calculate_average_prediction()
        self.assertEqual(average_prediction, expected_average_prediction, "Average prediction calculation incorrect")

    def test_calculate_average_actual_value(self):
        expected_average_actual_value = 110.7
        average_actual_value = self.analytics.calculate_average_actual_value()
        self.assertEqual(average_actual_value, expected_average_actual_value,
                         "Average actual value calculation incorrect")

if __name__ == "__main__":
    unittest.main()
