import json
import matplotlib.pyplot as plt
import numpy as np

class Analytics:
    def __init__(self, predictions, actual_value,filepath):
        self.predictions = predictions
        self.actual_value = actual_value
        self.filepath = filepath
        self.actions = []
    def read_actions_from_file(self):
        try:
            with open(self.filepath, "r") as json_file:
                self.actions = json.load(json_file)
        except FileNotFoundError:
            print(f"Error: Actions file '{self.filepath}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON in '{self.filepath}'. File may be improperly formatted.")

    def calculate_actual_gains(self):
        if len(self.predictions) != len(self.actual_value):
            print("Error: Length of predictions and actual values do not match.")
            return None

        actual_gains = 0
        for pred, actual in zip(self.predictions, self.actual_value):
            actual_gains += actual - pred

        return actual_gains

    def calculate_percentage_error(self):
        if len(self.predictions) != len(self.actual_value):
            print("Error: Length of predictions and actual values do not match.")
            return None

        error_sum = 0
        for pred, actual in zip(self.predictions, self.actual_value):
            error_sum += abs(actual - pred)

        return (error_sum / sum(self.actual_value)) * 100

    def calculate_average_prediction(self):
        if not self.predictions:
            print("Error: No predictions available.")
            return None

        return sum(self.predictions) / len(self.predictions)

    def calculate_average_actual_value(self):
        if not self.actual_value:
            print("Error: No actual values available.")
            return None

        return sum(self.actual_value) / len(self.actual_value)

    def calculate_rmse(self):
        if len(self.predictions) != len(self.actual_value):
            print("Error: Length of predictions and actual values do not match.")
            return None

        squared_errors = [(actual - pred) ** 2 for pred, actual in zip(self.predictions, self.actual_value)]
        mean_squared_error = np.mean(squared_errors)
        rmse = np.sqrt(mean_squared_error)
        return rmse

    def calculate_directional_accuracy(self):
        correct_directions = sum((np.sign(np.diff(self.predictions)) == np.sign(np.diff(self.actual_value))))
        total_directions = len(self.actual_value) - 1  # Total number of directions (excluding the first prediction)
        directional_accuracy = correct_directions / total_directions * 100
        return directional_accuracy

    def calculate_predictions_std(self):
        if not self.predictions:
            print("Error: No predictions available.")
            return None

        predictions_std = np.std(self.predictions)
        return predictions_std

    def calculate_actual_values_std(self):
        if not self.actual_value:
            print("Error: No actual values available.")
            return None

        actual_values_std = np.std(self.actual_value)
        return actual_values_std

    def calculate_correlation_coefficient(self):
        correlation = np.corrcoef(self.actual_value, self.predictions)[0, 1]
        return correlation

    def calculate_mape(self):
        if len(self.predictions) != len(self.actual_value):
            print("Error: Length of predictions and actual values do not match.")
            return None

        mape = np.mean(
            np.abs((np.array(self.actual_value) - np.array(self.predictions)) / np.array(self.actual_value))) * 100
        return mape

    def calculate_volatility(self):
        predicted_volatility = np.std(np.diff(self.predictions))  # Standard deviation of differences
        actual_volatility = np.std(np.diff(self.actual_value))
        return predicted_volatility, actual_volatility

    def calculate_lag(self):
        lag = np.mean(np.array(self.predictions) - np.array(self.actual_value))
        return lag

