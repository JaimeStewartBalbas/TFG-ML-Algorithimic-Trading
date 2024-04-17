import json
import matplotlib.pyplot as plt


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




