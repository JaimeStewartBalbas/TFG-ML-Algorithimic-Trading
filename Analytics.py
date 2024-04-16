import json
import matplotlib.pyplot as plt


class Analytics:
    def __init__(self, predictions, actual_value):
        self.predictions = predictions
        self.actual_value = actual_value
        self.actions = []

    def read_actions_from_file(self, file_path):
        try:
            with open(file_path, "r") as json_file:
                self.actions = json.load(json_file)
        except FileNotFoundError:
            print(f"Error: Actions file '{file_path}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON in '{file_path}'. File may be improperly formatted.")

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



    def plot_graphs(self):
        if len(self.predictions) != len(self.actual_value):
            print("Error: Length of predictions and actual values do not match.")
            return

        days = range(1, len(self.predictions) + 1)
        plt.plot(days, self.predictions, label='Predictions', marker='o')
        plt.plot(days, self.actual_value, label='Actual Values', marker='x')
        plt.xlabel('Day')
        plt.ylabel('Value')
        plt.title('Predictions vs Actual Values')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # Example usage:
    predictions = [10000, 12000, 8000, 15000, 10000]
    actual_value = [11000, 13000, 9000, 14000, 11000]

    analytics = Analytics(predictions, actual_value)
    analytics.read_actions_from_file("operations/actions.json")

    print("Actual Gains:", analytics.calculate_actual_gains())
    print("Percentage Error:", analytics.calculate_percentage_error())
    print("Average Prediction:", analytics.calculate_average_prediction())
    print("Average Actual Value:", analytics.calculate_average_actual_value())
    analytics.plot_graphs()
