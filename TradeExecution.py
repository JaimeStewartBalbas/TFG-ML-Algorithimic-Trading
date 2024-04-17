import json
import threading
import time
import datetime
class Trader:
    def __init__(self, predictions,filepath):
        self.predictions = predictions
        self.filepath = filepath

    def maximize_profit(self):
        if not self.predictions or len(self.predictions) < 2:
            print("No profit can be made.")
            return 0

        actions = []
        total_profit = 0
        for i in range(len(self.predictions) - 1):
            if self.predictions[i] < self.predictions[i + 1]:
                # Buy when price is lower than the next day's price
                buy_price = self.predictions[i]
                sell_price = self.predictions[i + 1]
                profit = sell_price - buy_price
                total_profit += profit
                actions.append({"operation": "buy", "Day": i + 1})
                actions.append({"operation": "sell", "Day": i + 2})

        if total_profit > 0:
            actions.append({"Expected total gains": total_profit})

        # Write actions to a JSON file
        with open(self.filepath, "w") as json_file:
            json.dump(actions, json_file, indent=4)

        return total_profit

    def simulate_trades(self):
        with open(self.filepath, "r") as json_file:
            actions = json.load(json_file)

        for action in actions:
            if action.get("operation") == "buy":
                day = action.get("Day")
                print(f"Waiting until day {day} to buy...")
                target_date = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(
                    days=day - 1)
                while datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) < target_date:
                    pass
                print(f"Buying on day {day}.")

            elif action.get("operation") == "sell":
                day = action.get("Day")
                print(f"Waiting until day {day} to sell...")
                target_date = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(
                    days=day - 1)
                while datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) < target_date:
                    pass
                print(f"Selling on day {day}.")


        print("Simulation complete.")


