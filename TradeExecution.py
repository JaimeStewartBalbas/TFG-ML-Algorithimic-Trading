import json
import threading
import time


class TradeExecution:
    def __init__(self, predictions):
        self.predictions = predictions

    def maximize_profit(self, output_file):
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
            actions.append({"Total gains": total_profit})

        # Write actions to a JSON file
        with open(output_file, "w") as json_file:
            json.dump(actions, json_file, indent=4)

        return total_profit

    def simulate_trades(self, input_file):
        with open(input_file, "r") as json_file:
            actions = json.load(json_file)

        for action in actions:
            if action.get("operation") == "buy":
                day = action.get("Day")
                print(f"Waiting until day {day} to buy...")
                while True:
                    current_day = len(self.predictions)
                    if current_day >= day:
                        print(f"Buying on day {day}.")
                        # Perform buying action here
                        break
                    time.sleep(1)  # Check every second until the expected day
            elif action.get("operation") == "sell":
                day = action.get("Day")
                print(f"Waiting until day {day} to sell...")
                while True:
                    current_day = len(self.predictions)
                    if current_day >= day:
                        print(f"Selling on day {day}.")
                        # Perform selling action here
                        break
                    time.sleep(1)
        print("Simulation complete.")


if __name__ == '__main__':

    ibex_predictions = [10000, 12000, 8000, 15000, 10000]
    trade_execution = TradeExecution(ibex_predictions)
    profit = trade_execution.maximize_profit("/operations/actions.json")

    # Simulate trades on a separate thread
    trader = threading.Thread(target=trade_execution.simulate_trades, args=("/operations/actions.json",))
    trader.start()
