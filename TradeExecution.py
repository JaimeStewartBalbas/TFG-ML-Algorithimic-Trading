class TradeExecution:

    def __init__(self, budget, predictions):
        self.budget = budget
        self.predictions = predictions
        self.trades = []
        self.current_stock = 0
        self.total_gains = 0

    def simulate_trading(self):
        for i, price in enumerate(self.predictions):
            if i == 0:
                continue  # Skip the first day, as there's no previous price to compare with
            if price > self.predictions[i - 1]:  # If the price is rising, buy
                self.__buy(price)
            elif price < self.predictions[i - 1]:  # If the price is falling, sell
                self.__sell(price)

    def __buy(self, price):
        shares_to_buy = min(self.budget // price, 1000)  # Maximum buy limit to prevent excessive exposure
        total_cost = shares_to_buy * price
        if shares_to_buy > 0:
            self.budget -= total_cost
            self.current_stock += shares_to_buy
            self.trades.append(('buy', shares_to_buy, price))

    def __sell(self, price):
        if self.current_stock > 0:
            self.total_gains += self.current_stock * price - (self.current_stock * self.predictions[self.predictions.index(price) - 1])
            self.budget += self.current_stock * price
            self.trades.append(('sell', self.current_stock, price))
            self.current_stock = 0

if __name__ == '__main__':
    # Example usage
    ibex_predictions = [10000, 12000, 8000, 15000, 10000]
    ibex_budget = 100000
    trade_execution = TradeExecution(ibex_budget, ibex_predictions)
    trade_execution.simulate_trading()
    print(trade_execution.trades)
    print("Total gains:", trade_execution.total_gains)
