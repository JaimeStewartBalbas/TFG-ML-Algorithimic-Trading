from DataRetrieval import HistoricalDataRetrieval
from RealtimeRetrieval import RealTimeCandlestick
from ModelTraining import ModelTrainer
from PredictionGeneration import Predictor
import warnings

from TradeExecution import Trader

if __name__ == '__main__':
    # We retrieve data from Historical Data Retrieval Module to train our model.
    historical_data = HistoricalDataRetrieval(name="Ibex 35",
                                              ticker="^IBEX",
                                              start_date="2000-01-01",
                                              end_date="2024-01-01")

    actual_data = HistoricalDataRetrieval(name="Ibex 35",
                                          ticker="^IBEX",
                                          start_date="2024-01-02",
                                          end_date="2024-02-13")
    print(actual_data.stock)
    # We take an overview of our data.
    print(historical_data.stock)
    historical_data.plot_data()

    # We create our model trainer
    trainer = ModelTrainer(full_data=historical_data.stock, window_size=30)

    # We prepare and train our model.
    trainer.prepare_data()
    trainer.train_model()

    # We test our model and we plot our results. We store the model securely through encryption.
    trainer.test_model()
    trainer.graph_predictions()
    trainer.save_model()

    # We create the predictor model
    predictor = Predictor(model_path=trainer.model_path,
                          window_size=trainer.window_size,
                          model_id=trainer.model_id,
                          ticker='^IBEX',
                          start_date='2000-01-01',
                          end_date='2024-01-01')

    # We load the model in the predictor and predict the following 30 days.
    predictor.load_model()
    predictions = predictor.predict_future(30)
    print(predictions)
    print(list(actual_data.stock["Close"]))
    # We compare predictions with actual values.
    predictor.plot_values(predictions, list(actual_data.stock["Close"]))

    trader = Trader(predictions=predictions)
    trader.maximize_profit('./operations/actions.json')
