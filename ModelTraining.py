import tkinter as tk
from tkinter import messagebox
import random
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU
import matplotlib.pyplot as plt

from DataRetrieval import HistoricalDataRetrieval

Model = {
    "SVM": 0,
    "RANDOMFOREST": 1,
    "LSTM": 2,
    "GRADIENTBOOSTING": 3,
    "GRU": 6,
}

class ModelTrainer(object):
    def __init__(self, full_data):
        self.x_train = None
        self.y_train = None
        self.model_id = self.select_model()
        self.model = None
        self.data = full_data

    def select_model(self):
        """This method is used to select the model through the Tkinter interface."""
        root = tk.Tk()
        root.title("Select Model")
        root.geometry("300x150")
        selected_model = tk.StringVar(root)
        selected_model.set("SVM")  # Default model
        model_menu = tk.OptionMenu(root, selected_model, *Model.keys())
        model_menu.pack(padx=10, pady=10)
        def select():
            model_name = selected_model.get()
            messagebox.showinfo("Selected Model", f"Selected Model: {model_name}")
            root.destroy()  # Close the window
            return Model[model_name]
        # Create a button to confirm the selection
        select_button = tk.Button(root, text="Select", command=select)
        select_button.pack(pady=5)
        root.mainloop()
        # Return the selected model
        return Model[selected_model.get()]

    def prepare_data(self):
        """This method prepares the data for training and validation."""
        # Filter only closing data.
        data = self.data.filter(["Close"])
        # Reshape to a numpy array.
        df = np.array(data).reshape(-1, 1)
        # We scale the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_df = self.scaler.fit_transform(df)
        training_data_len = math.ceil(len(scaled_df) * 0.8)
        train_data = scaled_df[0:training_data_len, :]
        x_train = []
        y_train = []
        self.window_size = 30
        for i in range(self.window_size, len(train_data)):
            x_train.append(train_data[i - self.window_size:i, 0])
            y_train.append(train_data[i, 0])
        test_data = scaled_df[training_data_len - self.window_size:, :]
        x_test = []
        for i in range(self.window_size, len(test_data)):
            x_test.append(test_data[i - self.window_size:i, 0])
        self.x_test = np.array(x_test)
        self.y_test = df[training_data_len:, :]
        self.x_train, self.y_train = np.array(x_train), np.array(y_train)
        # For LSTM and GRU models, reshape the data
        if self.model_id in [2, 6]:  # LSTM or GRU
            self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))
            self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], self.x_test.shape[1], 1))
        return self.x_train, self.y_train, self.x_test, self.y_test

    def train_model(self):
        if self.model_id == 0:  # SVM
            self.model = SVR(kernel='rbf')
            self.model.fit(self.x_train, self.y_train)
        elif self.model_id == 1:  # Random Forest
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(self.x_train, self.y_train)
        elif self.model_id == 2:  # LSTM
            self.model = Sequential()
            self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
            self.model.add(LSTM(units=50, return_sequences=False))
            self.model.add(Dense(units=25))
            self.model.add(Dense(units=1))
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            self.x_train = np.reshape(self.x_train,
                                      (self.x_train.shape[0], self.x_train.shape[1], 1))  # Reshape for LSTM
            self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=32)
        elif self.model_id == 3:  # Gradient Boosting
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1,
                                                   random_state=42, loss='squared_error')
            self.model.fit(self.x_train, self.y_train)
        elif self.model_id == 6:  # GRU
            self.model = Sequential()
            self.model.add(GRU(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
            self.model.add(GRU(units=50, return_sequences=False))
            self.model.add(Dense(units=25))
            self.model.add(Dense(units=1))
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            self.x_train = np.reshape(self.x_train,
                                      (self.x_train.shape[0], self.x_train.shape[1], 1))  # Reshape for GRU
            self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=32)


    def test_model(self):
        predictions = self.model.predict(self.x_test)
        predictions = predictions.reshape(-1, 1)
        self.predictions = self.scaler.inverse_transform(predictions)
        print(self.predictions)
        rmse = np.sqrt(np.mean(self.predictions - self.y_test) ** 2)
        print("RMSE: " + str(rmse))


    def graph_predictions(self):
        plt.figure(figsize=(16, 8))
        plt.title('historical price')
        plt.plot(self.y_test)
        plt.plot(self.predictions)
        plt.xlabel('Days', fontsize=18)
        plt.ylabel('Close_Price', fontsize=18)
        plt.legend(['test', 'predictions'], loc='lower right')
        plt.show()

    def predict_future(self, n_days):
        """Predict n days into the future."""
        # Get the last window of data from the training set
        last_window = self.data[-self.window_size:].values  # Convert DataFrame to NumPy array
        # Scale the last window
        last_window_scaled = self.scaler.transform(last_window.reshape(-1, 1))
        # Create a list to store the predicted values
        future_predictions = []

        for _ in range(n_days):
            # Reshape the last window for prediction
            x_future = last_window_scaled[-self.window_size:].reshape(1, -1)
            # Predict the next day's closing price
            prediction = self.model.predict(x_future)
            # Inverse transform the prediction
            prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))
            # Append the prediction to the list
            print(prediction)
            future_predictions.append(prediction[0][0])
            # Update the last window to include the predicted value
            last_window_scaled = np.append(last_window_scaled, prediction).reshape(-1, 1)[-self.window_size:]

        return future_predictions



if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    data = HistoricalDataRetrieval('IBEX', '^IBEX', 6*365) # Assuming HistoricalDataRetrieval is defined somewhere
    model_trainer = ModelTrainer(data.stock)
    model_trainer.prepare_data()
    model_trainer.train_model()
    model_trainer.test_model()
    model_trainer.graph_predictions()
    #print(model_trainer.predict_future(10))