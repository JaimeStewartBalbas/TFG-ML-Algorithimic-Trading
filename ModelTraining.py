import random
import tkinter as tk
from tkinter import messagebox
import numpy as np
import math

from cryptography.hazmat.primitives import padding
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import pickle
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from DataRetrieval import HistoricalDataRetrieval
import json

Model = {
    "SVM": 0,
    "RANDOMFOREST": 1,
    "LSTM": 2,
    "GRADIENTBOOSTING": 3
}

class ModelTrainer(object):
    def __init__(self, full_data,window_size):
        self.x_train = None
        self.y_train = None
        self.model_id = self.__select_model()
        self.model = None
        self.data = full_data
        self.window_size = window_size
        self.model_path = None

    def __select_model(self):
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

        if self.model_id in [2]:
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
            self.model.fit(self.x_train, self.y_train, epochs=15, batch_size=32)
        elif self.model_id == 3:  # Gradient Boosting
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1,
                                                   random_state=42, loss='squared_error')
            self.model.fit(self.x_train, self.y_train)



    def test_model(self):
        predictions = self.model.predict(self.x_test)
        predictions = predictions.reshape(-1, 1)
        self.predictions = self.scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(self.predictions - self.y_test) ** 2)
        print("RMSE: " + str(rmse))


    def graph_predictions(self):
        plt.figure(figsize=(16, 8))
        plt.title('historical price')
        plt.plot(self.y_test[:-100])
        plt.plot(self.predictions[:-100])
        plt.xlabel('Days', fontsize=18)
        plt.ylabel('Close_Price', fontsize=18)
        plt.legend(['test', 'predictions'], loc='lower right')
        plt.show()

    def save_model(self):
        """This method saves the model in a file system, encrypted with AES-256."""
        model_type = list(Model.keys())[self.model_id]
        self.model_path = './models/' + model_type + '.enc'
        if self.model is not None:
            # Generate a random 32-byte key for AES-256
            key = os.urandom(32)
            # Generate a random 16-byte IV for AES-CBC
            iv = os.urandom(16)

            encrypted_model = self.encrypt_model(self.model, key, iv)

            # Load existing keys from JSON file
            key_iv_filename = './keys.json'
            if os.path.exists(key_iv_filename):
                with open(key_iv_filename, 'r') as key_file:
                    existing_keys = json.load(key_file)
            else:
                existing_keys = []

            # Check if the model_id already exists in the list
            existing_entry = next((entry for entry in existing_keys if entry["model_id"] == str(self.model_id)), None)

            if existing_entry:
                # Update existing entry
                existing_entry.update({"AES_KEY": key.hex(), "IV_KEY": iv.hex()})
            else:
                # Add new entry to the list
                existing_keys.append({"model_id": str(self.model_id), "AES_KEY": key.hex(), "IV_KEY": iv.hex()})

            # Write updated keys to JSON file
            with open(key_iv_filename, 'w') as key_file:
                json.dump(existing_keys, key_file)

            # Write the encrypted model to the file
            with open(self.model_path, 'wb') as file:
                file.write(len(key).to_bytes(1, byteorder='big'))  # Write the key size (32 bytes) as a header
                file.write(encrypted_model)

            print("Model saved successfully.")
        else:
            print("No model to save. Train a model first.")

    def encrypt_model(self, model, key, iv):
        """Serialize and encrypt the model."""
        if isinstance(model, (str, bytes)):  # If model is already serialized (for Keras models)
            serialized_model = model if isinstance(model, bytes) else model.encode()
        elif list(Model.keys())[self.model_id] in ['LSTM']:  # For TensorFlow Keras models
            serialized_model = model.to_json()
        else:  # For scikit-learn models
            serialized_model = pickle.dumps(model)


        # Convert serialized_model to bytes if it's not already
        if not isinstance(serialized_model, bytes):
            serialized_model = serialized_model.encode()

        # Pad the serialized model to a multiple of the block size
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(serialized_model) + padder.finalize()

        # Encrypt the padded serialized model
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_model = encryptor.update(padded_data) + encryptor.finalize()

        return encrypted_model


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    data = HistoricalDataRetrieval('IBEX', '^IBEX', start_date='2004-01-01', end_date='2024-01-01')
    model_trainer = ModelTrainer(data.stock)
    model_trainer.prepare_data()
    model_trainer.train_model()
    model_trainer.test_model()
    model_trainer.graph_predictions()
    model_trainer.save_model()
