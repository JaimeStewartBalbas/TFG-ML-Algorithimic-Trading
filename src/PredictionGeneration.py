import io
import json
import numpy as np
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json
import pickle
import yfinance as yf

Model = {
    "SVM": 0,
    "RANDOMFOREST": 1,
    "LSTM": 2,
    "GRADIENTBOOSTING": 3,
    "GRU": 4,
}

class Predictor:
    def __init__(self, model_path:str, window_size, model_id,ticker,start_date,end_date):
        self.model_file = model_path
        self.scaler = MinMaxScaler()
        self.window_size = window_size
        self.model_id = model_id
        self.data = yf.download(ticker,
                                start=start_date,
                                end=end_date,
                                progress=False).filter(['Close'])
        self.model = None
        self.scaler.fit_transform(np.array(self.data).reshape(-1, 1))

    def read_and_decrypt_model(self):
        """Read the encrypted model from the file system and decrypt it."""
        with open(self.model_file, 'rb') as file:
            key_size = int.from_bytes(file.read(1), byteorder='big')
            encrypted_model = file.read()

        with open("../keys.json", 'r') as key_file:
            keys = json.load(key_file)
            key_entry = next((entry for entry in keys if entry["model_id"] == str(self.model_id)), None)
            if key_entry:
                key = bytes.fromhex(key_entry["AES_KEY"])
                iv = bytes.fromhex(key_entry["IV_KEY"])
            else:
                raise ValueError(f"Keys for model_id {self.model_id} not found.")


        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_model = decryptor.update(encrypted_model) + decryptor.finalize()

        return decrypted_model

    def load_model(self):
        """Load the decrypted model."""
        model = self.read_and_decrypt_model()
        if list(Model.keys())[self.model_id] in ['LSTM']:  # For TensorFlow Keras models
            self.model = model_from_json(model)
        else:  # For scikit-learn models
            self.model = pickle.loads(model)

    def predict_future(self, num_days):
        """Predict future stock prices."""
        last_window = self.data[-self.window_size:].copy()
        future_predictions = []
        for _ in range(num_days):
            # Extract only the closing price from the last window
            window = np.array(last_window["Close"].values[-self.window_size:]).reshape(-1, 1)
            scaled_window = self.scaler.transform(window)

            # Reshape for models expecting sequences (LSTM, GRU)
            if self.model_id in [2, 4]:  # LSTM or GRU
                scaled_window = np.reshape(scaled_window, (1, self.window_size, 1))
            else:
                scaled_window = scaled_window.T
            prediction = self.model.predict(scaled_window)
            prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))
            future_predictions.append(prediction[0][0])

            # Append the predicted value to the last window
            last_window.loc[len(last_window)] = {"Close": prediction[0][0]}

        return future_predictions

    def plot_values(self, values, actual_values, title="Predicted vs Actual", xlabel="Days", ylabel="Closing Price (USD)"):
        """Plot predicted and actual values."""
        if len(values) != len(actual_values):
            raise ValueError(f"Lengths of values ({len(values)}) and actual_values ({len(actual_values)}) do not match")

        print("Length of predicted values:", len(values))
        print("Length of actual values:", len(actual_values))


        plt.figure(figsize=(10, 6))
        plt.plot(values, label='Predicted', color='blue')
        plt.plot(actual_values, label='Actual', color='green')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()


