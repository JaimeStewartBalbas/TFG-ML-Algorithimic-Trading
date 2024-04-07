import io
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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
    "GRU": 6,
}

class PredictionGeneration:
    def __init__(self, model_file, scaler, window_size, model_id):
        self.model_file = model_file
        self.scaler = scaler
        self.window_size = window_size
        self.model_id = model_id
        self.model = None  # Initialize the model attribute

    def read_and_decrypt_model(self):
        """Read the encrypted model from the file system and decrypt it."""
        with open(self.model_file, 'rb') as file:
            # Read the key size (32 bytes) from the header
            key_size = int.from_bytes(file.read(1), byteorder='big')
            # Read the encrypted model data
            encrypted_model = file.read()

        # Read the keys from the JSON file
        with open("./keys.json", 'r') as key_file:
            keys = json.load(key_file)
            key = bytes.fromhex(keys["AES_KEY"])
            iv = bytes.fromhex(keys["IV_KEY"])

        # Decrypt the model
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_model = decryptor.update(encrypted_model) + decryptor.finalize()

        return io.BytesIO(decrypted_model)  # Return as binary stream

    def load_model(self, decrypted_model):
        """Load the decrypted model."""
        if list(Model.keys())[self.model_id] in ['LSTM', 'GRU']:  # For TensorFlow Keras models
            self.model = model_from_json(decrypted_model.read())  # Load directly from binary stream
        else:  # For scikit-learn models
            self.model = pickle.load(decrypted_model)  # Load directly from binary stream

    def predict_future(self, start_date, end_date):
        """Predict future stock prices."""
        data = yf.download('^IBEX', period='30d')

        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        num_days = (end_date - start_date).days

        last_window = data[-self.window_size:].copy()
        future_predictions = []
        for _ in range(num_days):
            window = np.array(last_window["Close"].values[-self.window_size:]).reshape(-1, 1)
            scaled_window = self.scaler.transform(window)

            if self.model_id in [2, 6]:  # LSTM or GRU
                scaled_window = np.reshape(scaled_window, (1, self.window_size, 1))
            else:
                scaled_window = scaled_window.T
            prediction = self.model.predict(scaled_window)
            prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))
            future_predictions.append(prediction[0][0])

            last_window.loc[len(last_window)] = {"Close": prediction[0][0]}

        return future_predictions

    def plot_values(self, values, actual_values, title="Predicted vs Actual", xlabel="Days", ylabel="Price"):
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


if __name__ == '__main__':
    scaler = MinMaxScaler()
    model_file = './models/SVM.enc'
    model_id = Model["SVM"]
    prediction_generator = PredictionGeneration(model_file=model_file, scaler=scaler, window_size=30, model_id=model_id)
    decrypted_model = prediction_generator.read_and_decrypt_model()
    model = prediction_generator.load_model(decrypted_model)
    training_data = yf.download('^IBEX', start='2004-01-01', end='2024-01-01')  # Adjust start and end dates
    prediction_generator.scaler.fit(training_data[['Close']])

    actual_values = yf.download('^IBEX', start='2024-01-01', end='2024-01-10')
    future_predictions = prediction_generator.predict_future(start_date='2024-01-01', end_date='2024-01-10')
    prediction_generator.plot_values(future_predictions, actual_values['Close'], title="Predicted vs Actual",
                                     xlabel="Days", ylabel="Price")
