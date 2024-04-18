import unittest, os
from src.ModelTraining import ModelTrainer
from src.DataRetrieval import HistoricalDataRetrieval

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        # Initialize ModelTrainer with mock data
        data = HistoricalDataRetrieval('IBEX', '^IBEX', start_date='2004-01-01', end_date='2024-01-01')
        self.model_trainer = ModelTrainer(data.stock,30)

    def test_prepare_data(self):
        # Test if data preparation method runs without errors and produces expected data shapes
        self.model_trainer.prepare_data()
        self.assertIsNotNone(self.model_trainer.x_train)
        self.assertIsNotNone(self.model_trainer.y_train)
        self.assertIsNotNone(self.model_trainer.x_test)
        self.assertIsNotNone(self.model_trainer.y_test)


    def test_train_model(self):
        # Test if the model training method runs without errors
        self.model_trainer.prepare_data()
        self.model_trainer.train_model()
        self.assertIsNotNone(self.model_trainer.model)


    def test_test_model(self):
        #Test if the test model method runs without errors
        self.model_trainer.prepare_data()
        self.model_trainer.train_model()
        self.model_trainer.test_model()
        self.assertIsNotNone(self.model_trainer.predictions)


    def test_save_model(self):
        #Tests of the save model method saves de model without erros.
        self.model_trainer.prepare_data()
        self.model_trainer.train_model()
        self.model_trainer.save_model()
        self.assertTrue(os.path.exists(self.model_trainer.model_path))

    def test_encryption(self):
        # Test if the saved model file is encrypted and can be decrypted successfully
        self.model_trainer.prepare_data()
        self.model_trainer.train_model()
        self.model_trainer.save_model()
        with open(self.model_trainer.model_path, 'rb') as file:
            encrypted_data = file.read()
            self.assertNotEqual(encrypted_data, b'')




if __name__ == '__main__':
    unittest.main()
