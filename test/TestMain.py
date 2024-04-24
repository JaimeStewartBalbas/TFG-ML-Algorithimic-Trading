import unittest
from src.main import main

class TestMain(unittest.TestCase):
    def test_Main(self):
        result = main()
        self.assertEqual(result, 0)



if __name__ == '__main__':
    unittest.main()
