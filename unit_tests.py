import unittest
import csv

class TestCSVFile(unittest.TestCase):
    def setUp(self):
        # Set the path to the CSV file you want to test
        self.csv_file_path = 'data/X_eval_imputed.csv'

    def test_csv_not_empty(self):
        with open(self.csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            # Check if the file is not empty
            self.assertGreater(len(rows), 0, "CSV file is empty")

    def test_csv_has_20_columns(self):
        with open(self.csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Check if each row contains exactly 550 columns
                self.assertEqual(len(row), 550, f"Row does not have 550 columns: {row}")


if __name__ == '__main__':
    unittest.main()