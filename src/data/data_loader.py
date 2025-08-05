import pandas as pd
import sys

def load_data(file_path):
    """Load dataset from CSV file with error handling."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the file is in the current directory or adjust the path.")
        sys.exit(1)