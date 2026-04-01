import joblib
import os

def save_object(file_path, obj):
    """Saves a python object to a specific path using joblib."""
    try:
        # Create the directory if it doesn't exist (e.g., artifacts/)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            joblib.dump(obj, file)
            
    except Exception as e:
        print(f"Error occurred while saving object to {file_path}")
        raise e


def load_object(file_path):
    """Loads a python object from a specific path using joblib."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at {file_path}")
            
        with open(file_path, "rb") as file:
            return joblib.load(file)
            
    except Exception as e:
        print(f"Error occurred while loading object from {file_path}")
        raise e