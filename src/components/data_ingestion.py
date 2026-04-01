import pandas as pd
from src.utils.exception import CustomException
import sys

def load_data(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise CustomException(e, sys)