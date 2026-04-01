import pandas as pd
from src.utils.common import load_object

class PredictPipeline:

    def __init__(self):
        self.model = load_object("artifacts/model.pkl")

    def predict(self, data_dict):
        df = pd.DataFrame([data_dict])
        return self.model.predict(df)[0]