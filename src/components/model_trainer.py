from sklearn.linear_model import LinearRegression
from src.utils.common import save_object
from sklearn.pipeline import Pipeline
import os

def train_model(X, y, preprocessor):

    model = LinearRegression()

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    full_pipeline.fit(X, y)

    os.makedirs("artifacts", exist_ok=True)
    save_object("artifacts/model.pkl", full_pipeline)

    return full_pipeline