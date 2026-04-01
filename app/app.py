from flask import Flask, request, jsonify
from src.pipeline.predict_pipeline import PredictPipeline
import os
print("Current Working Directory:", os.getcwd())
print("Files in src:", os.listdir('src') if os.path.exists('src') else "src not found")

app = Flask(__name__)

predictor = PredictPipeline()

@app.route("/")
def home():
    return "ML API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result = predictor.predict(data)
    return jsonify({"Final_Exam_Score": result})

if __name__ == "__main__":
    app.run(debug=True)