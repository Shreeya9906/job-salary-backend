from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS   

app = Flask(__name__)
CORS(app)

# Dummy model (replace later with model.pkl)
class DummyModel:
    def predict(self, X):
        return [50000 + (1000 * X[0][2])]  # salary = 50k + 1k*experience

model = DummyModel()

@app.route("/")
def home():
    return {"message": "✅ Salary Prediction API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    job_title = int(data.get("job_title", 0))
    skills = int(data.get("skills", 0))
    experience = int(data.get("experience", 0))

    features = np.array([[job_title, skills, experience]])
    prediction = model.predict(features)[0]

    return jsonify({
        "inputs": data,
        "predicted_salary": round(float(prediction), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
