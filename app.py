from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset again to get encoders (for categorical mapping)
df = pd.read_csv("ds_salaries.csv")

# Recreate encoders (same as training)
from sklearn.preprocessing import LabelEncoder
encoders = {}
categorical_features = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_size']

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

@app.route("/")
def home():
    return {"message": "✅ Salary Prediction API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Extract inputs
        work_year = int(data.get("work_year", 2024))
        experience_level = data.get("experience_level", "Entry-level")
        employment_type = data.get("employment_type", "Full-time")
        job_title = data.get("job_title", "Data Scientist")
        employee_residence = data.get("employee_residence", "India")
        company_size = data.get("company_size", "Small")

        # Encode categorical features
        exp_encoded = encoders['experience_level'].transform([experience_level])[0] if experience_level in encoders['experience_level'].classes_ else 0
        emp_type_encoded = encoders['employment_type'].transform([employment_type])[0] if employment_type in encoders['employment_type'].classes_ else 0
        job_title_encoded = encoders['job_title'].transform([job_title])[0] if job_title in encoders['job_title'].classes_ else 0
        residence_encoded = encoders['employee_residence'].transform([employee_residence])[0] if employee_residence in encoders['employee_residence'].classes_ else 0
        company_size_encoded = encoders['company_size'].transform([company_size])[0] if company_size in encoders['company_size'].classes_ else 0

        # Arrange features
        features = np.array([[work_year, exp_encoded, emp_type_encoded, job_title_encoded, residence_encoded, company_size_encoded]])

        # Predict
        prediction = model.predict(features)[0]

        return jsonify({
            "inputs": data,
            "predicted_salary_usd": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)

