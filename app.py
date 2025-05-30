from venv import logger

import pandas as pd
from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('best_xgboost_model.pkl')
logger.info("Model loaded successfully")
logger.info(f"Model type: {type(model)}")
print(type(model))

@app.route('/')
def home():
    return render_template("index3.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form.to_dict()

    # Create base DataFrame with form inputs
    input_data = pd.DataFrame([{
        'CreditScore': float(form_data['CreditScore']),
        'Age': float(form_data['Age']),
        'Tenure': float(form_data['Tenure']),
        'Balance': float(form_data['Balance']),
        'NumOfProducts': int(form_data['NumOfProducts']),
        'EstimatedSalary': float(form_data['EstimatedSalary']),
        'HasCrCard': int(form_data['HasCrCard']),
        'IsActiveMember': int(form_data['IsActiveMember']),
        'Geography': form_data['Geography'],
        'Gender': form_data['Gender']
    }])

    # Convert categorical variables to numerical
    input_data['Geography'] = input_data['Geography'].map({'0': 'France', '1': 'Spain', '2': 'Germany'})
    input_data['Gender'] = input_data['Gender'].map({'0': 'Female', '1': 'Male'})

    # Calculate derived features (same as during training)
    input_data['BalanceSalaryRatio'] = input_data['Balance'] / (input_data['EstimatedSalary'] + 1e-6)
    input_data['TenureByAge'] = input_data['Tenure'] / input_data['Age']
    input_data['CreditScoreGivenAge'] = input_data['CreditScore'] / input_data['Age']

    # One-hot encode categorical variables
    input_data = pd.get_dummies(input_data, columns=['Geography', 'Gender'])

    # Ensure all expected columns are present
    expected_columns = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'EstimatedSalary', 'BalanceSalaryRatio', 'TenureByAge',
        'CreditScoreGivenAge', 'HasCrCard', 'IsActiveMember',
        'Geography_Germany', 'Geography_France', 'Geography_Spain',
        'Gender_Male', 'Gender_Female'
    ]

    # Add any missing columns with 0 values
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to exactly match training data
    input_data = input_data[expected_columns]

    # Apply any scaling/normalization you used during training
    # (If you used StandardScaler during training, load and apply it here)

    # Make prediction
    prediction = model.predict(input_data)

    # Return result
    output = "Customer will leave" if prediction[0] == 1 else "Customer will stay"
    return render_template('index3.html', prediction_text=output)

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)