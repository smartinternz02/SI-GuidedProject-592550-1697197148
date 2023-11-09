# Import necessary libraries and modules
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np  # For handling NaN values
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask app
app = Flask(__name__)

# Load your machine learning model and other necessary data
model = joblib.load("RFS1.joblib")
ct = joblib.load('feature_values2')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        # Retrieve form data
        data = {
            'discharge_disposition_id': request.form["discharge_disposition_id"],
            'admission_source_id': request.form["admission_source_id"],
            'time_in_hospital': request.form["time_in_hospital"],
            'num_medications': request.form["num_medications"],
            'number_emergency': request.form["number_emergency"],
            'number_inpatient': request.form["number_inpatient"],
            'diag_1': request.form["diag_1"],
            'diag_2': request.form["diag_2"],
            'max_glu_serum': request.form["max_glu_serum"],
            'glimepiride': request.form["glimepiride"],
            'diabetesMed': request.form["diabetesMed"]
        }

        # Perform data preprocessing and transformation
        feature_cols = ['discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
                        'num_medications', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'max_glu_serum', 'glimepiride', 'diabetesMed']

        data_df = pd.DataFrame([data], columns=feature_cols)

        # Handle NaN values (replace NaN with appropriate values or drop rows)
        data_df = data_df.fillna(0)  # Replace NaN with 0, modify as needed

        # Create LabelEncoders for categorical features and transform the data
        for feature in ['discharge_disposition_id', 'admission_source_id', 'diag_1', 'diag_2', 'max_glu_serum', 'glimepiride', 'diabetesMed']:
            encoder = LabelEncoder()
            encoder.fit(data_df[feature])
            data_df[feature] = encoder.transform(data_df[feature])

        # Make predictions
        pred = model.predict(ct.transform(data_df))

        if pred[0] == 1:
            prediction = "This patient will be readmitted"
        else:
            prediction = "This patient will not be readmitted"

    return render_template("predict.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
