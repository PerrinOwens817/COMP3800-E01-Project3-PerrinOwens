import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model, scalers, and feature names
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('feature_scaler.pkl', 'rb') as fs_file:
    feature_scaler = pickle.load(fs_file)
with open('target_scaler.pkl', 'rb') as ts_file:
    target_scaler = pickle.load(ts_file)
with open('feature_names.pkl', 'rb') as fn_file:
    feature_names = pickle.load(fn_file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    # Prepare input data
    input_data = {key: 0 for key in feature_names}
    input_data.update({
        'App Usage Time (min/day)': float(form_data['AppUsageTime']),
        'Screen On Time (hours/day)': float(form_data['ScreenOnTime']),
        'Number of Apps Installed': float(form_data['NumberOfAppsInstalled']),
        'Data Usage (MB/day)': float(form_data['DataUsage'])
    })

    # Update categorical variables
    device_model_key = f"Device Model_{form_data['DeviceModel']}"
    if device_model_key in input_data:
        input_data[device_model_key] = 1

    os_key = f"Operating System_{form_data['OperatingSystem']}"
    if os_key in input_data:
        input_data[os_key] = 1

    gender_key = f"Gender_{form_data['Gender']}"
    if gender_key in input_data:
        input_data[gender_key] = 1

    ubc_key = f"User Behavior Class_{form_data['UserBehaviorClass']}"
    if ubc_key in input_data:
        input_data[ubc_key] = 1

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure input data columns match training data columns
    input_df = input_df[feature_names]

    # Scale the input data using the same scaler
    scaled_input = feature_scaler.transform(input_df)

    # Get the prediction
    prediction_scaled = model.predict(scaled_input)

    # Inverse-transform the prediction to get the original scale
    prediction_original = target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

    return render_template("index.html", prediction_text=f'{prediction_original[0][0]:.2f} mAh')

if __name__ == "__main__":
    app.run(debug=True)



