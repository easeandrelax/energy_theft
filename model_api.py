#model_api 

import joblib
import numpy as np
import pandas as pd # <-- NEW: Import Pandas to handle structured features
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configuration ---
MODEL_PATH = 'energy_theft_model.joblib'

# The 6 features the model expects, in the EXACT order they were trained:
FEATURE_NAMES = [
    'cons_mean',
    'cons_total',
    'diff_std',
    'lag1_corr',
    'month_std',
    'cons_total_zscore'
]

# 1. Load the model once when the server starts
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Note: In a production environment, you would log the error and might not exit immediately.
    exit()


app = Flask(__name__)
# Allow your frontend origin.
CORS(app, origins=["http://localhost:5173"])

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data sent from the client
    data = request.get_json(force=True)
    
    # 2. Extract and structure features using Pandas DataFrame
    try:
        # Create a dictionary to hold the feature values for one customer
        feature_dict = {}
        
        # Pull the values from the input data (assuming input keys match FEATURE_NAMES)
        for name in FEATURE_NAMES:
            if name not in data:
                # If a required feature is missing, raise a clear error
                return jsonify({
                    "error": f"Missing required feature: '{name}'"
                }), 400
            feature_dict[name] = [data[name]] # Wrap value in a list for the DataFrame constructor

        # Convert the dictionary into a Pandas DataFrame.
        # This guarantees the columns are present and in the correct order (FEATURE_NAMES).
        features_df = pd.DataFrame(feature_dict, columns=FEATURE_NAMES)
        
    except Exception as e:
        # Catch JSON parsing errors or other data-related issues
        return jsonify({
            "error": "Invalid input format or data type.",
            "details": str(e)
        }), 400

    # 3. Make the prediction
    # The XGBoost model can predict directly on the DataFrame
    prediction = model.predict(features_df)[0]
    
    # Get the probability for class 1 (Theft)
    probability = model.predict_proba(features_df)[0][1]

    # Convert native NumPy types to standard Python types for JSON serialization
    prediction_result = int(prediction)
    probability_result = float(probability)
    
    # 4. Return the result as JSON
    response = {
        'prediction': prediction_result,  # 1 for Theft, 0 for No Theft
        'probability': probability_result,
        'is_theft': bool(prediction_result)
    }

    return jsonify(response)

if __name__ == '__main__':
    # Run the Flask app on a specific port (e.g., 5000)
    app.run(host='127.0.0.1', port=5000)

