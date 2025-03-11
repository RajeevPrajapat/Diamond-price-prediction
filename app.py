from flask import Flask, render_template, request, Response
import pandas as pd
import numpy as np
import pickle 

app = Flask(__name__)

# Load Scaler and Model
try:
    Scaler = pickle.load(open('models/standard_scaler.pkl', 'rb'))
    model = pickle.load(open('models/RFmodel.pkl', 'rb')) 
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    Scaler = None

# Define feature names (must match training data)
feature_names = ['carat','cut','color','clarity','table']

# Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Route for Single Data Point Prediction
@app.route('/predict_price', methods=['POST', 'GET'])
def predict_data():
    if request.method == 'POST':
        try:
            if model is None or Scaler is None:
                return Response("Model or Scaler not loaded properly.", status=500)

            # Get form data
            data = [float(x) for x in request.form.values()]
            print("Received Input:", data)

            # Convert to DataFrame with correct column names
            input_df = pd.DataFrame([data], columns=feature_names)

            # Ensure input_df columns match the scaler's training feature order
            input_df = input_df[Scaler.feature_names_in_]  # This ensures correct feature names

            print("Input DataFrame:", input_df)

            # Transform input data
            final_input = Scaler.transform(input_df)  
            print("Transformed Data:", final_input)

            # Make prediction
            prediction = model.predict(final_input)
            print("Prediction:", prediction)

            # Determine result class and text
            result = prediction

            return render_template('price_prediction.html', result_text=result)

        except Exception as e:
            return Response(f"Error occurred: {str(e)}", status=500)
    else:
        return render_template('diamond_price.html')

if __name__ == "__main__":
    app.run(debug=True)
