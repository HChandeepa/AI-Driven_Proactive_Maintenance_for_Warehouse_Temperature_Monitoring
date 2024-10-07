from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load the Random Forest model from file
with open('random_forest_model.pkl', 'rb') as rf_file:
    random_forest_model = pickle.load(rf_file)

# Initialize Flask app
app = Flask(__name__)

# Define the root route
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the AI-Driven Proactive Maintenance API"

# Define the API route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    
    # Convert data to a DataFrame
    input_data = pd.DataFrame([data])

    # Extract features: 'Temperature', 'Humidity', 'HVAC_Status'
    features = ['Temperature', 'Humidity', 'HVAC_Status']
    input_features = input_data[features]
    
    # Make prediction
    prediction = random_forest_model.predict(input_features)[0]  # 0 or 1
    
    # Create response
    result = {
        'failure_risk': int(prediction)  # Convert NumPy integer to Python int
    }
    
    # Return response as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
