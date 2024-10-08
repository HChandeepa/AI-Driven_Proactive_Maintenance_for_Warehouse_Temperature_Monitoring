# AI-Driven Proactive Maintenance for Warehouse Temperature Monitoring

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Model Development](#model-development)
- [API Development](#api-development)
- [Deployment](#deployment)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)
- [License](#license)

## Introduction

The **AI-Driven Proactive Maintenance for Warehouse Temperature Monitoring** project aims to leverage machine learning techniques to predict potential equipment failures based on environmental data. By analyzing parameters such as temperature, humidity, and HVAC status, this project provides actionable insights that facilitate proactive maintenance strategies, thereby reducing downtime and operational costs.

---

## Project Overview

This project is structured around the following components:

1. **Data Collection**: Used a sample dataset.
2. **Model Development**: Building machine learning models to predict failure risks based on the collected data.
3. **API Development**: Creating a Flask API to expose the predictive models for external use.
4. **Deployment**: Hosting the API on PythonAnywhere for accessibility.

---

## Model Development

### Data Preprocessing

The initial step involved cleaning and preprocessing the dataset. Key tasks included:
- Handling missing values.
- Normalizing features for consistent scaling.
- Encoding categorical variables.

### Model Selection

Several machine learning models were evaluated, including:
- **Random Forest**
- **Logistic Regression**
- **Support Vector Machine (SVM)**

The models were trained and validated using a separate testing dataset. Performance metrics such as accuracy, ROC AUC, precision, recall, and F1-score were computed to select the best-performing model.

### Model Training

The Random Forest model was ultimately chosen based on its superior performance.

The trained model was saved using the `pickle` library for later use in the API.
```py
import pickle

# Save the model to a file
with open('random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(random_forest_model, model_file)
```
## API Development
A Flask application was developed to create an API for making predictions. The key features of the API include:

Home Route: Displays a welcome message.
Predict Route: Accepts POST requests to receive JSON data and returns the prediction of failure risk.
API Code
Here is a simplified version of the API code:
```py
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
```
## Deployment
The Flask API was deployed on PythonAnywhere. The following steps were followed:

1. Created a free account on PythonAnywhere.
2. Set up a new web app using Flask and the specified Python version.
3. Uploaded the app.py and the model file random_forest_model.pkl.
4. Configured the app and ensured it was running correctly.
5. The API can be accessed at: https://hchandeepa.pythonanywhere.com

## Testing the API
The API can be tested using tools like Postman or CURL. For example, to make a prediction, send a POST request to the /predict endpoint with the required JSON data.

Example JSON input:
```json
{
    "Temperature": 72,
    "Humidity": 50,
    "HVAC_Status": 1
}
```
## Usage
To use the API, simply send a POST request to the /predict endpoint with the appropriate JSON data containing Temperature, Humidity, and HVAC_Status. The API will return the predicted failure risk.

Example Usage in Python:

```py
import requests

data = {
    "Temperature": 72,
    "Humidity": 50,
    "HVAC_Status": 1
}

response = requests.post('https://hchandeepa.pythonanywhere.com/predict', json=data)
print(response.json())

```
## Technologies Used
- **Python**
- **Flask**
- **Scikit-learn**
- **Pandas**
- **Pickle**
- **PythonAnywhere for deployment**

## License
This project is licensed under the MIT License see the [LICENSE](LICENSE)file for details.
