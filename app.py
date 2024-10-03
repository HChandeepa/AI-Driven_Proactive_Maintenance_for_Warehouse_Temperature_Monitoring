from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the pre-trained models
rf_model = joblib.load('random_forest_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')
svm_model = joblib.load('svm_model.pkl')

app = Flask(__name__)

def predict_failure(model, temperature, humidity, hvac_status):
    data = np.array([[temperature, humidity, hvac_status]])
    return model.predict(data)[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    temperature = data['Temperature']
    humidity = data['Humidity']
    hvac_status = data['HVAC_Status']

    rf_prediction = predict_failure(rf_model, temperature, humidity, hvac_status)
    lr_prediction = predict_failure(lr_model, temperature, humidity, hvac_status)
    svm_prediction = predict_failure(svm_model, temperature, humidity, hvac_status)

    return jsonify({
        'Random Forest': rf_prediction,
        'Logistic Regression': lr_prediction,
        'SVM': svm_prediction
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
