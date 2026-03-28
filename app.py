from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    blood_pressure = int(request.form['blood_pressure'])
    cholesterol = int(request.form['cholesterol'])
    heart_rate = int(request.form['heart_rate'])
    diabetes = int(request.form['diabetes'])
    smoking = int(request.form['smoking'])
    chest_pain = int(request.form['chest_pain'])

    # Prepare input for prediction
    input_data = np.array([[age, gender, blood_pressure, cholesterol, heart_rate, diabetes, smoking, chest_pain]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Result
    result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))