from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import os
import sys
import types
import logging

# Ensure legacy numpy module path is available for old pickled models (numpy._core)
if 'numpy._core' not in sys.modules:
    ncore = types.ModuleType('numpy._core')
    ncore.__path__ = []
    sys.modules['numpy._core'] = ncore

# Map known submodules from numpy._core.* to numpy.core.* for compatibility
legacy_submodules = ['multiarray', 'umath', 'numeric', 'npyio', 'fromnumeric', 'numpyio']
for sub in legacy_submodules:
    legacy_name = f'numpy._core.{sub}'
    target_name = f'numpy.core.{sub}'
    if legacy_name not in sys.modules:
        try:
            sys.modules[legacy_name] = __import__(target_name, fromlist=['*'])
        except ModuleNotFoundError:
            pass

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------- Load model and scaler at module level ----------
# @app.before_first_request was removed in Flask 2.3+, so we load at import time.
model = None
scaler = None

def _load_model_and_scaler():
    """Load model and scaler from disk. Called once at import time."""
    global model, scaler
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, 'model.pkl')
    scaler_path = os.path.join(base, 'scaler.pkl')
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        app.logger.info('Loaded model and scaler from %s and %s', model_path, scaler_path)
    except FileNotFoundError as e:
        app.logger.error('Model or scaler file not found: %s', e)
    except Exception as e:
        app.logger.error('Failed loading model or scaler: %s', e)

_load_model_and_scaler()

# ---------- Routes ----------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure model and scaler are loaded
    if model is None or scaler is None:
        app.logger.error('Model or scaler not loaded when handling /predict')
        return render_template('index.html', prediction='Error: model not available. Check server logs.')

    try:
        # Extract form data with basic validation
        age = int(request.form.get('age', 0))
        gender = int(request.form.get('gender', 0))
        blood_pressure = int(request.form.get('blood_pressure', 0))
        cholesterol = int(request.form.get('cholesterol', 0))
        heart_rate = int(request.form.get('heart_rate', 0))
        diabetes = int(request.form.get('diabetes', 0))
        smoking = int(request.form.get('smoking', 0))
        chest_pain = int(request.form.get('chest_pain', 0))

        # Prepare input for prediction
        input_data = np.array([[age, gender, blood_pressure, cholesterol, heart_rate, diabetes, smoking, chest_pain]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # Result
        result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'

        return render_template('index.html', prediction=result)

    except ValueError as e:
        app.logger.exception('Invalid input values: %s', e)
        return render_template('index.html', prediction='Error: invalid input values.')
    except Exception as e:
        app.logger.exception('Prediction failed: %s', e)
        return render_template('index.html', prediction='Error during prediction. Check server logs.')

@app.route('/health')
def health():
    """Health check endpoint: returns 200 if model and scaler are loaded, 503 otherwise."""
    if model is not None and scaler is not None:
        return jsonify({'status': 'ok'}), 200
    else:
        return jsonify({'status': 'unhealthy', 'reason': 'model or scaler not loaded'}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
