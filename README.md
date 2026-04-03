# 🫀 Heart Disease Prediction System

> A Machine Learning web application that predicts heart disease risk based on patient medical parameters using a Random Forest Classifier served through a Flask web interface.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [1. Data Generation](#1-data-generation)
  - [2. Data Preprocessing](#2-data-preprocessing)
  - [3. Model Training](#3-model-training)
  - [4. Saving Model Artifacts](#4-saving-model-artifacts)
  - [5. Flask Web Application](#5-flask-web-application)
- [Full System Flow](#full-system-flow)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Train the Model](#train-the-model)
  - [Run the Web App](#run-the-web-app)
- [Testing the System](#testing-the-system)
  - [Sample Test Cases](#sample-test-cases)
  - [Edge Cases](#edge-cases)
- [Why the Scaler Matters](#why-the-scaler-matters)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Disclaimer — Why This Matters](#disclaimer--why-this-matters)

---

## Project Overview

This system takes 8 patient health parameters as input and predicts whether a patient is at risk of heart disease. It combines a trained machine learning model with a simple web form so that predictions can be made without writing any code.

The project is built with **synthetic data** and is intended as an educational demonstration — not for real clinical use.

---

## Technologies Used

| Category        | Technology                              |
|-----------------|-----------------------------------------|
| Language        | Python 3.7+                             |
| Data Handling   | Pandas, NumPy                           |
| Machine Learning| Scikit-learn (RandomForest, StandardScaler) |
| Web Framework   | Flask                                   |
| Frontend        | HTML, CSS                               |
| Serialization   | Pickle                                  |

---

## Project Structure

```
heart-disease-predictor/
│
├── data/
│   └── heart_disease.csv       # Auto-generated synthetic dataset
│
├── templates/
│   └── index.html              # Web form for user input
│
├── static/
│   └── style.css               # Frontend styling
│
├── model.py                    # Data generation, training & saving the model
├── app.py                      # Flask web application
├── model.pkl                   # Saved trained model (generated after running model.py)
├── scaler.pkl                  # Saved StandardScaler (generated after running model.py)
└── requirements.txt            # Python dependencies
```

---

## How It Works

### 1. Data Generation

Since no real patient dataset is used, `model.py` generates **1000 synthetic patient records** using NumPy. Each record includes 8 features:

| Feature        | Range / Values  | Type        |
|----------------|-----------------|-------------|
| Age            | 20 – 80         | Continuous  |
| Gender         | 0 (Female) / 1 (Male) | Binary |
| Blood Pressure | 90 – 180 mmHg   | Continuous  |
| Cholesterol    | 150 – 300 mg/dL | Continuous  |
| Heart Rate     | 60 – 100 bpm    | Continuous  |
| Diabetes       | 0 (No) / 1 (Yes)| Binary      |
| Smoking        | 0 (No) / 1 (Yes)| Binary      |
| Chest Pain     | 0 – 3 (severity)| Categorical |

**Target Variable:** `Heart Disease` — `1` (has disease) or `0` (does not have disease).

The target is not purely random. It is generated using logic that mirrors real-world risk patterns:

```
Higher risk → Older Age + High Blood Pressure + High Cholesterol + Smoking = 1
Lower risk  → Young Age + Normal BP + Normal Cholesterol + No Smoking = 0
```

A small amount of randomness is added to prevent perfect determinism and simulate real-world noise. The generated data is saved to `data/heart_disease.csv`.

---

### 2. Data Preprocessing

Before training, the data is prepared in two steps:

**Train/Test Split**

The dataset is split into:
- **80% Training set** — used to teach the model
- **20% Test set** — held back to evaluate the model on unseen data

This prevents the model from memorizing answers and ensures it can generalize to new inputs.

**Feature Scaling (StandardScaler)**

Raw feature values have very different scales (e.g., Age = 60, Cholesterol = 250). Without normalization, features with larger numbers can dominate the model's learning. `StandardScaler` transforms all features to have:
- **Mean = 0**
- **Standard Deviation = 1**

This makes all features equally weighted during training.

---

### 3. Model Training

The project uses a **Random Forest Classifier** — an ensemble machine learning algorithm.

**How Random Forest Works:**

```
Input Features
      ↓
 100 Decision Trees (each trained on a random subset)
      ↓
 Each tree gives a vote: 0 or 1
      ↓
 Majority vote = Final Prediction
```

Each tree learns different patterns from the data. By combining all 100 trees, the model becomes more accurate and robust than any single decision tree. This technique is called **bagging** (Bootstrap Aggregating).

**Key configuration:**
- `n_estimators = 100` — 100 decision trees in the forest
- `random_state = 42` — ensures reproducible results

**Expected Accuracy:** ~85% on the test set

---

### 4. Saving Model Artifacts

After training, two objects are saved to disk using **Pickle**:

| File         | Contents                           | Purpose                              |
|--------------|------------------------------------|--------------------------------------|
| `model.pkl`  | Trained RandomForestClassifier     | Used by Flask to make predictions    |
| `scaler.pkl` | Fitted StandardScaler              | Used by Flask to scale user input    |

Saving these means the Flask app can load them instantly on each request without retraining — which would be too slow for a web app.

---

### 5. Flask Web Application

`app.py` connects the trained model to the user through a web browser.

**Request Lifecycle:**

```
1. User opens http://127.0.0.1:5000/
2. Flask serves the HTML input form
3. User fills in 8 health parameters and submits
4. Flask receives the POST request
5. Loads model.pkl and scaler.pkl
6. Scales the user input using the same scaler from training
7. Passes scaled input to model.predict()
8. Returns result to the page:
   → "Heart Disease Detected" or "No Heart Disease Detected"
```

**Two routes in Flask:**
- `GET /` — Renders the empty input form
- `POST /` — Processes the form, runs prediction, returns result

---

## Full System Flow

```
                  ┌─────────────────────────┐
                  │   NumPy Random Data     │
                  └────────────┬────────────┘
                               ↓
                  ┌─────────────────────────┐
                  │   heart_disease.csv     │
                  └────────────┬────────────┘
                               ↓
               ┌───────────────┴───────────────┐
               ↓                               ↓
  ┌────────────────────────┐      ┌────────────────────────┐
  │    StandardScaler      │      │  RandomForestClassifier│
  │    (fit on training)   │      │  (train on scaled data)│
  └──────────┬─────────────┘      └──────────┬─────────────┘
             ↓                               ↓
         scaler.pkl                       model.pkl
             └─────────────┬───────────────┘
                           ↓
                  ┌─────────────────┐
                  │   Flask App     │
                  └────────┬────────┘
                           ↓
                  ┌─────────────────┐
                  │  User Web Form  │
                  └────────┬────────┘
                           ↓
               Scale Input → model.predict()
                           ↓
             "Heart Disease Detected / Not Detected"
```

---

## Getting Started

### Prerequisites

- Python 3.7 or higher installed
- `pip` package manager

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
pip install -r requirements.txt
```

### Train the Model

This step generates the dataset, trains the model, and saves `model.pkl` and `scaler.pkl`:

```bash
python model.py
```

You should see output like:
```
Generating synthetic dataset...
Training RandomForestClassifier...
Model Accuracy: 85.5%
Model saved to model.pkl
Scaler saved to scaler.pkl
```

### Run the Web App

#### Option A (Fastest, single command)

From the project directory (`C:\Users\Ravikiran\OneDrive\Desktop\bhavana`):

```bash
python app.py
```

#### Option B (One command with dependencies install, in case not installed yet)

```bash
pip install flask numpy scikit-learn pandas
python app.py
```

#### Option C (Fully recommended local environment)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
pip install -r requirements.txt
python app.py
```

#### Open in browser

- Local: `http://127.0.0.1:5000/`
- Network (any device on same LAN): `http://<your-PC-IP>:5000/` (e.g. `http://192.168.1.106:5000/`)

> If you are on Windows and get a firewall prompt, allow access for Python.

Fill in the patient details and click **Predict** to see the result.

---

## Deployment

This Flask application can be hosted entirely on a single platform (no frontend/backend split required). Here are the best options for complete, one-click deployment:

### Option 1: Heroku (Recommended - Easiest)

**Why Heroku?** Hosts your entire Flask app + ML models in one place with free tier available.

#### Step-by-Step Heroku Deployment

1. **Install Heroku CLI**
   Download from: https://devcenter.heroku.com/articles/heroku-cli

2. **Login to Heroku**
   ```bash
   heroku login
   ```
   This opens your browser for authentication.

3. **Create Heroku App**
   ```bash
   heroku create your-unique-app-name
   ```
   (Choose a unique name, e.g., `heart-disease-predictor-123`)

4. **Deploy Your App**
   ```bash
   git push heroku main
   ```

5. **Open Your Live App**
   ```bash
   heroku open
   ```

**Your app will be live at:** `https://your-app-name.herokuapp.com/`

**Free Tier Details:**
- 550-1000 hours/month free
- App sleeps after 30 minutes inactivity
- Wakes automatically on next visit

### Option 2: Render (Also Complete Hosting)

**Why Render?** Free tier with GitHub integration, hosts entire app.

1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Choose "Web Service" → Python
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn app:app`
6. Click Deploy!

**Free Tier:** 750 hours/month

### What Gets Hosted
- ✅ Flask backend with ML predictions
- ✅ HTML templates and CSS styling
- ✅ Trained models (model.pkl, scaler.pkl)
- ✅ All dependencies and configurations

### Other Hosting Options
- **Railway**: Simple Python hosting, GitHub integration
- **DigitalOcean App Platform**: Affordable ($5/month), good for Flask
- **AWS Elastic Beanstalk**: Scalable but more complex setup

**Note:** Netlify is not suitable as it doesn't support Python/Flask applications.

---

## Testing the System

### Sample Test Cases

**High Risk Patient**

| Parameter      | Value        |
|----------------|--------------|
| Age            | 60           |
| Gender         | Male (1)     |
| Blood Pressure | 150 mmHg     |
| Cholesterol    | 250 mg/dL    |
| Heart Rate     | 80 bpm       |
| Diabetes       | Yes (1)      |
| Smoking        | Yes (1)      |
| Chest Pain     | 2            |

→ **Expected Result:** Heart Disease Detected ✅

---

**Low Risk Patient**

| Parameter      | Value         |
|----------------|---------------|
| Age            | 30            |
| Gender         | Female (0)    |
| Blood Pressure | 120 mmHg      |
| Cholesterol    | 180 mg/dL     |
| Heart Rate     | 70 bpm        |
| Diabetes       | No (0)        |
| Smoking        | No (0)        |
| Chest Pain     | 0             |

→ **Expected Result:** No Heart Disease Detected ✅

### Edge Cases

- Test with minimum values (Age=20, BP=90, Cholesterol=150)
- Test with maximum values (Age=80, BP=180, Cholesterol=300)
- Leave required fields empty to verify form validation
- Enter non-numeric values to test input type enforcement

---

## Why the Scaler Matters

This is one of the most critical concepts in the project.

During training, StandardScaler transforms the data (e.g., Age=60 → scaled value ~0.87). The model learns from **scaled values**, not raw numbers.

When a user submits the web form, the raw input **must be transformed using the exact same scaler** before being passed to the model — otherwise the model receives numbers in the wrong range and produces incorrect predictions.

This is why `scaler.pkl` is saved separately and reloaded in `app.py` on every prediction request.

```python
# Correct prediction flow in app.py
scaler = pickle.load(open('scaler.pkl', 'rb'))
model  = pickle.load(open('model.pkl', 'rb'))

scaled_input = scaler.transform([user_input])   # ← Critical step
prediction   = model.predict(scaled_input)
```

---

## Limitations

- Uses **synthetic data** — results do not reflect real medical accuracy
- Small dataset of only 1000 samples
- No cross-validation or hyperparameter tuning
- Not suitable for any clinical or diagnostic use

---

## Future Enhancements

- Integrate real-world datasets (e.g., UCI Heart Disease Dataset)
- Add model explainability (SHAP values to show which features matter most)
- Add cross-validation and hyperparameter tuning
- Deploy to cloud (AWS, Heroku, or Render)
- Add user authentication and prediction history
- Build a REST API for programmatic access

---

## Disclaimer — Why This Matters

> ⚠️ **This project is for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.**

This disclaimer isn't just boilerplate — there's a real reason it's here, and it's worth understanding.

The system outputs very clinical-sounding results: **"Heart Disease Detected"** or **"No Heart Disease Detected"**. That language is direct and conclusive. The problem is — the model was trained entirely on **synthetic, randomly generated data**. It has no real medical validity whatsoever.

Here's why that matters in practice:

- Someone could run this app, see **"No Heart Disease Detected"**, and feel genuinely reassured about their health — even though the model has no real predictive power over their actual condition.
- On the flip side, **"Heart Disease Detected"** could cause unnecessary fear or panic in someone who is perfectly healthy.

Neither of those reactions is appropriate, and the output language alone is enough to cause them — especially for users who don't read into the technical details of how the model was built.

This is standard practice for any machine learning project that touches **health, medicine, finance, or law** — even hobby and student projects — because the framing of the output can easily be misread by someone outside the technical audience.

**In short:** the model is a demonstration of how a prediction pipeline works. It should never be used to make real health decisions. If you or anyone else has concerns about heart health, please consult a licensed medical professional.