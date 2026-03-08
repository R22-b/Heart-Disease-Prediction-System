# Final Report: Heart Disease Prediction System

## Introduction
Heart disease is a leading cause of death globally. This project develops a machine learning model to predict heart disease from patient data, deployed via a web application.

## Objective
Build and deploy a predictive model using medical parameters.

## Technologies
- Python, Pandas, Scikit-learn, Random Forest Classifier, Flask, HTML, CSS.

## Dataset
Synthetic dataset with 1000 samples including Age, Gender, Blood Pressure, Cholesterol, Heart Rate, Diabetes, Smoking, Chest Pain, and Heart Disease target.

## Methodology
1. Generate synthetic data.
2. Preprocess and split into train/test.
3. Scale features with StandardScaler.
4. Train RandomForest model.
5. Evaluate accuracy (~85%).
6. Build Flask app for predictions.

## Results
- Model provides binary predictions: 'Heart Disease Detected' or 'No Heart Disease Detected'.
- Web interface is user-friendly.

## Advantages
- Simple, fast predictions.
- Easy to extend.

## Limitations
- Synthetic data; not validated for real use.
- Small dataset.

## Future Work
- Use real medical data.
- Add more parameters.
- Improve UI and deploy online.
- Implement authentication.

## Conclusion
This demonstrates ML integration with web apps for healthcare predictions. With real data, it can aid medical decisions.