import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

age = np.random.randint(20, 80, n_samples)
gender = np.random.choice([0, 1], n_samples)  # 0: Female, 1: Male
blood_pressure = np.random.randint(90, 180, n_samples)
cholesterol = np.random.randint(150, 300, n_samples)
heart_rate = np.random.randint(60, 100, n_samples)
diabetes = np.random.choice([0, 1], n_samples)  # 0: No, 1: Yes
smoking = np.random.choice([0, 1], n_samples)  # 0: No, 1: Yes
chest_pain = np.random.choice([0, 1, 2, 3], n_samples)  # 0-3: Types of chest pain

# Simple logic for heart disease (target)
heart_disease = ((age > 50) & (blood_pressure > 140) & (cholesterol > 200) & (smoking == 1)) | np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
heart_disease = heart_disease.astype(int)

df = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'BloodPressure': blood_pressure,
    'Cholesterol': cholesterol,
    'HeartRate': heart_rate,
    'Diabetes': diabetes,
    'Smoking': smoking,
    'ChestPain': chest_pain,
    'HeartDisease': heart_disease
})

df.to_csv('data/heart_disease.csv', index=False)

# Data Preprocessing and Feature Selection
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Save the model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print('Model and scaler saved successfully.')