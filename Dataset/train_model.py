# =============================================
# AI Farming Assistant - Crop Prediction Model
# =============================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

print("🌾 Crop Recommendation Model Training Started...")
print("=" * 50)

# ---- Step 1: Data Load Karo ----
df = pd.read_csv('Crop_recommendation.csv')
print(f"✅ Dataset loaded! Total rows: {len(df)}")
print(f"📋 Columns: {list(df.columns)}")
print(f"🌱 Crops in dataset: {df['label'].unique()}")
print()

# ---- Step 2: Data Check Karo ----
print("📊 Dataset Info:")
print(df.head())
print()

# ---- Step 3: Features aur Target Alag Karo ----
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print()

# ---- Step 4: Train aur Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% testing ke liye
    random_state=42
)

print(f"📚 Training data: {len(X_train)} rows")
print(f"🧪 Testing data: {len(X_test)} rows")
print()

# ---- Step 5: Model Banao aur Train Karo ----
print("🤖 Training Random Forest Model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")
print()

# ---- Step 6: Accuracy Check Karo ----
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")
print()

# ---- Step 7: Model Save Karo ----
os.makedirs('models', exist_ok=True)
import os
os.makedirs('models', exist_ok=True)
with open('models/crop_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("💾 Model saved at: models/crop_model.pkl")
print()

# ---- Step 8: Test Prediction Karo ----
print("🧪 Testing with sample data...")
print("-" * 40)

sample_input = pd.DataFrame([[90, 42, 43, 20.8, 82.0, 6.5, 202.9]],
    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
)

predicted_crop = model.predict(sample_input)[0]
print(f"Sample Input: N=90, P=42, K=43, Temp=20.8°C, Humidity=82%, pH=6.5")
print(f"🌾 Recommended Crop: {predicted_crop.upper()}")
print()
print("=" * 50)
print("✅ ALL DONE! Model is ready to use!")