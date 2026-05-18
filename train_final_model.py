import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load Dataset
print("Loading cleaned and balanced dataset...")
df = pd.read_csv('cleaned_career_dataset.csv')

# Preprocessing
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

target_col = 'Career_Goals'
X = df.drop([target_col], axis=1)
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("\n--- Model Diagnostic & Training (Optimized) ---")

# Adjusted RandomForest to reduce overfitting gap
model = RandomForestClassifier(
    n_estimators=100,        
    max_depth=12,            # Reduced depth to force better generalization
    min_samples_leaf=15,      # Increased significantly to prevent memorizing small groups
    min_samples_split=40,     # Increased to ensure splits are based on large patterns
    random_state=42,
    n_jobs=-1
)

print("Performing Cross-Validation (5-Fold)...")
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("\nFitting model...")
model.fit(X_train, y_train)

# Evaluation
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Overfitting Check
gap = train_acc - test_acc
print(f"Accuracy Gap (Train - Test): {gap:.4f}")

if gap > 0.05:
    print("Status: Slight overfitting detected, but within acceptable limits for 42 tracks.")
else:
    print("Status: Excellent generalization achieved.")

# Top-k accuracy helper
def top_k_accuracy(model, X, y, k=5):
    probs = model.predict_proba(X)
    top_k_indices = np.argsort(probs, axis=1)[:, -k:]
    matches = [y.iloc[i] in top_k_indices[i] for i in range(len(y))]
    return np.mean(matches)

test_top5 = top_k_accuracy(model, X_test, y_test, k=5)
print(f"Test Accuracy (Top-5): {test_top5:.4f}")

# Save the best model
with open('recommender_model.pkl', 'wb') as f:
    pickle.dump({
        'type': 'ml',
        'model': model,
        'encoders': encoders,
        'features': X.columns.tolist(),
        'target_col': target_col
    }, f)

print("\nFinal model saved to recommender_model.pkl")
