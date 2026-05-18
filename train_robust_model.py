import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load Dataset 2
print("Loading dataset...")
df = pd.read_csv('dataset9000.csv')

# Preprocessing
encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop(['Role'], axis=1)
y = df['Role']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Robust Model to prevent overfitting:
# - max_depth: limit how deep trees can go
# - min_samples_leaf: ensure each leaf has enough samples
# - n_estimators: enough trees for stability
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10, 
    min_samples_leaf=5,
    random_state=42
)

print("Performing Cross-Validation...")
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

print("Training final model...")
model.fit(X_train, y_train)

# Check for overfitting
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

if train_acc - test_acc > 0.05:
    print("Warning: Model might be overfitting!")
else:
    print("Model generalization looks healthy.")

# Save the best model
with open('recommender_model.pkl', 'wb') as f:
    pickle.dump({
        'type': 'ml',
        'model': model,
        'encoders': encoders,
        'features': X.columns.tolist(),
        'target_col': 'Role'
    }, f)

print("\nRobust model saved to recommender_model.pkl")

# Plot feature importance to see if it relies on too few features
plt.figure(figsize=(12, 8))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title('Robust Model Feature Importances')
plt.tight_layout()
plt.savefig('robust_model_importance.png')
print("Feature importance plot saved as robust_model_importance.png")
