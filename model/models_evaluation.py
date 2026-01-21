import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load scaler
scaler = joblib.load("model/scaler.pkl")
X_test_scaled = scaler.transform(X_test)

models = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]

print("\nModel Evaluation Results:\n")

for model_name in models:
    model = joblib.load(f"model/{model_name}.pkl")

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print(f"🔹 {model_name}")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
    print(f"AUC       : {roc_auc_score(y_test, y_prob):.3f}")
    print(f"Precision : {precision_score(y_test, y_pred):.3f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.3f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred):.3f}")
    print(f"MCC       : {matthews_corrcoef(y_test, y_pred):.3f}")
    print("-" * 40)
