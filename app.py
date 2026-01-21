import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)


st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Predictior App")

st.markdown(
    """
    This application predicts the **presence of heart disease** using multiple  
    **machine learning classification models** trained on clinical patient data.

    ### Key Features
    - Upload test data in CSV format  
    - Select a machine learning model  
    - View performance metrics instantly  
    - Download a sample CSV file for testing  

    Note: Please upload only test data .
    """
)

st.divider()

@st.cache_data
def load_sample_data():
    return pd.read_csv("data/sampleData.csv")

sample_df = load_sample_data()

st.subheader("Download Sample CSV File and upload your data in the given format as per sample")
st.download_button(
    label="📥 Download Sample CSV",
    data=sample_df.to_csv(index=False),
    file_name="sample_heart_disease_test_data.csv",
    mime="text/csv"
)

@st.cache_data
def load_test_data():
    return pd.read_csv("data/Heart_Test_Data.csv")

sample_df = load_test_data()
st.subheader("Download Sample Test CSV File")
st.download_button(
    label="📥 Download Test CSV",
    data=sample_df.to_csv(index=False),
    file_name="sample_test_file.csv",
    mime="text/csv"
)

st.divider()

model_selector = st.selectbox(
    "🧠 Select Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

uploaded_file = st.file_uploader(
    "📂 Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
        st.stop()

    X = data.drop("target", axis=1)
    y = data["target"]

    # Load scaler & model
    scaler = joblib.load("model/scaler.pkl")
    X_scaled = scaler.transform(X)

    model = joblib.load(f"model/{model_selector}.pkl")

    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    st.subheader("Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y, y_pred):.3f}")
    col3.metric("Recall", f"{recall_score(y, y_pred):.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1_score(y, y_pred):.3f}")
    col5.metric("AUC Score", f"{roc_auc_score(y, y_prob):.3f}")
    col6.metric("MCC", f"{matthews_corrcoef(y, y_pred):.3f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))

else:
    st.info("PLease Upload a CSV file or download the sample file to begin.")
