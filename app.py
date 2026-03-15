import streamlit as st
import pandas as pd
import joblib
from groq import Groq
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)

# Initialize Groq client
client = Groq(
    api_key="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"
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

# ---------------------------
# Download Sample CSV
# ---------------------------

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

# ---------------------------
# Download Test CSV
# ---------------------------

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

# ---------------------------
# Model Selection
# ---------------------------

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

# ---------------------------
# Upload Dataset
# ---------------------------

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

    # Load scaler
    scaler = joblib.load("model/scaler.pkl")
    X_scaled = scaler.transform(X)

    # Load model
    model = joblib.load(f"model/{model_selector}.pkl")

    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # ---------------------------
    # Metrics
    # ---------------------------

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Model Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.3f}")
    col2.metric("Precision", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{f1:.3f}")
    col5.metric("AUC Score", f"{auc:.3f}")
    col6.metric("MCC", f"{mcc:.3f}")

    # ---------------------------
    # Confusion Matrix
    # ---------------------------

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))

   # -----------------------------
    # All Models Comparison
    # -----------------------------

    st.divider()
    st.subheader("📊 All Models Performance Comparison")

    models = {
        "Logistic Regression": joblib.load("model/Logistic Regression.pkl"),
        "Decision Tree": joblib.load("model/Decision Tree.pkl"),
        "KNN": joblib.load("model/KNN.pkl"),
        "Naive Bayes": joblib.load("model/Naive Bayes.pkl"),
        "Random Forest": joblib.load("model/Random Forest.pkl"),
        "XGBoost": joblib.load("model/XGBoost.pkl"),
    }

    results = []

    for name, m in models.items():

        y_pred_all = m.predict(X_scaled)
        y_prob_all = m.predict_proba(X_scaled)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y, y_pred_all),
            "Precision": precision_score(y, y_pred_all),
            "Recall": recall_score(y, y_pred_all),
            "F1 Score": f1_score(y, y_pred_all),
            "AUC Score": roc_auc_score(y, y_prob_all),
            "MCC": matthews_corrcoef(y, y_pred_all)
        })

    results_df = pd.DataFrame(results)

    st.dataframe(results_df)

    results_context = results_df.to_string(index=False)

    # -----------------------------
    # AI Chatbot
    # -----------------------------

    st.divider()
    st.subheader("🤖 AI Model Performance Analyst")

    st.markdown(
        "Ask questions about model performance, comparisons, "
        "or interpretation of evaluation metrics."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_question = st.chat_input("Ask a question about the model results")

    if user_question:

        prompt = f"""
You are a machine learning expert.

Use the model evaluation results below to answer questions.

Explain:
- which model performs best
- why performance differs
- advantages and disadvantages of algorithms
- interpretation of metrics

Model Results:
{results_context}

User Question:
{user_question}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content

        st.session_state.messages.append({"role": "user", "content": user_question})
        st.session_state.messages.append({"role": "assistant", "content": answer})

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

else:
    st.info("Please upload a CSV file or download the sample file to begin.")