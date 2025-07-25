import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Stroke Risk Predictor",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper Function for Model Loading ---
@st.cache_resource
def load_model_assets():
    """
    Loads the machine learning model, SHAP explainer, and training columns.
    Uses Streamlit's caching to load these only once.
    """
    try:
        # These files must be in the same directory as your Streamlit script
        model = joblib.load('stroke_model.pkl')
        explainer = joblib.load('shap_explainer.pkl')
        # This file is crucial and should be saved during model training
        # It contains the exact column order the model was trained on.
        # You can create it with: joblib.dump(X_train.columns, 'training_columns.pkl')
        train_cols = joblib.load('training_columns.pkl')
        return model, explainer, train_cols
    except FileNotFoundError as e:
        st.error(
            f"🚨 **Required file not found: `{e.filename}`.** Please make sure `stroke_model.pkl`, "
            "`shap_explainer.pkl`, and `training_columns.pkl` are in the same directory as this script."
        )
        # Stop the app execution if files are missing
        return None, None, None

# --- Load Assets ---
model, explainer, train_cols = load_model_assets()

# --- Sidebar for User Input ---
st.sidebar.header("👤 Patient Information")
st.sidebar.info("Please provide the patient's details below to get a stroke risk prediction.")

def get_user_input():
    """
    Collects user input from the sidebar and returns it as a pandas DataFrame.
    """
    gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.sidebar.slider("Age", 1, 100, 60)
    hypertension = st.sidebar.selectbox("History of Hypertension", ["No", "Yes"])
    heart_disease = st.sidebar.selectbox("History of Heart Disease", ["No", "Yes"])
    ever_married = st.sidebar.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.sidebar.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.sidebar.number_input("Average Glucose Level (mg/dL)", 50.0, 300.0, 105.0)
    bmi = st.sidebar.number_input("Body Mass Index (BMI)", 10.0, 70.0, 28.0)
    smoking_status = st.sidebar.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    # Create a dictionary from the user's input
    data = {
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }
    
    # Convert the dictionary to a pandas DataFrame
    input_df = pd.DataFrame([data])
    return input_df

# Get the user input and store it
raw_input_df = get_user_input()

def preprocess_input(df, training_columns):
    """
    Preprocesses the raw user input DataFrame to match the model's training data format.
    This is the most critical step for the app to work correctly.
    """
    # Apply one-hot encoding to categorical features
    processed_df = pd.get_dummies(df)
    
    # Reindex columns to match the training columns.
    # - This ensures all columns the model expects are present.
    # - `fill_value=0` sets any missing columns (from the one-hot encoding) to 0.
    # - It also ensures the column order is identical to the training data.
    processed_df = processed_df.reindex(columns=training_columns, fill_value=0)
    
    return processed_df

# --- Main Page Content ---
st.title("🧠 Stroke Risk Predictor & Explainability")
st.write(
    "This application uses a machine learning model to predict the likelihood of a patient having a stroke. "
    "It also uses SHAP (SHapley Additive exPlanations) to explain the prediction."
)

st.subheader("Your Input:")
st.dataframe(raw_input_df)

# Only proceed if the model and other assets were loaded successfully
if model and explainer and train_cols is not None:
    # Prediction and Explanation button
    if st.button("Analyze Stroke Risk", type="primary"):
        
        # Preprocess the raw input to be ready for the model
        processed_input_df = preprocess_input(raw_input_df, train_cols)

        # --- Prediction ---
        prediction = model.predict(processed_input_df)[0]
        proba = model.predict_proba(processed_input_df)[0][1]

        st.header("📈 Prediction Result")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.metric("Prediction", "High Risk", "Risk of Stroke Detected", delta_color="inverse")
            else:
                st.metric("Prediction", "Low Risk", "No Significant Stroke Risk Detected", delta_color="off")
                
        with col2:
            st.metric("Probability of Stroke", f"{proba:.2%}")

        # --- SHAP Explanation ---
        st.header("🔍 Explaining the Prediction")
        st.info(
            "The following plot explains how the model arrived at its prediction. It shows which factors "
            "increased or decreased the predicted risk of stroke."
        )
        
        # --- Waterfall Plot ---
        st.subheader("Prediction Breakdown Waterfall Plot")
        st.write(
            "This plot provides a detailed breakdown, starting from the base prediction value (E[f(x)]) and showing "
            "how each feature's contribution cumulatively builds up to the final prediction score (f(x))."
        )

        # Use the modern __call__ interface of the explainer. This is more robust.
        # It returns a rich Explanation object that can be sliced.
        shap_explanations = explainer(processed_input_df)

        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))

        # The shap_explanations object has dimensions (n_samples, n_features, n_classes)
        # We want the explanation for the first sample [0], for all features [:], for the positive class [1]
        # This provides the 1D explanation that the waterfall plot requires.
        shap.plots.waterfall(shap_explanations[0, :, 1], show=False)
        
        plt.tight_layout()
        st.pyplot(fig_waterfall)

# Add a footer
st.markdown("---")
st.write("Disclaimer: This is an educational tool and not a substitute for professional medical advice.")