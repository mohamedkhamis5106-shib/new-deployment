import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🔮 Diabetes Prediction App")

# محاولة تحميل الموديل والـ scaler
try:
    model = joblib.load("log_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("✅ Model and Scaler loaded successfully!")
except:
    st.error("⚠️ Model or Scaler not found. Please train and save them first.")

st.subheader("Enter Patient Data")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 30)
        bmi = st.number_input("BMI", 0.0, 50.0, 25.0)
    with col2:
        hba1c = st.number_input("HbA1c", 0.0, 20.0, 5.0)
        chol = st.number_input("Cholesterol", 0.0, 500.0, 200.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        sample = pd.DataFrame([[age, bmi, hba1c, chol]], columns=["AGE", "BMI", "HbA1c", "Chol"])
        sample = scaler.transform(sample)
        pred = model.predict(sample)[0]

        if pred == 1:
            st.error("⚠️ The patient is Diabetic")
        else:
            st.success("✅ The patient is Not Diabetic")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.subheader("Batch Prediction from CSV")

csv_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])
if csv_file:
    try:
        data = pd.read_csv(csv_file)
        st.write("📄 Uploaded Data Preview:", data.head())

        data_scaled = scaler.transform(data)
        preds = model.predict(data_scaled)

        data["Prediction"] = ["Diabetic" if p == 1 else "Not Diabetic" for p in preds]
        st.success("✅ Predictions done successfully!")
        st.dataframe(data)

        csv_out = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv_out, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Batch prediction error: {e}")
