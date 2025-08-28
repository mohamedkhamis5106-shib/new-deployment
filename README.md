
# Diabetes Prediction App (Streamlit)

A simple Streamlit app that performs EDA and trains a Logistic Regression model on a diabetes dataset.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

If you keep your CSV next to `app.py` with the exact name:
```
Dataset of Diabetes .csv
```
the app will auto-load it. Otherwise, upload it from the sidebar.

## Deploy to Streamlit Community Cloud

1. Push this folder to a **public GitHub repo**.
2. Go to https://streamlit.io/cloud → **Deploy an app**.
3. Choose your repo/branch and set main file path to `app.py`.
4. Click **Deploy**.

