import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction App (Simple Version)")

# ========= Upload dataset =========
file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Dataset Preview")
    
    # ========= EDA =========
    st.subheader("Exploratory Data Analysis")
    col = st.selectbox("Choose a numeric column for histogram", df.select_dtypes(include=["int64","float64"]).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

    if st.checkbox("Show correlation heatmap"):
        fig2, ax2 = plt.subplots()
        numeric_df = df.select_dtypes(include=["int64","float64"])
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

    if st.checkbox("Show pairplot"):
        st.info("This may take some time if dataset is large.")
        fig3 = sns.pairplot(df, hue="CLASS", diag_kind="kde")
        st.pyplot(fig3)


    # ========= Data Preprocessing =========
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"M":1,"F":0})

    if "CLASS" in df.columns:
        
        df["CLASS"] = df["CLASS"].astype(str).str.strip()

        if df["CLASS"].dtype == "object":
            le = LabelEncoder()
            df["CLASS"] = le.fit_transform(df["CLASS"])


        df = df.dropna()
    # ========= Training =========
        if st.button("Train Models"):
            X = df.drop(["ID", "No_Pation", "CLASS"], axis=1, errors="ignore")
            y = df["CLASS"]
        
            X = X.select_dtypes(include=["int64", "float64"])
            X = X.fillna(X.mean())
        
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
            log_model = LogisticRegression(max_iter=1000)
            log_model.fit(X_train, y_train)
            y_pred_log = log_model.predict(X_test)
            st.subheader("Logistic Regression Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_log))
            st.text("Classification Report:\n" + classification_report(y_test, y_pred_log))
        
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
        
            tr = DecisionTreeClassifier(class_weight="balanced")
            tr.fit(X_train, y_train)
            y_pred_tr = tr.predict(X_test)
            st.subheader("Decision Tree Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_tr))
            st.text("Classification Report:\n" + classification_report(y_test, y_pred_tr))
        
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred_tr), annot=True, fmt="d", cmap="Greens", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)
        
            model = svm.SVC(decision_function_shape="ovr")
            model.fit(X_train, y_train)
            y_pred_svm = model.predict(X_test)
            st.subheader("SVM Results")
            st.write("Accuracy:", accuracy_score(y_test, y_pred_svm))
            st.text("Classification Report:\n" + classification_report(y_test, y_pred_svm))
        
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Oranges", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)


        # SVM
        model = svm.SVC(decision_function_shape="ovr")
        model.fit(X_train, y_train)
        y_pred_svm = model.predict(X_test)
        st.subheader("SVM Results")
        st.write("Accuracy:", accuracy_score(y_test, y_pred_svm))
        st.text("Classification Report:\n" + classification_report(y_test, y_pred_svm))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Oranges", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

        # ========= Prediction =========
        st.subheader("Make a Prediction")
        age = st.number_input("Age", 0, 120, 30)
        bmi = st.number_input("BMI", 0.0, 50.0, 25.0)
        hba1c = st.number_input("HbA1c", 0.0, 20.0, 5.0)
        chol = st.number_input("Cholesterol", 0.0, 500.0, 200.0)

        if st.button("Predict"):
            sample = pd.DataFrame([[age, bmi, hba1c, chol]], columns=["AGE","BMI","HbA1c","Chol"])
            sample = scaler.transform(sample)
            pred = model.predict(sample)[0]
            st.success("Diabetic" if pred==1 else "Not Diabetic")
