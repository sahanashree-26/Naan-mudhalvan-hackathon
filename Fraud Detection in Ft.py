import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Smart Freeze: AI Fraud Detection App")

uploaded_file = st.file_uploader("Upload Transaction CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_columns = ['TransactionAmount', 'LoginAttempts', 'AccountBalance', 'CustomerAge']
    if not all(col in df.columns for col in required_columns):
        st.error("Required columns not found in the uploaded file.")
    else:
        df = df.dropna()

        # Label creation: simple rule-based fraud (for demo)
        df['is_fraud'] = ((df['TransactionAmount'] > 1000) | (df['LoginAttempts'] > 3)).astype(int)

        # Feature selection
        X = df[['TransactionAmount', 'LoginAttempts', 'AccountBalance', 'CustomerAge']]
        y = df['is_fraud']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader(f"Model Accuracy: {accuracy*100:.2f}%")

        # Predict on full dataset
        df['Prediction'] = model.predict(X)

        def freeze_alert(row):
            if row['Prediction'] == 1:
                return 'ALERT: Fraudulent transaction detected. Access frozen.'
            else:
                return 'Safe Transaction'

        df['SmartFreeze'] = df.apply(freeze_alert, axis=1)

        st.subheader("Predictions and Smart Freeze Alerts")
        st.dataframe(df[['TransactionID', 'Prediction', 'SmartFreeze']])

        # Optional download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "fraud_predictions.csv", "text/csv")
