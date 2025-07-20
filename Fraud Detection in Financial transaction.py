import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.title("AI-Based Fraud Detection with Smart Freeze")

uploaded_file = st.file_uploader("Upload the original dataset (CSV only)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.write(df.head())

        # Feature Engineering
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
        df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])
        df['TimeGap'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds() / 60
        df['AmountToBalance'] = df['TransactionAmount'] / (df['AccountBalance'] + 1)
        df['HighLoginAttempts'] = df['LoginAttempts'].apply(lambda x: 1 if x > 3 else 0)

        # Simulated labels: For demonstration, let's say transactions over 50,000 + high login attempts = fraud
        df['is_fraud'] = ((df['TransactionAmount'] > 50000) & (df['LoginAttempts'] > 3)).astype(int)

        st.subheader("Processed Data Preview")
        st.write(df[['TransactionID', 'TransactionAmount', 'LoginAttempts', 'is_fraud']].head())

        # Select features and target
        features = ['TransactionAmount', 'TimeGap', 'AmountToBalance', 'HighLoginAttempts']
        X = df[features].fillna(0)
        y = df['is_fraud']

        # Train model
        model = RandomForestClassifier()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display model report
        st.subheader("Model Report")
        st.text(classification_report(y_test, y_pred))

        # Run predictions on all
        df['Prediction'] = model.predict(X)
        df['Frozen Features'] = df['Prediction'].apply(lambda x: ["Transfer Money", "Change Password"] if x == 1 else [])

        st.subheader("Fraud Detection Results")
        st.write(df[['TransactionID', 'Prediction', 'Frozen Features']].head(20))

        # Alert Message
        fraud_count = df[df['Prediction'] == 1].shape[0]
        if fraud_count > 0:
            st.error("ALERT: Fraudulent transactions detected! Some features have been frozen.")
        else:
            st.success("No fraudulent activity detected.")

    except Exception as e:
        st.error(f"Error loading or processing file: {e}")
else:
    st.info("Please upload the original CSV file provided for the hackathon.")
