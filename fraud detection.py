import streamlit as st
import pandas as pd

st.title("Smart Freeze Fraud Detection")

uploaded_file = st.file_uploader("Upload your transaction CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if all(col in df.columns for col in ['TransactionID', 'TransactionAmount', 'Account Balance', 'Login Attempts']):
        df['Amount/Balance'] = df['TransactionAmount'] / (df['Account Balance'] + 1)
        df['High Login Attempts'] = df['Login Attempts'].apply(lambda x: 1 if x > 3 else 0)
        df['is_fraud'] = df.apply(lambda row: 1 if row['Amount/Balance'] > 0.5 or row['High Login Attempts'] == 1 else 0, axis=1)
        
        for i, row in df.iterrows():
            if row['is_fraud'] == 1:
                st.error(f"ALERT: Transaction {row['TransactionID']} is likely FRAUDULENT. Freezing app features.")
            else:
                st.success(f"Transaction {row['TransactionID']} is safe.")
        
        st.subheader("Data with Predictions")
        st.dataframe(df[['TransactionID', 'is_fraud']])

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", csv, "fraud_results.csv", "text/csv")
    else:
        st.warning("Required columns not found: Please include TransactionID, TransactionAmount, Account Balance, Login Attempts")
else:
    st.info("Please upload a CSV file to begin.")
