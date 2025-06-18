import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostRegressor
import joblib

# Load the model
model = joblib.load(r"C:\Users\Harini CS\Downloads\catboost_model.pkl")

st.title("🏨 Hotel Customer Visit Predictor")
st.markdown("Predict when a customer will return to the hotel based on visit patterns.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert date columns
    df['lastvisitdate'] = pd.to_datetime(df['lastvisitdate'], format="%d-%m-%Y")
    df['previousvisitdate'] = pd.to_datetime(df['previousvisitdate'], format="%d-%m-%Y")

    # Extract features
    df['lastvisitmonth'] = df['lastvisitdate'].dt.month
    df['lastvisitweekday'] = df['lastvisitdate'].dt.weekday
    df['previousvisitmonth'] = df['previousvisitdate'].dt.month
    df['previousvisitweekday'] = df['previousvisitdate'].dt.weekday

    features = [
        'totalvisityear', 'lastvisitmonth', 'lastvisitweekday',
        'previousvisitmonth', 'previousvisitweekday', 'visits_in_month'
    ]

    X = df[features]

    # Predict visit gap and next visit date
    df['predicted_gap'] = model.predict(X).round().astype(int)
    df['predicted_next_visitdate'] = df['lastvisitdate'] + pd.to_timedelta(df['predicted_gap'], unit='d')

    # Show preview
    st.write("Sample Predictions:")
    st.dataframe(df[['customername', 'lastvisitdate', 'predicted_next_visitdate', 'predicted_gap']].head())

    # --- Filter by Month ---
    st.subheader("📅 Monthly Prediction Summary")

    selected_month = st.selectbox("Select Month", list(range(1, 13)), format_func=lambda x: datetime(2025, x, 1).strftime('%B'))
    selected_year = st.number_input("Enter Year", min_value=2024, max_value=2030, value=datetime.now().year, step=1)

    filtered = df[
        (df['predicted_next_visitdate'].dt.month == selected_month) &
        (df['predicted_next_visitdate'].dt.year == selected_year)
    ]

    st.success(f"{len(filtered)} customer(s) are expected to visit in {datetime(2025, selected_month, 1).strftime('%B')} {selected_year}.")
    st.dataframe(filtered[['customername', 'predicted_next_visitdate']])
