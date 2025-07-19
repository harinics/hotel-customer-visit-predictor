import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostRegressor
import joblib

# Load the model
model = joblib.load("catboost_model.pkl")
st.title("Hotel Customer Visit Predictor")
st.markdown("Predict when a customer will return to the hotel based on visit patterns.")

uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

if uploaded_file:
  
    df = pd.read_csv(uploaded_file)

    df['lastvisitdate'] = pd.to_datetime(df['lastvisitdate'], format="%d-%m-%Y")
    df['previousvisitdate'] = pd.to_datetime(df['previousvisitdate'], format="%d-%m-%Y")

    df['lastvisitmonth'] = df['lastvisitdate'].dt.month
    df['lastvisitweekday'] = df['lastvisitdate'].dt.weekday
    df['previousvisitmonth'] = df['previousvisitdate'].dt.month
    df['previousvisitweekday'] = df['previousvisitdate'].dt.weekday


    features = [
        'totalvisityear', 'lastvisitmonth', 'lastvisitweekday',
        'previousvisitmonth', 'previousvisitweekday', 'visits_in_month'
    ]

    X = df[features]
    df['predicted_gap'] = model.predict(X).round().astype(int)
    df['predicted_next_visitdate'] = df['lastvisitdate'] + pd.to_timedelta(df['predicted_gap'], unit='d')

    df['predicted_next_visitdate'] = pd.to_datetime(df['predicted_next_visitdate'])

    st.write("Sample Predictions:")
    st.dataframe(df[['customername', 'lastvisitdate', 'predicted_next_visitdate', 'predicted_gap']].head())

    st.subheader("Monthly Prediction Summary")
    selected_month = st.selectbox("Select Month", list(range(1, 13)), format_func=lambda x: datetime(2025, x, 1).strftime('%B'))
    selected_year = st.number_input("Enter Year", min_value=2024, max_value=2030, value=datetime.now().year, step=1)

    filtered = df[
        (df['predicted_next_visitdate'].dt.month == selected_month) &
        (df['predicted_next_visitdate'].dt.year == selected_year)
    ]

    st.success(f"{len(filtered)} customer(s) are expected to visit in {datetime(2025, selected_month, 1).strftime('%B')} {selected_year}.")
    if not filtered.empty:
        st.dataframe(filtered[['customername', 'predicted_next_visitdate']])
    else:
        st.info("No customer visits predicted for this month.")

    st.subheader("Calendar View: Daily Visit Count")
    daily_counts = filtered['predicted_next_visitdate'].dt.day.value_counts().sort_index()
    total_days = calendar.monthrange(selected_year, selected_month)[1]
    calendar_data = {day: daily_counts.get(day, 0) for day in range(1, total_days + 1)}

    week_layout = []
    week = []
    first_weekday = calendar.monthrange(selected_year, selected_month)[0]  # 0 = Monday

    for _ in range(first_weekday):
        week.append("")

    for day in range(1, total_days + 1):
        week.append(f"{day} ({calendar_data[day]})")
        if len(week) == 7:
            week_layout.append(week)
            week = []

    if week:
        while len(week) < 7:
            week.append("")
        week_layout.append(week)

    calendar_df = pd.DataFrame(week_layout, columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    st.dataframe(calendar_df)
