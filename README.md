"# hotel-customer-visit-predictor" 
In this project you can use your own dataset and train the model and fine how many customers will turn up in the particular month 

It uses a machine learning Catboost Regressor to estimate the number of days between the visits

## Features

- Upload a CSV file of past customer visit records.
- Predict when each customer is likely to return.
- See how many customers are expected in a specific month.
- Built with Streamlit for an easy-to-use web interface.

##  Technologies Used

- Python
- Pandas
- CatBoost Regressor
- Scikit-learn
- Streamlit


Requirements to be installed 
pip install streamlit 
pip install catboost 
pip install pandas 
pip install joblib

Dataset Format
Your CSV file should include columns like:

customername

lastvisitdate

previousvisitdate

totalvisityear

visits_in_month

lastvisitmonth, lastvisitweekday, etc.

Dates should be in dd-mm-yyyy format.

Output
Predicts predicted_gap (days until next visit)

Adds predicted_next_visitdate

You can filter how many customers will visit in any given month
