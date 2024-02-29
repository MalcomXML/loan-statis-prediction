# Loan Status Prediction App

## Overview
This Streamlit app predicts the loan status of an individual based on provided input features such as gender, car ownership, property ownership, and various other factors. The prediction is made using a Decision Tree Classifier trained on a dataset of historical loan data.

## Features
- Interactive user interface for inputting loan-related information.
- Utilizes a machine learning model to predict loan status.
- Provides a straightforward prediction output - 'Bad' or 'Good'.

## Usage
1. Visit the [Loan Status Prediction App](https://loan-status-prediction-app-jeeva.streamlit.app/) website.
2. Input your information through the user-friendly interface.
3. Click the 'Predict Loan Status' button to get the prediction result.

## Technologies Used
- Python
- Pandas
- Streamlit
- Scikit-learn

## Installation
To run the app locally, make sure you have Python and pip installed. Then, run the following commands:
```bash
pip install -r requirements.txt
streamlit run app.py
