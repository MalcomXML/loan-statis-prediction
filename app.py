import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier


# Assuming 'df' is your dataframe
# Selecting features and target
df = pd.read_csv('data\df.csv')
X = df.drop('STATUS', axis=1)
y = df['STATUS']

# Define preprocessing steps for numeric and categorical features
numeric_features = ['NO OF CHILDREN', 'INCOME TOTAL', 'AGE', 'FAMILY SIZE']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['GENDER', 'CAR', 'PROPERTY', 'EDUCATION TYPE', 'FAMILY STATUS', 'HOUSING TYPE', 'OCCUPATION TYPE', 'INCOME TYPE']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Append classifier to preprocessing pipeline with Decision Tree
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', DecisionTreeClassifier(random_state=42))])

# Fit the model on the entire data
model.fit(X, y)

# STREAMLIT APP

# Title
st.title('Loan Status Prediction App')

# Collecting user inputs
gender = st.selectbox('Gender', ['M', 'F'])
car = st.selectbox('Car Ownership', ['Y', 'N'])
property_ownership = st.selectbox('Property Ownership', ['Y', 'N'])
num_children = st.slider('Number of Children', 0, 10, 1)
income_total = st.slider('Total Income', 0, 300000, 5000)
age = st.slider('Age', 18, 80, 30)
family_size = st.slider('FAMILY SIZE', 1, 10, 3)
education_type = st.selectbox('Education Type', df['EDUCATION TYPE'].unique())
family_status = st.selectbox('Family Status', df['FAMILY STATUS'].unique())
housing_type = st.selectbox('Housing Type', df['HOUSING TYPE'].unique())
occupation_type = st.selectbox('Occupation Type', df['OCCUPATION TYPE'].unique())
income_type = st.selectbox('Income Type', df['INCOME TYPE'].unique())

# Button to trigger prediction
if st.button('Predict Loan Status'):
    # Make prediction
    input_data = pd.DataFrame({
        'GENDER': [gender],
        'CAR': [car],
        'PROPERTY': [property_ownership],
        'NO OF CHILDREN': [num_children],
        'INCOME TOTAL': [income_total],
        'AGE': [age],
        'FAMILY SIZE': [family_size],
        'EDUCATION TYPE': [education_type],
        'FAMILY STATUS': [family_status],
        'HOUSING TYPE': [housing_type],
        'OCCUPATION TYPE': [occupation_type],
        'INCOME TYPE': [income_type]
    })

    prediction = model.predict(input_data)

    # Display the prediction
    st.subheader('Prediction:')
    loan_status = 'Bad' if prediction[0] == 0 else 'Good'
    st.write(f'The predicted loan status is: {loan_status}')

