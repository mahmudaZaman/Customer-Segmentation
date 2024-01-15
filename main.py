import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from src.features.data_transformation import data_preprocess
from src.models.train_model import run_train_pipeline



def streamlit_run():
    # Load the pre-trained model
    with open('/Users/shuchi/Documents/work/personal/Customer-Segmentation/src/artifacts/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Streamlit UI
    st.title('Customer Segmentation Prediction')

    # Collect input features
    gender = st.selectbox('Gender', ['Male', 'Female'])
    ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
    age = st.slider('Age', 18, 100, 25)
    graduated = st.selectbox('Graduated', ['Yes', 'No'])
    profession = st.selectbox('Profession',
                              ['Artist', 'Healthcare', 'Entertainment', 'Doctor', 'Engineer', 'Executive', 'Homemaker',
                               'Lawyer', 'Marketing', 'None'])
    work_experience = st.slider('Work Experience', 0, 50, 0)
    spending_score = st.selectbox('Spending Score', ['Low', 'Average', 'High'])
    family_size = st.slider('Family Size', 1, 10, 1)
    var_1 = st.selectbox('Var 1', ['Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5'])

    # Create a DataFrame with user input
    new_customer_data = pd.DataFrame({
        'Gender': [gender],
        'Ever_Married': [ever_married],
        'Age': [age],
        'Graduated': [graduated],
        'Profession': [profession],
        'Work_Experience': [work_experience],
        'Spending_Score': [spending_score],
        'Family_Size': [family_size],
        'Var_1': [var_1]
    })
    if st.button('Do Customer Segmentation'):
        # Preprocess the input data
        new_customer_data = data_preprocess(new_customer_data)
        # Make predictions
        prediction = model.predict(new_customer_data)
        # Display the predicted segment
        st.subheader('Predicted Customer Segment:')
        st.write('Customer Segment:', prediction[0])


def model_run():
    run_train_pipeline()


if __name__ == '__main__':
    mode = os.getenv("mode", "streamlit")
    print("mode", mode)
    if mode == "model":
        model_run()
    else:
        streamlit_run()
