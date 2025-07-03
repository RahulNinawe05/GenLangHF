

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# import pandas as pd
# import pickle

# # Load the trained model (correct path)
# model = tf.keras.models.load_model('../model.h5')

# # Load the encoders and scaler
# with open('../label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender = pickle.load(file)

# with open('../onehot_encoder_geo.pkl', 'rb') as file:
#     onehot_encoder_geo = pickle.load(file)

# with open('../scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)

# # ----------------------------
# # Streamlit App
# # ----------------------------
# st.title('Customer Churn Prediction')

# # User Input
# geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
# gender = st.selectbox('Gender', label_encoder_gender.classes_)
# age = st.slider('Age', 18, 92)
# balance = st.number_input('Balance')
# credits_score = st.number_input('Credit Score')
# estimated_salary = st.number_input('Estimated Salary')
# tenure = st.slider('Tenure', 0, 10)
# num_of_products = st.slider('Number of Products', 1, 4)
# has_cr_card = st.selectbox('Has Credit Card', [0, 1])
# is_active_member = st.selectbox('Is Active Member', [0, 1])

# # Encode Gender
# gender_encoded = label_encoder_gender.transform([gender])[0]

# # Construct input dataframe
# input_data = pd.DataFrame({
#     'CreditScore': [credits_score],
#     'Gender': [gender_encoded],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [has_cr_card],
#     'IsActiveMember': [is_active_member],
#     'EstimatedSalary': [estimated_salary]
# })

# # One-hot encode Geography
# geo_encoded = onehot_encoder_geo.transform([[geography]])
# geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# # Combine
# input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

# # Scale
# scaled_input_data = scaler.transform(input_data)

# # Predict
# prediction = model.predict(scaled_input_data)
# prediction_prob = prediction[0][0]

# # Display result
# if prediction_prob > 0.5:
#     st.error('⚠️ The Customer is likely to Churn.')
# else:
#     st.success('✅ The Customer is not likely to Churn.')


import os
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Path to this file's folder
base_path = os.path.dirname(__file__)

# Load model and assets from 'projects/' folder
model = tf.keras.models.load_model(os.path.join(base_path, 'model.h5'))

with open(os.path.join(base_path, 'label_encoder_gender.pkl'), 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(os.path.join(base_path, 'onehot_encoder_geo.pkl'), 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(os.path.join(base_path, 'scaler.pkl'), 'rb') as file:
    scaler = pickle.load(file)


# ----------------------------
# Streamlit App
# ----------------------------
st.title('Customer Churn Prediction')

# User Input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credits_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Encode Gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# Construct input dataframe
input_data = pd.DataFrame({
    'CreditScore': [credits_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography (as dense array)
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

# Use correct column names
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out())

# Combine input and geo-encoded data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

# Scale input
scaled_input_data = scaler.transform(input_data)

# Predict
prediction = model.predict(scaled_input_data)
prediction_prob = prediction[0][0]

st.write(f'Churn Probablity : {prediction_prob: .2f}')

# Display result
if prediction_prob > 0.5:
    st.error('⚠️ The Customer is likely to Churn.')
else:
    st.success('✅ The Customer is not likely to Churn.')

