import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle
import pandas as pd
import numpy as np

#load the model
from keras.src.legacy.saving import legacy_h5_format
model = legacy_h5_format.load_model_from_hdf5("model.h5")


 ##load the encoder ans scaler
with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)
with open('labal_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

## streamlit app
st.title("Customer Churn Prediction")

#user inputs

geography=st.selectbox("Geography",onehot_encoder_geo.categories_[0])
gender=st.selectbox("Gender",label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

#prepare the input data
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

#combine all features
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Load column order
with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

# Reorder input_data to match training
input_data = input_data.reindex(columns=columns, fill_value=0)

#scale the input data
input_data_scaled=scaler.transform(input_data)

#make prediction
prediction=model.predict(input_data_scaled)
churn_probability=prediction[0][0]

if churn_probability > 0.5:
    st.write(f"The customer is likely to churn with a probability of {churn_probability:.8f}")
    st.write(f"📊 Rounded probability: {churn_probability:.2f}")
else:
    st.write(f"The customer is unlikely to churn with a probability of {churn_probability:.8f}")
    st.write(f"📊 Rounded probability: {churn_probability:.2f}")

