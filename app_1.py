# -*- coding: utf-8 -*-


import pandas as pd
from pycaret.regression import load_model, predict_model
import streamlit as st
st.set_page_config(page_title="Diamond Price Prediction App")
@st.cache(allow_output_mutation=True)
def get_model():
    return load_model('Catboost_regression_diamond')
def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['Label'][0]
model = get_model()
st.title("Diamond Regression App")
st.markdown("Elige los parámetros para calcular el precio del diamante")
form = st.form("diamond")
carat_weight = form.slider('Carat Weight', min_value = 0.0, max_value= 3.0,
                           value=0.0, step = 0.01, format= '%f')
value_cut = form.selectbox('Cut', ['Fair','Good','Ideal','Signature-Ideal','Very Good'])
value_color = form.selectbox('Color', ['D','E','F','G','H','I'])
value_clarity = form.selectbox('Clarity', ['FL','IF','SI1','VS1','VS2','VVS1','VVS2'])
value_polish = form.selectbox('Polish', ['EX','G','ID','VG'])
value_symmetry = form.selectbox('Symmetry', ['EX','G','ID','VG'])
value_report = form.selectbox('Report', ['AGSL','GIA'])
predict_button = form.form_submit_button('Predict')
input_dict = {'Carat Weight': carat_weight, 'Cut': value_cut, 'Color': value_color, 
              'Clarity': value_clarity, 'Polish': value_polish, 
              'Symmetry': value_symmetry,'Report': value_report}
input_df = pd.DataFrame([input_dict])
if predict_button:
    out = predict(model, input_df)
    st.success(f'La predicción del precio es ${out}.')