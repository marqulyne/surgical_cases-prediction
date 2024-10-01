import streamlit as st
import joblib
import pandas as pd

# Load the models
surgical_model = joblib.load('surgical_cases_model.joblib')

st.title("Surgical Cases Prediction App")

tab1, tab2 = st.tabs(["Prediction", "About"])

with tab1:
    st.header("Enter Patient Data for Prediction")

    medical = st.number_input("Medical Cases", min_value=0, value=100)
    paediatrics = st.number_input("Pediatric Cases", min_value=0, value=50)
    neonate = st.number_input("Neonate Cases", min_value=0, value=20)
    covid_19 = st.number_input("COVID-19 Cases", min_value=0, value=30)
    radiology = st.number_input("Radiology Cases", min_value=0, value=70)
    ph = st.number_input("Public Health Indicator", min_value=0.0, value=1.0)

    # Select which model to use for prediction
    model_choice = st.selectbox("Select the model to use for prediction", 
                                  ["Surgical Cases Model", "Regression Model"])

    if st.button("Predict"):
        input_data = pd.DataFrame({
            'Medical': [medical],
            'Paediatrics': [paediatrics],
            'Neonate': [neonate],
            'Covid_19': [covid_19],
            'Radiology': [radiology],
            'PH': [ph]
        })

        input_data_encoded = pd.get_dummies(input_data, drop_first=True)
        
        # Adjust the feature names based on the chosen model
        if model_choice == "Surgical Cases Model":
            input_data_encoded = input_data_encoded.reindex(columns=surgical_model.feature_names_in_, fill_value=0)
            prediction = surgical_model.predict(input_data_encoded)
        else:
            input_data_encoded = input_data_encoded.reindex(columns=regression_model.feature_names_in_, fill_value=0)
            prediction = regression_model.predict(input_data_encoded)

        st.write(f"Predicted surgical cases: {prediction[0]}")  # Display the prediction result

with tab2:
    st.header("About This App")
    st.write("""This application predicts the number of surgical cases based on various medical indicators.
    Enter the required patient data, select the model, and click 'Predict' to see the results.""")
    st.write("Developed by [Marquline Opiyo].")
