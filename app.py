import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load('disease_prediction_model.pkl')

# Symptom list
symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills',
            'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting',
            'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets',
            'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level',
            'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion',
            'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes',
            'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
            'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
            'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
            'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
            'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
            'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
            'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
            'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
            'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
            'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
            'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
            'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation',
            'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history',
            'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
            'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
            'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
            'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
            'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
            'yellow_crust_ooze']

# Streamlit page settings
st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="wide")

st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Predict Disease", "About"])

# Home Page
if page == "Home":
    st.title("🩺 Welcome to the Disease Prediction System")
    st.write("Predict diseases based on symptoms using Machine Learning.")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Symptoms", f"{len(symptoms)}+")
    col2.metric("ML Model", "Random Forest")
    col3.metric("Prediction Speed", "Fast ✅")

    st.subheader("How it works:")
    st.markdown("""
    - Select your symptoms
    - Predict top 3 likely diseases
    - See results with probabilities
    """)

# Predict Disease Page
elif page == "Predict Disease":
    st.title("🔍 Predict Your Disease")
    selected_symptoms = st.multiselect("✅ Select your symptoms:", symptoms)

    if selected_symptoms:
        with st.expander("🔎 View Selected Symptoms"):
            st.write(selected_symptoms)

    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom.")
        else:
            with st.spinner("⏳ Predicting..."):
                # Progress bar
                progress = st.progress(0)
                for percent_complete in range(0, 101, 20):
                    progress.progress(percent_complete)
                
                input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
                input_data = np.array(input_data).reshape(1, -1)

                probs = model.predict_proba(input_data)
                top3 = np.argsort(probs[0])[-3:][::-1]
                predicted_classes = model.classes_[top3]
                probabilities = probs[0][top3]

                st.success("✅ Prediction Completed!")
                st.subheader("📈 Prediction Results:")

                # Show metrics
                for i in range(3):
                    st.metric(label=f"{i+1}. {predicted_classes[i]}", value=f"{probabilities[i]*100:.2f}%")

                # Visualization - Bar Chart
                st.subheader("🔵 Probability Chart")
                fig, ax = plt.subplots()
                ax.barh(predicted_classes[::-1], probabilities[::-1], color=['#1f77b4', '#2ca02c', '#ff7f0e'])
                ax.set_xlabel("Probability")
                ax.set_xlim(0,1)
                st.pyplot(fig)

# About Page
elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    - **Project:** Disease Prediction Based on Symptoms  
    - **Developed Using:** Python, Scikit-Learn, Streamlit  
    - **Goal:** Assist early disease detection using Machine Learning  
    - **Made With ❤️ for Healthcare Innovation**
    """)
