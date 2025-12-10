import pickle
import streamlit as st
import joblib
import numpy as np

stunting_model = pickle.load(open('stunting_model.sav','rb'))
scaler = joblib.load("scaler_minmax.joblib")

st.title('Stunting Prediction App')
inputs = {}

# Gender (encoding: 0 = Laki-laki, 1 = Perempuan)
gender= st.selectbox("Jenis Kelamin", options=["Laki-laki", "Perempuan"], index=0)
inputs["gender"] = 0 if gender == "Laki-laki" else 1

# Age (bulan) - integer
inputs["age_months"] = st.number_input("Umur (bulan)", min_value=0, max_value=60, value=12, step=1, format="%d")

# Birth weight (kg) - float
inputs["birth_weight_kg"] = st.number_input("Berat Lahir (kg)", min_value=0.5, max_value=6.0, value=3.0, step=0.1, format="%.2f")

# Birth length (cm) - float
inputs["birth_length_cm"] = st.number_input("Panjang Lahir (cm)", min_value=30.0, max_value=60.0, value=49.0, step=0.5, format="%.1f")

# Current / body weight (kg) - float
inputs["current_weight_kg"] = st.number_input("Berat Badan Saat Ini (kg)", min_value=1.0, max_value=30.0, value=8.0, step=0.1, format="%.2f")

# Current / body length (cm) - float
inputs["current_length_cm"] = st.number_input("Panjang Badan Saat Ini (cm)", min_value=30.0, max_value=140.0, value=70.0, step=0.5, format="%.1f")

# Exclusive breastfeeding (ASI eksklusif 6 bulan pertama): 1 = Ya, 0 = Tidak
asi_opt = st.selectbox("ASI Eksklusif (6 bulan pertama)", options=["Ya", "Tidak"], index=0)
inputs["exclusive_breastfeeding"] = 1 if asi_opt == "Ya" else 0

if st.button('Test Prediksi Stunting'):
    X_input = np.array([[
        inputs["gender"],
        inputs["age_months"],
        inputs["birth_weight_kg"],
        inputs["birth_length_cm"],
        inputs["current_weight_kg"],
        inputs["current_length_cm"],
        inputs["exclusive_breastfeeding"]
    ]], dtype=float)
    # scaling sebelum prediksi
    X_scaled = scaler.transform(X_input)
    stunting_prediction = stunting_model.predict(X_scaled)

    if stunting_prediction[0] == 1:
        stunting_diagnosis = 'Anak Anda berisiko mengalami stunting.'
    else:
        stunting_diagnosis = 'Anak Anda tidak berisiko mengalami stunting.'

    st.write(stunting_diagnosis)
else:
    st.info('Tekan tombol "Test Prediksi Stunting" untuk mendapatkan prediksi.')