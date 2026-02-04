import pandas as pd
import joblib
import streamlit as st

model = joblib.load("model_random_forest.joblib")

st.sidebar.title(":Machine Learning")
st.sidebar.success("Dibuat oleh Farel")

st.title("tomato:Regresi Penjualan Tomat")
st.markdown("Aplikasi machine learning regression untuk menghitung total penjualan tomat berdasarkan fitur `Harga`, `Hari`, `Cuaca`, dan `Promo`")

harga = st.slider("Harga", 0, 20000, 7000)
hari = st.selectbox("Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"],index=3)
cuaca = st.selectbox("Cuaca", ["Cerah", "Berawan", "Mendung", "Hujan"])
promo = st.pills("Promo", ["Ya", "Tidak"], default="Ya")

if st.button("Pediksi"):
	data_baru = pd.DataFrame([[harga, hari, cuaca, promo]], columns=["Harga", "Hari", "Cuaca", "Promo"])
	prediksi = model.predict(data_baru)[0]
	st.success(f"Model memprediksi total penjualan tomat {prediksi:.0f}")

	st.balloons()




