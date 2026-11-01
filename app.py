
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Forest Cover Prediction",
    page_icon="🌲",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(
        Path(__file__).parent / "forest_cover_best_model.joblib"
    )

bundle = load_model()
model = bundle["model"]
features = bundle["features"]

forest_types = {
    1: "Spruce / Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood / Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

def feature_engineering(df):
    X = df.copy()

    h = X["Horizontal_Distance_To_Hydrology"]
    v = X["Vertical_Distance_To_Hydrology"]
    road = X["Horizontal_Distance_To_Roadways"]
    fire = X["Horizontal_Distance_To_Fire_Points"]

    X["Hydrology_3D_Distance"] = np.hypot(h, v)
    X["Elevation_Hydro_Diff"] = X["Elevation"] - v
    X["Road_Fire_Diff"] = road - fire
    X["Road_Fire_Ratio"] = road / (fire + 1)
    X["Hillshade_Diff"] = X["Hillshade_9am"] - X["Hillshade_3pm"]
    X["Hillshade_Mean"] = X[
        ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    ].mean(axis=1)

    return X

st.title("🌲 Forest Cover Type Prediction")
st.write("Random Forest + Feature Engineering")
st.metric("Validation Accuracy", "87.17%")

st.info(
    "Colorado Roosevelt National Forest verileriyle "
    "eğitilmiş sınıflandırma modeli."
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Arazi Bilgileri")

    elevation = st.number_input("Rakım (m)", value=2800)
    aspect = st.slider("Yön (derece)", 0, 360, 180)
    slope = st.slider("Eğim (derece)", 0, 90, 15)
    h_hydro = st.number_input(
        "Su kaynağına yatay uzaklık (m)", min_value=0, value=300
    )
    v_hydro = st.number_input(
        "Su kaynağına dikey uzaklık (m)", value=30
    )
    road = st.number_input(
        "Yola uzaklık (m)", min_value=0, value=1500
    )
    fire = st.number_input(
        "Yangın noktalarına uzaklık (m)", min_value=0, value=2000
    )

with col2:
    st.subheader("Çevresel Özellikler")

    shade9 = st.slider("Hillshade 9 AM", 0, 255, 220)
    shade12 = st.slider("Hillshade Noon", 0, 255, 230)
    shade3 = st.slider("Hillshade 3 PM", 0, 255, 150)

    wilderness = st.selectbox(
        "Wilderness Area",
        [1, 2, 3, 4]
    )

    soil = st.selectbox(
        "Soil Type",
        list(range(1, 41))
    )

# Modelin orijinal 54 değişkenini oluştur
row = {
    "Elevation": elevation,
    "Aspect": aspect,
    "Slope": slope,
    "Horizontal_Distance_To_Hydrology": h_hydro,
    "Vertical_Distance_To_Hydrology": v_hydro,
    "Horizontal_Distance_To_Roadways": road,
    "Hillshade_9am": shade9,
    "Hillshade_Noon": shade12,
    "Hillshade_3pm": shade3,
    "Horizontal_Distance_To_Fire_Points": fire
}

for i in range(1, 5):
    row[f"Wilderness_Area{i}"] = int(i == wilderness)

for i in range(1, 41):
    row[f"Soil_Type{i}"] = int(i == soil)

if st.button("Orman Türünü Tahmin Et", type="primary"):
    X = pd.DataFrame([row])

    if bundle["engineered"]:
        X = feature_engineering(X)

    missing = set(features) - set(X.columns)
    if missing:
        st.error(f"Eksik model özellikleri: {missing}")
    else:
        X = X[features]
        prediction = int(model.predict(X)[0])

        st.success(
            f"Tahmin edilen tür: {forest_types[prediction]}"
        )

st.caption(
    "Dataset: Kaggle Forest Cover Type Prediction. "
    "Sonuçlar model tahminidir."
)
