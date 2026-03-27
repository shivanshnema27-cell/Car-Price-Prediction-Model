import streamlit as st
import numpy as np
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    with open("scaler.pkl", "rb") as f:
        scaler = joblib.load(f)
    with open("ridge_model.pkl", "rb") as f:
        model = joblib.load(f)
    return scaler, model

scaler, model = load_model()

# -------------------- FEATURE LIST --------------------
feature_names = [
    'symboling', 'wheelbase', 'carlength', 'carwidth', 'curbweight',
    'enginesize', 'horsepower', 'citympg',
    
    'carbody_convertible', 'carbody_hardtop', 'carbody_hatchback',
    'carbody_sedan', 'carbody_wagon',
    
    'drivewheel_4wd', 'drivewheel_fwd', 'drivewheel_rwd',
    
    'enginelocation_front', 'enginelocation_rear',
    
    'enginetype_dohc', 'enginetype_dohcv', 'enginetype_l',
    'enginetype_ohc', 'enginetype_ohcf', 'enginetype_ohcv',
    'enginetype_rotor',
    
    'cylindernumber_eight', 'cylindernumber_five',
    'cylindernumber_four', 'cylindernumber_six',
    'cylindernumber_three', 'cylindernumber_twelve',
    'cylindernumber_two'
]

# -------------------- STYLING --------------------
st.markdown("""
<style>
.stButton>button {
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    color: black;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 30px;
        font-weight: bold;">
        🚗 Car Price Prediction Dashboard
    </div>
    """,
    unsafe_allow_html=True
)

st.write("### 📊 Enter Car Details")

# -------------------- NUMERICAL INPUTS --------------------
col1, col2 = st.columns(2)

with col1:
    symboling = st.number_input("Symboling")
    wheelbase = st.number_input("Wheelbase")
    carlength = st.number_input("Car Length")
    carwidth = st.number_input("Car Width")

with col2:
    curbweight = st.number_input("Curb Weight")
    enginesize = st.number_input("Engine Size")
    horsepower = st.number_input("Horsepower")
    citympg = st.number_input("City MPG")

# -------------------- CATEGORICAL INPUTS --------------------
carbody = st.selectbox("Car Body", 
    ['convertible','hardtop','hatchback','sedan','wagon'])

drivewheel = st.selectbox("Drive Wheel",
    ['4wd','fwd','rwd'])

enginelocation = st.selectbox("Engine Location",
    ['front','rear'])

enginetype = st.selectbox("Engine Type",
    ['dohc','dohcv','l','ohc','ohcf','ohcv','rotor'])

cylinders = st.selectbox("Cylinder Number",
    ['eight','five','four','six','three','twelve','two'])

# -------------------- ENCODING FUNCTION --------------------
def encode_input():
    data = {
        'symboling': symboling,
        'wheelbase': wheelbase,
        'carlength': carlength,
        'carwidth': carwidth,
        'curbweight': curbweight,
        'enginesize': enginesize,
        'horsepower': horsepower,
        'citympg': citympg,
    }

    # initialize all features to 0
    for col in feature_names:
        if col not in data:
            data[col] = 0

    # set selected categories = 1
    data[f'carbody_{carbody}'] = 1
    data[f'drivewheel_{drivewheel}'] = 1
    data[f'enginelocation_{enginelocation}'] = 1
    data[f'enginetype_{enginetype}'] = 1
    data[f'cylindernumber_{cylinders}'] = 1

    return np.array([data[col] for col in feature_names]).reshape(1, -1)

# -------------------- PREDICTION --------------------
if st.button("🔮 Predict Price"):
    X = encode_input()

    with st.spinner("⏳ Predicting..."):
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)

    st.success("✅ Prediction Complete!")

    st.markdown(
        f"""
        <div style="
            background: #1E1E1E;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            color: white;">
            <h3>💰 Predicted Car Price</h3>
            <h1 style='color:#00FFAA;'>{prediction[0]:.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric("Predicted Price", f"{prediction[0]:.2f}")

# -------------------- SIDEBAR --------------------
st.sidebar.title("ℹ️ About")
st.sidebar.write("Model: Ridge / Lasso Regression")
st.sidebar.write("✔ Proper One-Hot Encoding")
st.sidebar.write("✔ Clean UI")
st.sidebar.write("✔ Accurate Feature Mapping")

# -------------------- FOOTER --------------------
st.markdown(
    "<hr><p style='text-align:center;color:gray;'>Built with Streamlit 🚀</p>",
    unsafe_allow_html=True
)