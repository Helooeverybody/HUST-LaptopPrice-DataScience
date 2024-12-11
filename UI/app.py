import streamlit as st
import pandas as pd
from streamlit import session_state as ss
import pickle 
import pandas as pd
st.set_page_config(layout="wide")
path="UI/data/"

@st.cache_data
def load_mini_data():
    with open(path+"small_lap_data.pkl","rb") as f:
        return pickle.load(f)
@st.cache_data
def load_neighbours():
    with open(path+"neighbours.pkl","rb") as game_pkl:
        data=pickle.load(game_pkl)
    return data
@st.cache_data
def load_image_url():
    with open(path+"laptop_image_url.pkl","rb") as img_url_dict:
        data=pickle.load(img_url_dict)
    return data
@st.cache_data
def load_lap_data():
    data=pd.read_csv("data/laptop_final.csv")
    return data
@st.cache_data
def load_cost_model():
    with open(path+"cost_prediction_model.pkl","rb") as f:
        data=pickle.load(f)
    return data
@st.cache_data
def load_cost_data():
    with open(path+"cost_pred_preprocessed_data.pkl","rb") as f:
        data=pickle.load(f)
    return data
ss.lap_data=load_lap_data()
ss.mini_lap_data=load_mini_data()
ss.nearest_neighbours=load_neighbours()
ss.img_url_dict=load_image_url()
ss.cost_data=load_cost_data()
ss.cost_model=load_cost_model()
st.markdown("<h1 style='text-align: center; color: black;'>LAPTOP KINGDOM</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2.5, 1, 2])  # Adjust column widths for better centering

col1.write("")
col3.write("")

# Create a large button with centered text in the center column
with col2:
    submit_button=st.page_link("pages/main.py",label="LET'S GO")