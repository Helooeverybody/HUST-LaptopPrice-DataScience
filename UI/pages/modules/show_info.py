import streamlit as st
import numpy as np
from streamlit import session_state as ss
import matplotlib.pyplot as plt
from pages.modules.laptop_display import get_img_url,display_poster
import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import pickle
st.set_page_config(layout="wide")
product_image_url="https://static.vecteezy.com/system/resources/thumbnails/034/555/145/small_2x/realistic-perspective-front-laptop-with-keyboard-isolated-incline-90-degree-computer-notebook-with-empty-screen-template-front-view-of-mobile-computer-with-keypad-backdrop-digital-equipment-cutout-vector.jpg"  # Replace with an actual image URL
# Sample data 


def get_similar_items(ind):
    return ss.nearest_neighbours[ind]
def predict_price(laptop_link):
    data_to_train = ss.cost_data[ss.cost_data['link'] == laptop_link].drop(columns=["link"])
    res=ss.cost_model.predict(data_to_train)
    return res[0]
def show_laptop_info():
    ind=ss.selected_item_id
    row=ss.mini_lap_data.loc[ind]
    product_specs = {
    "CPU": row["CPU: Name"],
    "RAM": row["RAM"],
    "Storage": row["Disk"],
    "Display": row["Display"],
    "GPU":row["GPU: Name"]}
    scores = {
        "Display ": row["Display Score"],
        "Portability": row["Portability Score"],
        "Work": row["Work Score"],
        "Play": row["Play Score"]
    }
    cost="???" if np.isnan(row['Cost']) else row['Cost']
    for k in scores:
        if np.isnan(scores[k]):
            scores[k]=0
    categories = list(scores.keys())
    values = list(scores.values())
    # Create a horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(categories, values, color=["#740938", "#AF1740", "#CC2B52", "#DE7C7D"], height=0.5)  # Set the figure size
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # Add labels and title
    ax.set_xlabel('Score')
    # Set up Streamlit layout
    col1,col2,col3,col4 = st.columns([1,4, 3.5,0.5])
    col2.markdown(f"<p class='highlighted-laptop-name'>{row["name"]}</p>", unsafe_allow_html=True)
    col1,col2,col3,col4 = st.columns([1,4, 3.5,0.5])
    # Display product image
    with col2:
        st.markdown(f"""
        <div class="highlighted-image-container">
            <img src={get_img_url(row["name"])} alt="Laptop Image">
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(fig)
    # Build HTML content for product specifications as a table
    specs_html = """
    <div class="specs-card">
        <h3>Specifications</h3>
        <table class="specs-table">
    """
    for spec, value in product_specs.items():
        specs_html += f"<tr><td class='spec-item'>{spec}</td><td>{value}</td></tr>"
    specs_html += "</table></div>"

    # Display the HTML content in one st.markdown call
    with col3:
        st.markdown(specs_html, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cost-box">
            {cost} $
        </div>
        """, unsafe_allow_html=True)
        if st.button("Predict price: ",type="primary"):
            pred_cost=predict_price(row.link)
            st.markdown(f"""
            <div class="predicted-cost-box">
                {pred_cost:.2f} $
            </div>""",unsafe_allow_html=True)
def show_similar_items():
    st.markdown("<h2 style='text-align: center;'>Related Products</h2>", unsafe_allow_html=True)
    # Display the related products in a row (with images and names)
    col0,col1, col2, col3,col4= st.columns([0.5,1,1,1,0.5])
    related_products_ind=get_similar_items(ss.selected_item_id)[:6]
    buttons={}
    c=0
    for i in related_products_ind:
        product=ss.mini_lap_data.loc[i]
        with [col1, col2, col3][c%3]:
            display_poster(i)
            cost=ss.mini_lap_data.loc[i].Cost
            if st.button(f"{cost}$",type="primary",key=str(i)):
                ss.selected_item_id=i
                ss.page = "page_2"  
                st.rerun() 
        c+=1