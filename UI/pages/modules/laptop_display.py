import streamlit as st
from streamlit import session_state as ss
from pathlib import Path
def get_img_url(name):
    default_image_url="https://static.vecteezy.com/system/resources/thumbnails/034/555/145/small_2x/realistic-perspective-front-laptop-with-keyboard-isolated-incline-90-degree-computer-notebook-with-empty-screen-template-front-view-of-mobile-computer-with-keypad-backdrop-digital-equipment-cutout-vector.jpg"
    if name in ss.img_url_dict:
        img_url=ss.img_url_dict[name]
    else: img_url=default_image_url
    return img_url
def display_poster(id):
    row = ss.mini_lap_data.loc[id]
    name = row['name']
    name_to_display=row["name_to_display"]
    st.markdown(f"""
        <div class="laptop-poster-container">
            <div class="normal-image-container">
                <img src={get_img_url(name)} alt="Laptop Image">
            </div>
            <p class='laptop-name' >{name_to_display}</p>
        <div>
        """, unsafe_allow_html=True)