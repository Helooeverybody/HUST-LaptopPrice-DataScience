import streamlit as st
from streamlit_option_menu import option_menu
#from profile import main as profile_main
from pages.insights import main as insight_main
from pages.shop import main as shop_main
from pages.update import main as update_main
from streamlit import session_state as ss
import pickle 
import pandas as pd
def main():
    st.markdown("<h1 style='text-align: center; color: black;'>Laptop Paradise</h1>", unsafe_allow_html=True)
    selected=option_menu(
        menu_title=None,
        options=["Insights","Shop","Update"],
        icons=["book","bag-fill","tools"],
        orientation="horizontal"
    )
    if selected=="Insights":
        insight_main()
    elif selected=="Shop":
        shop_main()
    elif selected=="Update":
        update_main()
if __name__=="__main__":
    main()