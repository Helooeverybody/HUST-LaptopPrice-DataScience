import streamlit as st
from streamlit import session_state as ss
import pandas as pd
def update_data():
    d={"apple":[1,2,3],"banana":["a","b","b"]}
    return pd.DataFrame(d)
def main():
    st.dataframe(ss.lap_data)
    if st.button("Update data"):
        try:
            new_df=update_data()
        except Exception as e:
            st.write("Something failed while updating data")
        else:
            ss.lap_data=new_df
            st.rerun()
        