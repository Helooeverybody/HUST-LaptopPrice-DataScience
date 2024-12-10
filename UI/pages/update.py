import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import os
print(os.getcwd())
from pages.data_updating_tools.update import a_mess,merge
def update_data():
    d={"apple":[1,2,3],"banana":["a","b","b"]}
    return pd.DataFrame(d)
def main():
    st.dataframe(ss.lap_data)
    if st.button("Update data"):
        myhahaha = a_mess("data/")
        try:
            if myhahaha is False:
                raise Exception()
            else:
                merge(myhahaha,"data/")
                
        except Exception as e:
            st.write("Something failed while updating data")
        else:
            ss.lap_data=myhahaha
            st.rerun()
        