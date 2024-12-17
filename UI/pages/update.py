import streamlit as st
from streamlit import session_state as ss
import pandas as pd
from pages.data_updating_tools.update import a_mess,clean,merge
def main():
    st.dataframe(ss.lap_data)
    st.write(f"Data shape {ss.lap_data.shape}")
    col1,col2,col3=st.columns(3)
    crawled=False
    cleaned=False
    path="data/"
    if "crawl_clicked" not in ss:
        ss.crawl_clicked=False
    def reset():
        ss.crawl_clicked=True
    
    if st.button("Crawl data",type="primary"):
        new_born = a_mess(path)
        try:
            if new_born is False:
                raise Exception()
            else:
                st.toast("Crawled data successfully!")
                st.write("Crawled data:")
                st.dataframe(new_born)
                crawled=True      
        except Exception as e:
            st.write("Something failed while crawling data")
            print(e)
        if crawled:
            st.write("Start cleaning data.")
            try:
                washed_baby=clean(new_born,path)       
            except Exception as e:
                st.write("Something failed while cleaning data")
                print(e)
            else:
                st.write("Cleaned data:")
                st.dataframe(washed_baby)
                st.toast("Data successfully cleaned!")
                cleaned=True
    def lets_merge():
        try:
            family=merge(ss.lap_data,washed_baby,path)        
        except Exception as e:
            st.write("Something failed while merging data")
            print(e)
        else:
            st.toast("Data updated!")
            ss.lap_data=family
    if cleaned:
        if st.button('Merge data',type="primary",on_click=lets_merge):
            pass