import streamlit as st
from streamlit import session_state as ss
import pickle 
import random
import pandas as pd
from pages.modules.laptop_display import display_poster
from pages.modules.show_info import show_laptop_info,show_similar_items
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS
# Simulating a large list of items
laps=ss.mini_lap_data
items=laps["name"].value_counts().index.to_list()

# Simulating a function to get similar items (can be replaced with a recommendation model)
def get_similar_items(ind):
    return ss.neighbour[ind]

def page_1():
    # Initialize session state for search query and filter flag
    # Toggle to Enable/Disable Filters
    use_filters = st.sidebar.checkbox("Enable Filters", value=True)

    # Initialize filtered data
    filtered_data = laps
    # default_option="Search for a laptop"
    # selected_item = st.selectbox("Search", options=[default_option]+items, index=0,key="search_box")
    query=st.text_input("Search",value=None,key="search_box")
    def reset():
        st.session_state.search_box = None
    if "filtered" not in ss:
        ss.filtered=False
    def filter_callback():
        ss.filtered=True
    if use_filters:
        st.sidebar.header("Filter Laptops")

        # Brand Selection
        brand_filter = st.sidebar.multiselect("Choose Brand(s):", options=laps["Brand"].unique(), default=laps["Brand"].unique())

        # Price Range Slider
        price_range = st.sidebar.slider("Select Price Range:", min_value=int(laps["Cost"].min()), max_value=int(laps["Cost"].max()), value=(int(laps["Cost"].min()), int(laps["Cost"].max())))

        # Laptop Type Selection
        type_filter = st.sidebar.selectbox("Choose Laptop Type(s):", options=["Any types","Gaming","Working"], index=0)
        # Filter Button
        def match_type(row):
            if type_filter=="Any types":
                return True
            elif type_filter=="Gaming":
                return row["Play Score"]>=7.0 and row["Display Score"]>=7.0
            else:
                return row["Work Score"]>=7.0 and row["Portability Score"]>=8.0 and row["Display Score"]>=6.5
        
        if st.sidebar.button("Apply Filters",on_click=reset):
            ss.filtered=True
        if ss.filtered:
            filtered_data = laps[
                (laps["Brand"].isin(brand_filter)) &
                (laps["Cost"] >= price_range[0]) &
                (laps["Cost"] <= price_range[1]) &
                (laps.apply(match_type,axis=1)) 
            ]
    else:
        # If filters are disabled, show the full dataset
        filtered_data = laps

    # Apply Search Query (always active after filters)
    if query:
        filtered_data = filtered_data[filtered_data["name_to_display"].apply(lambda x: query.lower() in x.lower())]
    if filtered_data.empty:
        st.header("No products found")
    else:
        col1,col2,col3=st.columns(3)
        # Display Results
        ind=filtered_data.index.tolist()
        reduced_ind=ind[:50]
        c=0
        for i in reduced_ind:
            with [col1,col2,col3][c%3]:
                display_poster(i)
                if st.button("View detail",type="primary",key=str(i),on_click=filter_callback):
                    ss.selected_item_id=i
                    ss.page="page_2"
                    st.rerun()    
            c+=1
            
def page_2():
    # Back button to go back to page 1
    if st.button("Back"):
        ss.page = "page_1"  # Navigate back to page_1 without resetting the entire page
        del ss.selected_item_id  # Optionally clear selected item for a fresh start
        st.rerun()  # Force a rerun to immediately apply the session state change
    if "selected_item_id" in ss:
        show_laptop_info()
        show_similar_items()
    else:
        st.write("No item selected.")

def main():
    # Set the page state in session_state
    load_css("UI/pages/style.css")
    if "page" not in ss:
        ss.page = "page_1"

    # Navigate between pages based on the session state
    if ss.page == "page_1":
        page_1()
    elif ss.page == "page_2":
        page_2()

if __name__ == "__main__":
    main()