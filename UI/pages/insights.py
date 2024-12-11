import streamlit as st
from streamlit import session_state as ss
import pickle 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from copy import deepcopy
df=ss.lap_data
df1 = deepcopy(df)
df1.dropna(subset = ["Work Score","Play Score", "Portability Score","Display Score","Total Score", "Cost"], inplace = True)
df1 = df1[df1["Cost"]<=10000]
bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 5000, 7000, 10000] 
labels = [f"${int(bins[i])}-${int(bins[i+1])}" if bins[i+1] != float('inf') else f"{int(bins[i])}+" 
          for i in range(len(bins) - 1)]
df1['Price Segment'] = pd.cut(df1['Cost'], bins=bins, labels=labels, right=False)
df1['Brand'] = df1['name'].apply(lambda x : x.split()[0])
grouped = df1.groupby(['Price Segment', 'Brand']).agg({
    'Work Score': 'mean',
    'Play Score': 'mean',
    'Display Score': 'mean',
    'Portability Score': 'mean',
    'Total Score': 'mean'
}).reset_index()
scores = [x+" Score" for x in ['Total', 'Display', 'Play', 'Work', 'Portability']]
def brands_dominance_over_bins():
    df2=deepcopy(df)
    df2 = df2[~df2["Cost"].isna()]
    df2 = df2[~df2["Total Score"].isna()]

    df2["Brand"] = df2["name"].apply(lambda a: a.split()[0])

    # Define bins for cost and score
    bins={
    "Cost": [0, 500, 1000, 1500, 2000, 2500, 3000, 5000, 7000, 10000] , # Adjust based on dataset
    "Score": [0, 3, 5, 7, 9, 10],
    }
    labels={
    "Cost": ["<500", "500-1000", "1000-1500", "1500-2000", "2000-2500", "2500-3000", "3000-5000", "5000-7000", "7000-10000"],
    "Score": ["0-3", "3-5", "5-7", "5-9", "9-10"]
    }
    # Define bins for cost and score
    cost_bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 5000, 7000, 10000]  # Adjust based on dataset
    score_bins = [0, 3, 5, 7, 9, 10]  # Adjust based on dataset
    cost_labels = ["<500", "500-1000", "1000-1500", "1500-2000", "2000-2500", "2500-3000", "3000-5000", "5000-7000", "7000-10000"]
    score_labels = ["0-3", "3-5", "5-7", "7-9", "9-10"]

    # Bin the data
    df2["Cost Bin"] = pd.cut(df2["Cost"], bins=cost_bins, labels=cost_labels, right=False)
    df2["Total Score Bin"] = pd.cut(df2["Total Score"], bins=score_bins, labels=score_labels, right=False)
    df2["Play Score Bin"] = pd.cut(df2["Play Score"], bins=score_bins, labels=score_labels, right=False)
    df2["Work Score Bin"] = pd.cut(df2["Work Score"], bins=score_bins, labels=score_labels, right=False)
    df2["Display Score Bin"] = pd.cut(df2["Display Score"], bins=score_bins, labels=score_labels, right=False)
    df2["Port Score Bin"] = pd.cut(df2["Portability Score"], bins=score_bins, labels=score_labels, right=False)
    
    # Group by Brand and Bin
    brand_cost_dist = df2.groupby(["Brand", "Cost Bin"]).size().unstack(fill_value=0)
    brand_score_dist = df2.groupby(["Brand", "Total Score Bin"]).size().unstack(fill_value=0)
    brand_play_score_dist = df2.groupby(["Brand", "Play Score Bin"]).size().unstack(fill_value=0)
    brand_work_score_dist = df2.groupby(["Brand", "Work Score Bin"]).size().unstack(fill_value=0)
    brand_display_score_dist = df2.groupby(["Brand", "Display Score Bin"]).size().unstack(fill_value=0)
    brand_port_score_dist = df2.groupby(["Brand", "Port Score Bin"]).size().unstack(fill_value=0)
    # Normalize within bins
    brand_cost_dist_normalized = brand_cost_dist.div(brand_cost_dist.sum(axis=0), axis=1) * 100
    brand_score_dist_normalized = brand_score_dist.div(brand_score_dist.sum(axis=0), axis=1) * 100
    brand_play_score_dist_normalized = brand_play_score_dist.div(brand_play_score_dist.sum(axis=0), axis=1 ) * 100
    brand_work_score_dist_normalized = brand_work_score_dist.div(brand_work_score_dist.sum(axis=0), axis=1) * 100
    brand_display_score_dist_normalized = brand_display_score_dist.div(brand_display_score_dist.sum(axis=0), axis=1) * 100
    brand_port_score_dist_normalized = brand_port_score_dist.div(brand_port_score_dist.sum(axis=0), axis=1) * 100

    colormap = sns.color_palette("Set2", n_colors=len(brand_cost_dist_normalized.columns))
    criterion=st.selectbox("Select criterion",["Cost","Total Score","Display Score","Work Score","Portability Score","Play Score"])
    draw_assets={
        "Cost":brand_cost_dist,
        "Total Score": brand_score_dist,
        "Play Score": brand_play_score_dist,
        "Work Score": brand_work_score_dist,
        "Display Score": brand_display_score_dist,
        "Portability Score": brand_port_score_dist
    }
    def brands_dominance_over_bins_vis(feature):
        # Plot brand dominance by cost bin
        plt.figure(figsize=(9, 6))
        draw_assets[feature].T.plot(kind="bar", stacked=True, colormap="tab20", figsize=(12, 6))
        plt.title(f"Brand Dominance Across {feature} Bins")
        plt.xlabel(feature+ " Bin")
        plt.ylabel("Percentage of Laptops")
        plt.legend(title="Brand", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(plt)
    brands_dominance_over_bins_vis(criterion)
def brand_with_best_quality_over_price_range():
    # data=ss.lap_data
    # data['Price Segment'] = pd.qcut(data['Cost'], q=5, labels=['Low', 'Lower Mid', 'Mid', 'Upper Mid', 'High'])
    # data["Brand"] = data["name"].apply(lambda a: a.split()[0])
    # grouped = data.groupby(['Price Segment', 'Brand']).agg({
    # 'Work Score': 'mean',
    # 'Play Score': 'mean',
    # 'Display Score': 'mean',
    # 'Portability Score': 'mean',
    # 'Total Score': 'mean'
    # }).reset_index()

    best_work = grouped.loc[grouped.groupby('Price Segment')['Work Score'].idxmax()]
    best_play = grouped.loc[grouped.groupby('Price Segment')['Play Score'].idxmax()]
    best_display = grouped.loc[grouped.groupby('Price Segment')['Display Score'].idxmax()]
    best_port = grouped.loc[grouped.groupby('Price Segment')['Portability Score'].idxmax()]
    best_total = grouped.loc[grouped.groupby('Price Segment')['Total Score'].idxmax()]

    best_work['Category'] = 'Work'
    best_work['Score'] = best_work['Work Score']

    best_play['Category'] = 'Gaming'
    best_play['Score'] = best_play['Play Score']

    best_display['Category'] = 'Display'
    best_display['Score'] = best_display['Display Score']

    best_port['Category'] = 'Portability'
    best_port['Score'] = best_port['Portability Score']

    best_total['Category'] = 'General'
    best_total['Score'] = best_total['Total Score']

    best_brands = pd.concat([best_work, best_play, best_display,best_port, best_total]).reset_index(drop=True)
    fig = px.bar(
    best_brands,
    x="Price Segment",
    y="Score",
    color="Category",
    text="Brand",
    title="Best Brand by Price Segment and Category",
    labels={"Score": "Average Score", "Price Segment": "Price Range", "Category": "Purpose"},
    hover_data={"Brand": True},
    color_discrete_sequence=px.colors.qualitative.Set2,
    height=400,
    barmode ='group'
)
    fig.update_traces(textposition='outside')
    fig.update_layout(
        font=dict(size=14),
        title_font=dict(size=18),
        legend=dict(title="Category", font_size=12),
        xaxis=dict(title="Price Segment", tickfont_size=12),
        yaxis=dict(title="Average Score", tickfont_size=12),
    )
    st.plotly_chart(fig)
def average_score_by_brand():
    col1,col2=st.columns(2)
    segment = col1.selectbox("Select price segment",labels)
    task = col2.selectbox("Select task",scores)
    filtered_data = grouped[grouped["Price Segment"] == segment][["Brand", task]].sort_values(by = task, ascending = False)
    filtered_data.rename(columns={task: "Score"}, inplace=True)

    fig_task_all = px.bar(
        filtered_data,
        x="Brand",
        y="Score",
        text="Score",
        title=f"Average {task.replace(' Score', '')} Score by Brand in {segment} Price Segment",
        labels={"Score": "Average Score", "Brand": "Brand"},
        color="Brand",  
        height=560,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig_task_all.update_traces(textposition="outside",texttemplate = "%{text:.2f}")
    fig_task_all.update_layout(
        font=dict(size=14),
        title_font=dict(size=18),
        legend=dict(title="Brand", font_size=12),
        xaxis=dict(title="Brand", tickfont_size=12),
        yaxis=dict(title="Average Score", tickfont_size=12)
)

    st.plotly_chart(fig_task_all)
def avg_score_by_product_line():
    dfs = deepcopy(df1)
    col1,col2,col3=st.columns(3)
    brand_list=dfs["Brand"].unique().tolist()
    current_brand=st.selectbox("Select a brand",brand_list)
    segment = col1.selectbox("Select a price segment",labels)
    task = col2.selectbox("Select a task",scores)
    dfx = dfs[dfs['Brand']==current_brand]
    dfx["Brand product line"] = dfx["name"].apply(lambda x : x.split()[1])

    grouped2 = dfx.groupby(['Price Segment', 'Brand product line']).agg({
        'Work Score': 'mean',
        'Play Score': 'mean',
        'Display Score': 'mean',
        'Portability Score': 'mean',
        'Total Score': 'mean'
    }).reset_index()
    grouped2 = grouped2.astype({'Price Segment': 'str', 'Brand product line': 'str'})

    grouped2.fillna(0,inplace=True)

    best_work = grouped2.loc[grouped2.groupby('Price Segment')['Work Score'].idxmax()]
    best_play = grouped2.loc[grouped2.groupby('Price Segment')['Play Score'].idxmax()]
    best_display = grouped2.loc[grouped2.groupby('Price Segment')['Display Score'].idxmax()]
    best_port = grouped2.loc[grouped2.groupby('Price Segment')['Portability Score'].idxmax()]
    best_total = grouped2.loc[grouped2.groupby('Price Segment')['Total Score'].idxmax()]

    best_work['Category'] = 'Work'
    best_work['Score'] = best_work['Work Score']

    best_play['Category'] = 'Gaming'
    best_play['Score'] = best_play['Play Score']

    best_display['Category'] = 'Display'
    best_display['Score'] = best_display['Display Score']

    best_port['Category'] = 'Portability'
    best_port['Score'] = best_port['Portability Score']

    best_total['Category'] = 'General'
    best_total['Score'] = best_total['Total Score']

    best_brands = pd.concat([best_work, best_play, best_display,best_port, best_total]).reset_index(drop=True)

    filtered_data = grouped2[grouped2["Price Segment"] == segment][["Brand product line", task]].sort_values(by = task, ascending = False)
    filtered_data.rename(columns={task: "Score"}, inplace=True)

    fig_task_all = px.bar(
        filtered_data,
        x="Brand product line",
        y="Score",
        text="Score",
        title=f"Average {task.replace(' Score', '')} Score by Product Line in {segment} Price Segment of {current_brand}",
        labels={"Score": "Average Score", "Brand product Line": "Brand product line"},
        color="Brand product line",  
        height=560,
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig_task_all.update_traces(textposition="outside",texttemplate = "%{text:.2f}")
    fig_task_all.update_layout(
        font=dict(size=14),
        title_font=dict(size=18),
        legend=dict(title="Product Line", font_size=12),
        xaxis=dict(title="Product Line", tickfont_size=12),
        yaxis=dict(title="Average Score", tickfont_size=12)
    )

    st.plotly_chart(fig_task_all)
def score_vs_cost_by_year():
    # Load and clean the data
    df4=deepcopy(df)
    df4 = df4[df4["Cost"]<=12000]
    df4 = df4.dropna(subset = ["Cost","Total Score", "GPU: Architecture","CPU: Codename"])
    df4["Combined Architecture"] = df4["CPU: Codename"] + " + " + df4["GPU: Architecture"]
    # Extract the release years for CPU and GPU
    df4["CPU Release Year"] = df4["CPU: Release quarter"].apply(lambda x: int(str(x).split('.')[0]))
    df4["GPU Release Year"] = df4["GPU: Release quarter"].apply(lambda x: int(str(x).split('.')[0]))

    # Use the later release year between CPU and GPU
    df4["Release Year"] = df4[["CPU Release Year", "GPU Release Year"]].max(axis=1)
    # Function to process and plot data for combined architecture with Plotly
    def process_and_plot_combined_plotly(df4, group_column, score_column, title):
        # Group by the specified column
        grouped3 = df4.groupby(group_column).agg(
            Release_Year_Mode=("Release Year", lambda x: x.mode()[0]),  # Mode of release year
            Mean_Cost=("Cost", "mean"),  # Mean cost
            Mean_Score=(score_column, "mean")  # Mean score for the selected type
        ).reset_index()
        # Create interactive scatter plot
        fig = px.scatter(
            grouped3,
            x="Mean_Score", 
            y="Mean_Cost",  
            color="Release_Year_Mode",
            # color_continuous_scale="Viridis",
            color_discrete_sequence=px.colors.qualitative.Set1,
            hover_name=group_column,  # Show the name of the architecture on hover
            labels={"Mean_Cost": "Mean Cost", "Mean_Score": f"Mean {score_column}", "Release_Year_Mode": "Release Year"},
            title=title
        )
        fig.update_layout(
            title=title,
            xaxis_title=f"Mean {score_column}",  # Adjusted axis label
            yaxis_title="Mean Cost",  # Adjusted axis label
            coloraxis_colorbar=dict(title="Release Year"),
            autosize=False,
            width=1120,  # 16:9 width
            height=630   # 16:9 height
        )
        st.plotly_chart(fig)
    score=st.selectbox("Select score",['Total Score',"Work Score","Play Score"])
    process_and_plot_combined_plotly(
        df4,
        "Combined Architecture",
        score,
        f"Combined Architecture: {score} vs Cost by Release Year"
    )

def main():
    
    st.title("Laptop Insights Dashboard")
    #section 1: Average score of each laptop brands
    st.header("1. Average score of each laptop brands in different price segments")
    st.write("This section provides insights into the performance over a specific task of laptop brands acrross different price segments")
    average_score_by_brand()
    #section 2: Average score of product line of a specific brand
    st.header("2.  Average score by product line in different price segments of a brand")
    st.write("This section gives deep insight into product line of specific laptop brand, showing their performance in a given task ")
    avg_score_by_product_line()
    # Section 3: Top laptops with highest performance
    st.header("3. Top laptop brands with highest performance in each category")
    st.write("This section will provide insights into the top-performing laptops based on performance metrics.")
    brand_with_best_quality_over_price_range()
    # Section 4: Laptop brands dominate in each price range
    st.header("4. Laptop brands domination in different segments of score/price")
    st.write("This section will analyze which laptop brands has the largest number of products in different segments of score/price.")
    brands_dominance_over_bins()

    # Section 5: GPU trend
    st.header("5. Scores vs Cost by release year")
    st.write("This section shows the trends in performance improvement across CPU and GPU generations")
    score_vs_cost_by_year()

    st.markdown("---")
    st.write("Powered by Streamlit")