# src/app.py
import os
import streamlit as st
from utils.functions import *


def main():
    st.title("Digital Marketing Customer Clasification 📊")


    ##### Information #####

    with st.expander("Information 📖"):
        st.header("About the App.", divider= "gray")
        st.subheader("Scope", divider= "red")
        st.write(information_text("Scope"))
        st.subheader("How to use the app", divider= "blue")
        st.write(information_text("How to use the app"))


    ##### Dataset Selection #####

    dataset_selector = st.radio(
    "Select your dataset",
    ["Use a sample dataset", "Select your own dataset"],
    horizontal= True
    )

    if dataset_selector == "Use a sample dataset":
        file_path = os.path.join(os.path.dirname(__file__), 'data', 'test_dataset.csv')
        df = read_dataset(file_path)
    else:
        uploaded_dataset = st.file_uploader("Choose a CSV file")
        if uploaded_dataset is not None:
            df = read_dataset(uploaded_dataset)


    ##### Model #####
    if dataset_selector == "Use a sample dataset" or uploaded_dataset is not None:
        df_result = model(df)


    ##### Sidebar Building #####

    with st.sidebar:
        st.header("Menu")
        st.page_link("app.py", label="Home", icon="🏠")

        if dataset_selector == "Use a sample dataset" or uploaded_dataset is not None:
            filter_customer_selector = st.radio(
                "Filter Customers",
                ["Uplift Score", "From chart"],
                horizontal= True
            )
            if filter_customer_selector == "Uplift Score":
                uplift_score = st.number_input(
                    "Insert a number", value=0.1, placeholder="Type Min Uplift Score"
                )
                x_score = 0.4
                y_score = 0
            else:
                ymin = round(float(df_result['y'].min()),2)
                ymax = round(float(df_result['y'].max()),2)
                xmin = round(float(df_result['x'].min()),2)
                xmax = round(float(df_result['x'].max()),2)
                yavg = round(float(df_result['y'].mean()),2)
                xavg = round(float(df_result['x'].mean()),2)
                x_score = st.slider("Select minimum X score", xmin, xmax, xavg)
                y_score = st.slider("Select minimum Y score", ymin, ymax, yavg)

            
            st.write("Profit Analysis")
            benefit_score = st.number_input(
                "Benefit of each conversion", value=10, placeholder="US Dollar ($)"
            )
            cost_score = st.number_input(
                "Cost of each treatment", value=1, placeholder="US Dollar ($)"
            )


    ##### Display Dataset ##### 

    if dataset_selector == "Use a sample dataset" or uploaded_dataset is not None: 
        with st.expander("Show Dataset"):
            st.write(df.head())


    ##### Filter Customers #####

        df_result['Profit'] = (benefit_score * df_result['p_tr']) - (cost_score * (df_result['p_tn'] + df_result['p_cr'])) - ((benefit_score + cost_score) * df_result['p_cn'])

        if filter_customer_selector == "Uplift Score":
            df_filtered = df_result[(df_result['uplift_score'] >= uplift_score)]
        else:
            df_filtered = df_result[(df_result['x'] > x_score) & (df_result['y'] > y_score)]
        
        
        #df_filtered['Profit'] = (benefit_score * df_filtered['p_tr']) - (cost_score * (df_filtered['p_tn'] + df_filtered['p_cr'])) - ((benefit_score + cost_score) * df_filtered['p_cn'])

    ##### Plot Results #####

        tab1, tab2, tab3 = st.tabs(["Customer Segmentation", "Uplifting Model", "Download Data"])

        with tab1:
            display_profit_analysis(df_filtered, df_result)
            st.plotly_chart(draw_scatter_plot(df_result,x_score,y_score))
        with tab2:
            display_profit_analysis(df_filtered, df_result)
            st.plotly_chart(plot_histogram(df_filtered))
        with tab3:
            display_profit_analysis(df_filtered, df_result)
            st.write(df_filtered)
            download_dataset(df)
            

if __name__ == "__main__":
    main()
