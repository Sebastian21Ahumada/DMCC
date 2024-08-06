# src/utils/functions.py
import pandas as pd
import sqlite3
import pickle
import os
from sklearn.decomposition import PCA
import plotly.express as px
import streamlit as st

# Read sql and convert to dataframe
def read_sqlite_to_dataframe(sqlite_db_path, query):
    conn = sqlite3.connect(sqlite_db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

# Read Sample dataset
def read_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

## Load and use classification model
def model(df):
    model_path = os.path.join(os.path.dirname(__file__),  'xgboost_model.pkl')
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)


    uplift_proba = loaded_model.predict_proba(df)
    predictions = loaded_model.predict(df)

    result = df.copy()

    result['p_cn'] = uplift_proba[:, 0]
    result['p_cr'] = uplift_proba[:, 1]
    result['p_tn'] = uplift_proba[:, 2]
    result['p_tr'] = uplift_proba[:, 3]

    result['uplift_score'] = result.eval('\
        p_cn/(p_cn + p_cr) \
        + p_tr/(p_tn + p_tr) \
        - p_tn/(p_tn + p_tr) \
        - p_cr/(p_cn + p_cr)')

    result['predictions'] = predictions


    columns_of_interest = ['p_cn', 'p_tr', 'p_cr', 'p_tn']
    data_to_reduce = result[columns_of_interest]
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data_to_reduce)
    reduced_df = pd.DataFrame(reduced_data, columns=['x', 'y'])
    result[['x', 'y']] = reduced_df

    mapping = {
    0: "Do not disturb",
    1: "Sure thing",
    2: "Lost Causes",
    3: "Persuadables"
    }

    result['type'] = result['predictions'].map(mapping)

    return result


### function to draw the scatter plot with the model results
def draw_scatter_plot(df, x, y):

    color_map = {
        "Do not disturb": "red",
        "Sure thing": "blue",
        "Lost Causes": "yellow",
        "Persuadables": "green"
    }

    fig = px.scatter(df, x='x', y='y', color = 'type',title='Visualization of customers types',
                    color_discrete_map=color_map)

    x_min = df['x'].min()
    x_max = df['x'].max()
    y_min = df['y'].min()
    y_max = df['y'].max()

    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=y_min,  # Extiende la línea a lo largo de todo el eje y
        y1=y_max,
        line=dict(color="Black", width=2, dash="dash")
    )

    fig.add_shape(
        type="line",
        x0=x_min,
        x1=x_max,
        y0=y,  # Extiende la línea a lo largo de todo el eje y
        y1=y,
        line=dict(color="Black", width=2, dash="dash")
    )

    return fig

def plot_histogram(df):
    fig_histogram = px.histogram(
        df, 
        x='uplift_score', 
        title='Histograma de uplift_score',
        labels={'uplift_score': 'Uplift Score'}  
    )
    return fig_histogram



### Profit Analysis
def display_profit_analysis(df_filtered, df_result):
    st.subheader("Profit Analysis")
    st.write("How much am I earning based on the filtered customers compared to the total customers in the dataset")

    ## Metrics
    profit_score = round(float(df_filtered['Profit'].sum()), 2)
    profit_score0 = round(float(df_result['Profit'].sum()), 2)
    profit_score_delta = profit_score - profit_score0

    total_customer = int(df_filtered['Profit'].count())
    total_customer0 = int(df_result['Profit'].count())
    total_customer_delta = total_customer - total_customer0

    total_customer_do_not_disturb = int(df_filtered[df_filtered['type'] == "Do not disturb"]['Profit'].count())
    total_customer_do_not_disturb0 = int(df_result[df_result['type'] == "Do not disturb"]['Profit'].count())
    total_customer_do_not_disturb_delta = total_customer_do_not_disturb - total_customer_do_not_disturb0

    total_customer_lost_causes = int(df_filtered[df_filtered['type'] == "Lost Causes"]['Profit'].count())
    total_customer_lost_causes0 = int(df_result[df_result['type'] == "Lost Causes"]['Profit'].count())
    total_customer_lost_causes_delta = total_customer_lost_causes - total_customer_lost_causes0

    total_customer_persuadables = int(df_filtered[df_filtered['type'] == "Persuadables"]['Profit'].count())
    total_customer_persuadables0 = int(df_result[df_result['type'] == "Persuadables"]['Profit'].count())
    total_customer_persuadables_delta = total_customer_persuadables - total_customer_persuadables0

    total_customer_sure_thing = int(df_filtered[df_filtered['type'] == "Sure thing"]['Profit'].count())
    total_customer_sure_thing0 = int(df_result[df_result['type'] == "Sure thing"]['Profit'].count())
    total_customer_sure_thing_delta = total_customer_sure_thing - total_customer_sure_thing0

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric(label="Profit", value="$" + str(profit_score), delta="$" + str(profit_score_delta))
    col2.metric(label="Total Customers", value=total_customer, delta=total_customer_delta, delta_color="off")
    col3.metric(label="Do not disturb", value=total_customer_do_not_disturb, delta=total_customer_do_not_disturb_delta, delta_color="inverse")
    col4.metric(label="Lost Causes", value=total_customer_lost_causes, delta=total_customer_lost_causes_delta, delta_color="inverse")
    col5.metric(label="Persuadable", value=total_customer_persuadables, delta=total_customer_persuadables_delta)
    col6.metric(label="Sure thing", value=total_customer_sure_thing, delta=total_customer_sure_thing_delta, delta_color="inverse")


def convert_df(df):
    return df.to_csv().encode("utf-8")

def download_dataset(df):
    csv = convert_df(df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="large_df.csv",
        mime="text/csv",
    )

def information_text(text):
    if text == 'Scope':
        display_text = """The objective of the web application is to classify potential customers obtained from a 
                 marketing campaign database. This will allow us to identify, based on certain characteristics. The app
                 used a modeling technique called uplift, which is an important yet novel area of research in machine learning 
                 which aims to explain and to estimate the causal impact of a treatment at the individual level.
                 """
    elif text == 'How to use the app':
        display_text = """ 
        The general use of the application is as follows:

        **1.- Load a dataset:** The app has a pre-loaded test dataset that can be used, but it also offers the possibility to load a custom dataset. The pre-loaded dataset consists of 12 columns, ranging from f1 to f12, representing the features to be used for customer classification. In real data, the features may correspond to such things as customer purchase history, demographics, and other quantities a data scientist may engineer with the hypothesis that they would be useful in modeling uplift.
        
        **2.- Set the desired parameters:** Once the dataset is loaded, menu bar parameters appear that can be manipulated by the user. The idea is to test different combinations to filter the final customers according to the model's results. Additionally, a profit analysis is provided based on the user-defined revenue and costs.
        
        **3.- UI Area:** As the user changes the parameters in the sidebar menu, they can see how it affects the customers in the UI area, located in the center of the app. 
        There are three tabs available. The **Customer Segmentation** tab display a scatter plot shows the distribution of different customers based on the model's predictions and which group they belong to. 
        The **Uplifting Model** display the histogram with the uplift modeling results, and finally, the **Download data** tab show the dataset with the final list of filtered customers to be used in the marketing campaign, based on the parameters selected by the user. """

    return display_text