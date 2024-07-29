# src/app.py
import os
import streamlit as st
from utils.functions import read_sqlite_to_dataframe

def main():
    #Change
    sqlite_db_path = os.path.join(os.path.dirname(__file__), '..','..', 'data', 'data.db')
    # Read data 
    query = "SELECT * FROM data limit 10"
    df = read_sqlite_to_dataframe(sqlite_db_path, query)
    
    st.write(df.head())

if __name__ == "__main__":
    main()
