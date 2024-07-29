# src/utils/functions.py
import pandas as pd
import sqlite3

# Read sql and convert to dataframe
def read_sqlite_to_dataframe(sqlite_db_path, query):
    conn = sqlite3.connect(sqlite_db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df