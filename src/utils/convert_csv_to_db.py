# convert_csv_to_db.py
import os
import pandas as pd
import sqlite3

def csv_to_sqlite(csv_file_path, sqlite_db_path):
    df = pd.read_csv(csv_file_path)
    conn = sqlite3.connect(sqlite_db_path)
    df.to_sql('data', conn, if_exists='replace', index=False)
    conn.close()
    print(f"CSV file {csv_file_path} converted to SQLite database {sqlite_db_path} successfully.")

if __name__ == "__main__":
    csv_file_path = os.path.join('data', 'criteo-uplift-v2.1.csv')
    sqlite_db_path = os.path.join('data', 'data.db')
    csv_to_sqlite(csv_file_path, sqlite_db_path)
