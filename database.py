#database.py

import psycopg2
import pandas as pd
from config import DB_CONFIG

class DatabaseHandler:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            print("Successfully connected to database")
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            raise

    def get_historical_data(self, symbol, start_date, end_date):
        query = """
            SELECT 
                date_time,
                open_price,
                high_price,
                low_price,
                close_price,
                volume_crypto,
                volume_usd
            FROM crypto_data_hourly
            WHERE symbol = %s
            AND date_time BETWEEN %s AND %s
            ORDER BY date_time ASC
        """
        
        try:
            df = pd.read_sql_query(
                query,
                self.conn,
                params=(symbol, start_date, end_date),
                parse_dates=['date_time']
            )
            
            # Set date_time as index
            df.set_index('date_time', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed")