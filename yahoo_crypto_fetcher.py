# yahoo_crypto_fetcher.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import psycopg2
import time

class CryptoDataFetcher:
    def __init__(self):
        self.conn = None
        self.connect_db()
        
        # List of cryptocurrencies to fetch
        self.cryptos = [
            'XRP-USD', 'AAVE-USD', 'ADA-USD', 'ALGO-USD', 
            'BTC-USD', 'BCH-USD', 'ETH-USD', 'LTC-USD', 'UNI-USD'
        ]
        
    def connect_db(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host="localhost",
                port="5432",
                dbname="cryptocurrencies",
                user="myuser",
                password="mypassword"
            )
            print("Successfully connected to database")
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            raise

    def get_first_and_last_date(self, symbol):
        """Get the first and last date in database for a given symbol"""
        try:
            cursor = self.conn.cursor()
            query = """
                SELECT MIN(date_time), MAX(date_time)
                FROM crypto_data_hourly 
                WHERE symbol = %s
            """
            cursor.execute(query, (symbol.replace('-', '/'),))
            result = cursor.fetchone()
            cursor.close()
            return result
        except Exception as e:
            print(f"Error getting dates for {symbol}: {str(e)}")
            return None, None

    def find_missing_periods(self, df, expected_interval_hours=1):
        """Find gaps in the data"""
        df = df.sort_index()
        time_diffs = df.index.to_series().diff()
        gaps = time_diffs[time_diffs > pd.Timedelta(hours=expected_interval_hours)]
        return gaps

    def fetch_and_process_data(self, symbol, start_date=None, end_date=None):
        """Fetch and process data for a single cryptocurrency"""
        try:
            print(f"\nProcessing {symbol}")
            
            # Convert dates to datetime if they're strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            # Initialize ticker
            ticker = yf.Ticker(symbol)
            
            # Fetch hourly data
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d') if end_date else None,
                interval='1h'
            )
            
            if df.empty:
                print(f"No data found for {symbol}")
                return
            
            # Check for missing data
            gaps = self.find_missing_periods(df)
            if not gaps.empty:
                print("\nFound gaps in data:")
                for timestamp, gap in gaps.items():
                    print(f"Gap of {gap} at {timestamp}")
            
            # Reset index to make datetime a column and ensure timezone-naive
            df = df.reset_index()
            df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None)
            
            # Add Unix timestamp
            df['unix_timestamp'] = df['Datetime'].astype('int64') // 10**9
            
            # Handle missing volumes
            df['Volume'] = df['Volume'].fillna(0)
            
            # Rename columns
            df = df.rename(columns={
                'Datetime': 'date_time',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume_crypto'
            })
            
            # Calculate volume in USD
            df['volume_usd'] = df['volume_crypto'] * df['close_price']
            
            # Scale down large numbers
            scale_threshold = 1e11
            
            def scale_value(val):
                try:
                    if abs(float(val)) >= scale_threshold:
                        return float(val) / 1e3
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0
            
            numeric_columns = ['open_price', 'high_price', 'low_price', 
                             'close_price', 'volume_crypto', 'volume_usd']
            
            for col in numeric_columns:
                df[col] = df[col].apply(scale_value)
            
            # Convert symbol format
            db_symbol = symbol.replace('-', '/')
            
            print(f"Retrieved {len(df)} records from {df['date_time'].min()} to {df['date_time'].max()}")
            
            # Print summary statistics
            print("\nData Summary:")
            print(f"Average Volume: {df['volume_crypto'].mean():.2f}")
            print(f"Zero Volume Records: {(df['volume_crypto'] == 0).sum()}")
            
            cursor = self.conn.cursor()
            rows_processed = 0
            
            for _, row in df.iterrows():
                # Check if we already have this record
                check_query = """
                    SELECT 1 FROM crypto_data_hourly 
                    WHERE date_time = %s AND symbol = %s
                """
                cursor.execute(check_query, (row['date_time'], db_symbol))
                exists = cursor.fetchone() is not None
                
                if exists:
                    # Update existing record
                    update_query = """
                        UPDATE crypto_data_hourly 
                        SET open_price = %s, 
                            high_price = %s, 
                            low_price = %s, 
                            close_price = %s, 
                            volume_crypto = %s, 
                            volume_usd = %s,
                            unix_timestamp = %s
                        WHERE date_time = %s AND symbol = %s
                    """
                    values = (
                        float(row['open_price']), float(row['high_price']),
                        float(row['low_price']), float(row['close_price']),
                        float(row['volume_crypto']), float(row['volume_usd']),
                        int(row['unix_timestamp']),
                        row['date_time'], db_symbol
                    )
                    cursor.execute(update_query, values)
                else:
                    # Insert new record
                    insert_query = """
                        INSERT INTO crypto_data_hourly 
                        (date_time, symbol, open_price, high_price, low_price, 
                         close_price, volume_crypto, volume_usd, unix_timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    values = (
                        row['date_time'], db_symbol,
                        float(row['open_price']), float(row['high_price']),
                        float(row['low_price']), float(row['close_price']),
                        float(row['volume_crypto']), float(row['volume_usd']),
                        int(row['unix_timestamp'])
                    )
                    cursor.execute(insert_query, values)
                
                rows_processed += 1
            
            self.conn.commit()
            cursor.close()
            print(f"Successfully processed {rows_processed} records for {symbol}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            if 'cursor' in locals():
                cursor.close()
            self.conn.rollback()
            raise

    def update_all_cryptos(self):
        """Update data for all cryptocurrencies"""
        for symbol in self.cryptos:
            try:
                # Get first and last dates in database
                first_date, last_date = self.get_first_and_last_date(symbol)
                
                if last_date:
                    # Add one hour to avoid duplicates
                    start_date = last_date + timedelta(hours=1)
                    print(f"Fetching {symbol} new data from {start_date}")
                    
                    if start_date < datetime.now():
                        self.fetch_and_process_data(symbol, start_date)
                    else:
                        print(f"Data for {symbol} is up to date")
                    
                    # Check for gaps in historical data
                    if first_date:
                        earliest_desired = datetime.now() - timedelta(days=365)
                        if first_date > earliest_desired:
                            print(f"Fetching historical data from {earliest_desired} to {first_date}")
                            self.fetch_and_process_data(symbol, earliest_desired, first_date)
                else:
                    # If no existing data, fetch last 365 days
                    start_date = datetime.now() - timedelta(days=365)
                    print(f"No existing data for {symbol}, fetching from {start_date}")
                    self.fetch_and_process_data(symbol, start_date)
                
                # Sleep briefly to avoid hitting rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error updating {symbol}: {str(e)}")
                continue

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("\nDatabase connection closed")

def main():
    fetcher = None
    try:
        fetcher = CryptoDataFetcher()
        fetcher.update_all_cryptos()
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        
    finally:
        if fetcher:
            fetcher.close()

if __name__ == "__main__":
    main()