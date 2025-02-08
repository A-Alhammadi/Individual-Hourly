# indicators.py

import pandas as pd
import numpy as np
from config import BACKTEST_CONFIG

class TechnicalIndicators:
    @staticmethod
    def add_ema(df, custom_params=None):
        """Add EMA indicators with either default or custom parameters"""
        df = df.copy()
        config = custom_params if custom_params is not None else BACKTEST_CONFIG['ema']
        
        df[f'ema_{config["short"]}'] = df['close_price'].ewm(span=config['short'], adjust=False).mean()
        df[f'ema_{config["medium"]}'] = df['close_price'].ewm(span=config['medium'], adjust=False).mean()
        
        return df

    @staticmethod
    def add_market_characteristics(df):
        """Add market characteristics focusing on volatility for regime detection"""
        df = df.copy()
        
        # Volatility (20-period standard deviation of returns)
        df['volatility'] = df['close_price'].pct_change().rolling(window=20).std()
        
        # Trend Strength (using configured EMAs)
        if 'ema_9' not in df.columns:
            df = TechnicalIndicators.add_ema(df)
        
        # Basic market characteristics
        df['trend_strength'] = ((df[f'ema_{BACKTEST_CONFIG["ema"]["short"]}'] - 
                                df[f'ema_{BACKTEST_CONFIG["ema"]["medium"]}']) / 
                                df[f'ema_{BACKTEST_CONFIG["ema"]["medium"]}'].abs()).fillna(0)
        
        return df

    @classmethod
    def add_all_indicators(cls, df, custom_params=None):
        """
        Add EMA indicators and market characteristics
        
        Args:
            df (pd.DataFrame): Input DataFrame
            custom_params (dict, optional): Custom parameters for indicators
        """
        print("\nAdding indicators...")
        df = df.copy()
        
        try:
            # Add EMAs
            params = custom_params.get('ema') if custom_params else None
            df = cls.add_ema(df, params)
            
            # Add market characteristics for regime detection
            df = cls.add_market_characteristics(df)
            
        except Exception as e:
            print(f"Error adding indicators: {str(e)}")
            raise
        
        # Drop NaN values created by indicators
        df.dropna(inplace=True)
        
        return df