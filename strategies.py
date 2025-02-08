#strategies.py

import json
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import os
from config import BACKTEST_CONFIG
from indicators import TechnicalIndicators

class TradingStrategies:
    @staticmethod
    def ema_strategy(df, custom_params=None):
        """EMA strategy with signal strength filters but flexible exits"""
        signals = pd.Series(index=df.index, data=0)
        config = custom_params if custom_params is not None else BACKTEST_CONFIG['ema']
        
        # Calculate EMAs
        short_ema = df['close_price'].ewm(span=config['short'], adjust=False).mean()
        medium_ema = df['close_price'].ewm(span=config['medium'], adjust=False).mean()
        
        # Calculate trend strength (percentage difference between EMAs)
        trend_strength = abs(short_ema - medium_ema) / medium_ema
        
        # Get minimum strength threshold for entries
        min_strength = config.get('min_trend_strength', 0.002)  # 0.2% default
        
        # Generate signals
        for i in range(len(df)):
            current_strength = trend_strength.iloc[i]
            
            if signals.iloc[i-1] == 0:  # Not in a position
                # Strong trend required for entry
                if short_ema.iloc[i] > medium_ema.iloc[i] and current_strength > min_strength:
                    signals.iloc[i] = 1  # Buy signal
                elif short_ema.iloc[i] < medium_ema.iloc[i] and current_strength > min_strength:
                    signals.iloc[i] = -1  # Sell signal
            else:  # In a position
                # Any crossover can trigger exit
                if signals.iloc[i-1] == 1 and short_ema.iloc[i] < medium_ema.iloc[i]:
                    signals.iloc[i] = -1  # Exit long
                elif signals.iloc[i-1] == -1 and short_ema.iloc[i] > medium_ema.iloc[i]:
                    signals.iloc[i] = 1  # Exit short
        
        return signals

    @classmethod
    def get_all_strategies(cls):
        """Returns only the EMA strategy"""
        return {
            'EMA': cls.ema_strategy
        }