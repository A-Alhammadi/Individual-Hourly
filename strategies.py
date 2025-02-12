#strategies.py

import pandas as pd
import numpy as np
from config import BACKTEST_CONFIG

class TradingStrategies:
    @staticmethod
    def ema_strategy(df, custom_params=None):
        """Adaptive EMA strategy with dynamic position sizing"""
        print("\nGenerating EMA strategy signals...")
        print(f"Input data shape: {df.shape}")
        
        # Verify required columns exist
        required_columns = ['ema_short', 'ema_medium', 'trend_strength', 'position_scale']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Verify data quality
        print("\nVerifying data quality:")
        for col in required_columns:
            print(f"{col} stats:")
            print(f"  Mean: {df[col].mean():.6f}")
            print(f"  Max: {df[col].max():.6f}")
            print(f"  Min: {df[col].min():.6f}")
            print(f"  NaN count: {df[col].isna().sum()}")
        
        signals = pd.Series(0.0, index=df.index)
        config = custom_params if custom_params is not None else BACKTEST_CONFIG['ema']
        
        min_strength = config.get('min_trend_strength', 0.002)  # Updated default
        print(f"Using min_trend_strength: {min_strength}")
        
        # Convert to numpy arrays for faster computation
        ema_short = df['ema_short'].to_numpy()
        ema_medium = df['ema_medium'].to_numpy()
        position_scale = df['position_scale'].to_numpy()
        trend_strength = df['trend_strength'].to_numpy()
        close_prices = df['close_price'].to_numpy()
        
        # Track trade statistics
        crossovers = 0
        filtered_trades = 0
        trend_strengths = []
        trade_prices = []
        
        # Initialize tracking variables
        current_position = 0
        was_above = ema_short[0] > ema_medium[0]
        entry_price = None
        
        for i in range(1, len(df)):
            # Track EMA crossovers
            is_above = ema_short[i] > ema_medium[i]
            
            if is_above != was_above:
                crossovers += 1
                current_strength = trend_strength[i]
                trend_strengths.append(current_strength)
                
                if current_strength < min_strength:
                    filtered_trades += 1
                    print(f"Filtered crossover at {df.index[i]}, trend_strength: {current_strength:.6f} vs min: {min_strength}")
                else:
                    print(f"Valid crossover at {df.index[i]}, trend_strength: {current_strength:.6f}")
                    
                    if current_position == 0:  # No position
                        if is_above:
                            signals.iloc[i] = 1.0 * position_scale[i]
                            current_position = 1
                            entry_price = close_prices[i]
                            trade_prices.append(entry_price)
                            print(f"Long signal at {df.index[i]}, price: {entry_price:.2f}")
                        else:
                            signals.iloc[i] = -1.0 * position_scale[i]
                            current_position = -1
                            entry_price = close_prices[i]
                            trade_prices.append(entry_price)
                            print(f"Short signal at {df.index[i]}, price: {entry_price:.2f}")
                    else:  # In a position
                        if (current_position == 1 and not is_above) or (current_position == -1 and is_above):
                            signals.iloc[i] = -current_position * position_scale[i]
                            exit_price = close_prices[i]
                            trade_prices.append(exit_price)
                            
                            if current_position == 1:
                                print(f"Exit long at {df.index[i]}, P/L: {((exit_price/entry_price)-1)*100:.2f}%")
                            else:
                                print(f"Exit short at {df.index[i]}, P/L: {((entry_price/exit_price)-1)*100:.2f}%")
                            
                            current_position = 0
                            entry_price = None
            else:
                signals.iloc[i] = current_position * position_scale[i]
            
            was_above = is_above
        
        print("\nStrategy Debug Info:")
        print(f"Total crossovers detected: {crossovers}")
        print(f"Trades filtered by strength: {filtered_trades}")
        
        if trend_strengths:
            print("\nTrend Strength Statistics (at crossovers):")
            print(f"Mean: {np.mean(trend_strengths):.6f}")
            print(f"Max: {np.max(trend_strengths):.6f}")
            print(f"Min: {np.min(trend_strengths):.6f}")
            print(f"Median: {np.median(trend_strengths):.6f}")
        
        if trade_prices:
            print("\nTrade Price Statistics:")
            print(f"Mean: {np.mean(trade_prices):.2f}")
            print(f"Max: {np.max(trade_prices):.2f}")
            print(f"Min: {np.min(trade_prices):.2f}")
        
        num_signals = (signals != 0).sum()
        print(f"Generated {num_signals} signals")
        print(f"Output signals shape: {signals.shape}")
        
        return signals


    @classmethod
    def get_all_strategies(cls):
        """Returns only the adaptive EMA strategy"""
        return {
            'Adaptive EMA': cls.ema_strategy
        }