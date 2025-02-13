#strategies.py

import pandas as pd
import numpy as np
from config import BACKTEST_CONFIG

class TradingStrategies:
    @staticmethod
    def ema_strategy(df, custom_params=None):
        """Adaptive EMA strategy with RSI confirmation"""
        print("\nGenerating EMA-RSI strategy signals...")
        print(f"Input data shape: {df.shape}")
        
        # Verify required columns exist
        required_columns = ['ema_short', 'ema_medium', 'trend_strength', 'position_scale', 'rsi']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        signals = pd.Series(0.0, index=df.index)
        config = custom_params if custom_params is not None else BACKTEST_CONFIG
        
        min_strength = config['ema'].get('min_trend_strength', 0.002)
        rsi_config = config.get('rsi', BACKTEST_CONFIG['rsi'])
        rsi_weight = rsi_config.get('weight', 0.5)
        ema_weight = 1 - rsi_weight
        
        # Convert to numpy arrays for faster computation
        ema_short = df['ema_short'].to_numpy()
        ema_medium = df['ema_medium'].to_numpy()
        position_scale = df['position_scale'].to_numpy()
        trend_strength = df['trend_strength'].to_numpy()
        close_prices = df['close_price'].to_numpy()
        rsi_values = df['rsi'].to_numpy()
        
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
            
            # Get RSI signals
            rsi_value = rsi_values[i]
            rsi_bullish = rsi_value < rsi_config['oversold']
            rsi_bearish = rsi_value > rsi_config['overbought']
            
            if is_above != was_above:
                crossovers += 1
                current_strength = trend_strength[i]
                trend_strengths.append(current_strength)
                
                if current_strength < min_strength:
                    filtered_trades += 1
                    print(f"Filtered crossover at {df.index[i]}, trend_strength: {current_strength:.6f}")
                else:
                    # Calculate combined signal strength
                    ema_signal = 1 if is_above else -1
                    rsi_signal = 1 if rsi_bullish else (-1 if rsi_bearish else 0)
                    combined_signal = (ema_signal * ema_weight) + (rsi_signal * rsi_weight)
                    
                    # Only trade if EMA and RSI agree (combined signal > threshold)
                    signal_threshold = rsi_config.get('signal_threshold', 0.3)  # Get from config with fallback
                    
                    if abs(combined_signal) > signal_threshold:
                        if current_position == 0:  # No position
                            if combined_signal > 0:
                                signals.iloc[i] = 1.0 * position_scale[i]
                                current_position = 1
                                entry_price = close_prices[i]
                                trade_prices.append(entry_price)
                                print(f"Long signal at {df.index[i]}, price: {entry_price:.2f}, RSI: {rsi_value:.1f}")
                            elif combined_signal < 0:
                                signals.iloc[i] = -1.0 * position_scale[i]
                                current_position = -1
                                entry_price = close_prices[i]
                                trade_prices.append(entry_price)
                                print(f"Short signal at {df.index[i]}, price: {entry_price:.2f}, RSI: {rsi_value:.1f}")
                        else:  # In a position
                            if (current_position == 1 and combined_signal < 0) or (current_position == -1 and combined_signal > 0):
                                signals.iloc[i] = -current_position * position_scale[i]
                                exit_price = close_prices[i]
                                trade_prices.append(exit_price)
                                
                                if current_position == 1:
                                    print(f"Exit long at {df.index[i]}, P/L: {((exit_price/entry_price)-1)*100:.2f}%, RSI: {rsi_value:.1f}")
                                else:
                                    print(f"Exit short at {df.index[i]}, P/L: {((entry_price/exit_price)-1)*100:.2f}%, RSI: {rsi_value:.1f}")
                                
                                current_position = 0
                                entry_price = None
            else:
                signals.iloc[i] = current_position * position_scale[i]
            
            was_above = is_above
        
        return signals


    @classmethod
    def get_all_strategies(cls):
        """Returns only the adaptive EMA strategy"""
        return {
            'Adaptive EMA': cls.ema_strategy
        }