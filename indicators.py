# indicators.py

import pandas as pd
import numpy as np
from config import BACKTEST_CONFIG

class TechnicalIndicators:
    @staticmethod
    def calculate_parkinson_volatility(df, window=720):  # 30 days * 24 hours
        """Calculate Parkinson volatility estimator using high/low prices"""
        print("\nCalculating Parkinson volatility...")
        print(f"Input data shape: {df.shape}")
        print(f"Window size: {window} hours ({window/24:.1f} days)")
        
        # Calculate log high/low ratio
        high_prices = df['high_price']
        low_prices = df['low_price']
        log_hl_ratio = np.log(high_prices / low_prices)
        
        # Print diagnostic info for high/low prices
        if hasattr(df, 'is_optimization'):
            hl_ratio = high_prices / low_prices
            print("\nHigh/Low Price Diagnostics:")
            print(f"H/L Ratio - Mean: {hl_ratio.mean():.4f}, Max: {hl_ratio.max():.4f}, Min: {hl_ratio.min():.4f}")
            print(f"Log H/L Ratio - Mean: {log_hl_ratio.mean():.4f}, Max: {log_hl_ratio.max():.4f}, Min: {log_hl_ratio.min():.4f}")
        
        # Calculate Parkinson volatility
        parkinson_factor = 1.0 / (4.0 * np.log(2.0))
        squared_ratios = log_hl_ratio.pow(2)
        
        # Calculate intermediate raw volatility for diagnostics
        raw_volatility = np.sqrt(parkinson_factor * squared_ratios)
        if hasattr(df, 'is_optimization'):
            print("\nRaw Volatility (before rolling):")
            print(f"Mean: {raw_volatility.mean():.4f}")
            print(f"Max: {raw_volatility.max():.4f}")
            print(f"Min: {raw_volatility.min():.4f}")
            print(f"Std: {raw_volatility.std():.4f}")
        
        # Calculate rolling variance with minimum periods
        rolling_variance = squared_ratios.rolling(
            window=window,
            min_periods=window//2  # Need at least half the window
        ).mean()
        
        # Calculate annualized volatility (using hours)
        hours_per_year = 365 * 24
        #volatility = np.sqrt(parkinson_factor * rolling_variance) * np.sqrt(hours_per_year)
        volatility = np.sqrt(parkinson_factor * rolling_variance)

        if hasattr(df, 'is_optimization'):
            print("\nRolling Annualized Volatility:")
            print(f"Mean: {volatility.mean():.4f}")
            print(f"Max: {volatility.max():.4f}")
            print(f"Min: {volatility.min():.4f}")
            print(f"Std: {volatility.std():.4f}")
            if volatility.mean() < 0.1:
                print("\nWARNING: Mean volatility appears very low for crypto!")
            elif volatility.mean() > 2.0:
                print("\nWARNING: Mean volatility appears very high!")
            if np.isnan(volatility).any():
                print("\nWARNING: NaN values detected in volatility calculation!")
                print(f"NaN count: {np.isnan(volatility).sum()}")
        
        # Fill any remaining NaN values with forward fill then backward fill
        volatility = volatility.ffill().bfill()
        
        return volatility

    @staticmethod
    def calculate_volatility_stats(volatility_series, window=720): #window was 720
        """
        Calculate rolling median and MAD of volatility using vectorized operations.
        Reduced window from 720 -> 240, so we adapt quicker.
        Fill in missing values after computing to avoid big NaN blocks.
        """
        import numpy as np

        values = volatility_series.to_numpy()
        # Instead of window//2, consider something smaller:
        # min_periods = max(window // 4, 10)  # e.g. at least 10 bars
        min_periods = window // 3
        n = len(values)

        rolling_median = np.full(n, np.nan)
        rolling_mad = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_vals = values[i - window + 1 : i + 1]
            if len(window_vals) >= min_periods:
                med = np.nanmedian(window_vals)
                rolling_median[i] = med
                rolling_mad[i] = np.nanmedian(np.abs(window_vals - med))

            if i % 1000 == 0:
                print(f"Processing volatility stats: {i}/{n} ({(i/n)*100:.1f}%)")

        # Convert back to pandas Series
        rolling_median = pd.Series(rolling_median, index=volatility_series.index)
        rolling_mad = pd.Series(rolling_mad, index=volatility_series.index)

        # Fill forward/back to remove initial NaNs
        rolling_median = rolling_median.ffill().bfill()
        rolling_mad = rolling_mad.ffill().bfill()

        return rolling_median, rolling_mad


    @staticmethod
    def calculate_zscore(value, median, mad):
        """Calculate Z-score using (value - median) / MAD"""
        return (value - median) / (mad + 1e-10)

    @staticmethod
    def sigmoid_adjustment(zscore):
        """Map Z-score to adjustment factor between -1 and 1"""
        return 2.0 / (1.0 + np.exp(-zscore)) - 1.0

    @staticmethod
    def adjust_ema_period(base_period, adjustment_factor):
        """Adjust EMA period based on volatility, with bounds"""
        multiplier = 2.0 ** adjustment_factor
        adjusted = base_period * multiplier
        return int(np.clip(adjusted, 0.5 * base_period, 2.0 * base_period))

    @staticmethod
    def calculate_effective_dead_zone(recent_changes, lookback_hours=2400):
        """Calculate adaptive dead zone based on recent parameter changes"""
        base_dead_zone = 0.5
        change_ratio = min(recent_changes / lookback_hours, 1.0)
        return base_dead_zone * (1.0 + change_ratio)

    @staticmethod
    def calculate_position_scale(volatility_ratio, base_size=1.0):
        """Calculate position size scalar based on volatility"""
        inverse_ratio = 1.0 / (volatility_ratio + 1e-10)
        return np.clip(inverse_ratio * base_size, 0.5, 2.0)

    @staticmethod
    def add_ema(df, custom_params=None):
        """Add EMA indicators with either default or custom parameters"""
        df = df.copy()
        config = custom_params if custom_params is not None else BACKTEST_CONFIG['ema']
        
        df[f'ema_{config["short"]}'] = df['close_price'].ewm(
            span=config['short'],
            adjust=False
        ).mean()
        df[f'ema_{config["medium"]}'] = df['close_price'].ewm(
            span=config['medium'],
            adjust=False
        ).mean()
        
        return df

    @staticmethod
    def add_market_characteristics(df):
        """Add market characteristics focusing on volatility for regime detection"""
        df = df.copy()
        
        df['volatility'] = TechnicalIndicators.calculate_parkinson_volatility(df)
        
        if 'ema_short' not in df.columns:
            df = TechnicalIndicators.add_adaptive_ema(df)
        
        # Use the centralized calculation
        df['trend_strength'] = TechnicalIndicators.calculate_trend_strength(df)
        
        return df

    @staticmethod
    def add_adaptive_ema(df, lookback_hours=2400):
        """Add adaptive EMA indicators with volatility-based adjustments"""
        print("\nAdding adaptive EMAs...")
        df = df.copy()
        
        volatility = TechnicalIndicators.calculate_parkinson_volatility(df)
        df['volatility'] = volatility
        
        rolling_median, rolling_mad = TechnicalIndicators.calculate_volatility_stats(volatility)
        
        zscores = TechnicalIndicators.calculate_zscore(volatility, rolling_median, rolling_mad)
        
        config = BACKTEST_CONFIG['ema']
        base_short = float(config['short'])
        base_medium = float(config['medium'])
        
        ema_short = np.zeros(len(df))
        ema_medium = np.zeros(len(df))
        adjusted_short_period = np.full(len(df), base_short)
        adjusted_medium_period = np.full(len(df), base_medium)
        volatility_ratio = np.ones(len(df))
        position_scale = np.ones(len(df))
        
        ema_short[0] = df['close_price'].iloc[0]
        ema_medium[0] = df['close_price'].iloc[0]
        
        parameter_changes = pd.Series(0, index=df.index)
        recent_changes = []
        
        for i in range(1, len(df)):
            recent_changes = [
                t for t in recent_changes
                if t > df.index[i] - pd.Timedelta(hours=lookback_hours)
            ]
            dead_zone = 0.5 * (1 + min(len(recent_changes)/lookback_hours, 1.0))
            current_zscore = zscores.iloc[i]
            
            if abs(current_zscore.item()) > dead_zone:
                safe_zscore = np.clip(current_zscore.item(), -50, 50)  # Prevents extreme values
                adjustment = 2.0 / (1.0 + np.exp(-safe_zscore)) - 1.0
                new_short = float(base_short * (2.0 ** adjustment))
                new_medium = float(base_medium * (2.0 ** adjustment))
                new_short = max(base_short/2, min(base_short*2, new_short))
                new_medium = max(base_medium/2, min(base_medium*2, new_medium))
                
                if (abs(new_short - adjusted_short_period[i-1]) > 0.1 * base_short or
                    abs(new_medium - adjusted_medium_period[i-1]) > 0.1 * base_medium):
                    parameter_changes.iloc[i] = 1
                    recent_changes.append(df.index[i])
                
                adjusted_short_period[i] = new_short
                adjusted_medium_period[i] = new_medium
            else:
                adjusted_short_period[i] = adjusted_short_period[i-1]
                adjusted_medium_period[i] = adjusted_medium_period[i-1]
            
            alpha_short = 2.0 / (adjusted_short_period[i] + 1.0)
            alpha_medium = 2.0 / (adjusted_medium_period[i] + 1.0)
            
            ema_short[i] = df['close_price'].iloc[i] * alpha_short + ema_short[i-1] * (1.0 - alpha_short)
            ema_medium[i] = df['close_price'].iloc[i] * alpha_medium + ema_medium[i-1] * (1.0 - alpha_medium)
            
            vol_ratio = volatility.iloc[i] / (rolling_median.iloc[i] + 1e-10)
            volatility_ratio[i] = vol_ratio
            position_scale[i] = min(2.0, max(0.5, 1.0 / vol_ratio))
        
        df['ema_short'] = ema_short
        df['ema_medium'] = ema_medium
        df['adjusted_short_period'] = adjusted_short_period
        df['adjusted_medium_period'] = adjusted_medium_period
        df['volatility_ratio'] = volatility_ratio
        df['position_scale'] = position_scale
        
        print("\nEMA Statistics:")
        print("Short EMA:")
        print(f"Mean: {df['ema_short'].mean():.2f}")
        print(f"Std: {df['ema_short'].std():.2f}")
        print("Medium EMA:")
        print(f"Mean: {df['ema_medium'].mean():.2f}")
        print(f"Std: {df['ema_medium'].std():.2f}")
        
        print(f"\nCompleted adding EMAs. Data shape: {df.shape}")
        return df
    
    @staticmethod
    def diagnose_volatility_calculations(df, window=720):
        """Diagnose issues in volatility calculations"""
        print("\nDiagnosing Volatility Calculations:")
        
        print("\n1. High/Low Price Analysis:")
        hl_ratio = df['high_price'] / df['low_price']
        print(f"High/Low ratio stats:")
        print(f"Mean: {hl_ratio.mean():.4f}")
        print(f"Max: {hl_ratio.max():.4f}")
        print(f"Min: {hl_ratio.min():.4f}")
        print(f"Std: {hl_ratio.std():.4f}")
        
        high_prices = df['high_price']
        low_prices = df['low_price']
        log_hl_ratio = np.log(high_prices / low_prices)
        parkinson_factor = 1.0 / (4.0 * np.log(2.0))
        squared_ratios = log_hl_ratio.pow(2)
        
        raw_volatility = np.sqrt(parkinson_factor * squared_ratios)
        print("\n2. Raw Parkinson Volatility (before rolling):")
        print(f"Mean: {raw_volatility.mean():.4f}")
        print(f"Max: {raw_volatility.max():.4f}")
        print(f"Min: {raw_volatility.min():.4f}")
        print(f"Std: {raw_volatility.std():.4f}")
        
        rolling_variance = squared_ratios.rolling(
            window=window,
            min_periods=window//2
        ).mean()
        
        hours_per_year = 365 * 24
        rolling_vol = np.sqrt(parkinson_factor * rolling_variance) * np.sqrt(hours_per_year)
        
        print(f"\n3. Rolling Window Volatility ({window} hours):")
        print(f"Mean: {rolling_vol.mean():.4f}")
        print(f"Max: {rolling_vol.max():.4f}")
        print(f"Min: {rolling_vol.min():.4f}")
        print(f"Std: {rolling_vol.std():.4f}")
        
        return raw_volatility, rolling_vol

    @staticmethod
    def diagnose_mad_calculations(rolling_vol, window=720):
        """Diagnose issues in MAD calculations"""
        print("\n4. Analyzing MAD Calculations:")
        
        rolling_median = pd.Series(index=rolling_vol.index, dtype=float)
        rolling_mad = pd.Series(index=rolling_vol.index, dtype=float)
        
        for i in range(window - 1, len(rolling_vol)):
            window_vals = rolling_vol[max(0, i - window + 1):i + 1]
            med = np.nanmedian(window_vals)
            rolling_median.iloc[i] = med
            rolling_mad.iloc[i] = np.nanmedian(np.abs(window_vals - med))
        
        print("\nMedian values:")
        print(f"Mean: {rolling_median.dropna().mean():.4f}")
        print(f"Max: {rolling_median.dropna().max():.4f}")
        print(f"Min: {rolling_median.dropna().min():.4f}")
        print(f"Std: {rolling_median.dropna().std():.4f}")
        
        print("\nMAD values:")
        print(f"Mean: {rolling_mad.dropna().mean():.4f}")
        print(f"Max: {rolling_mad.dropna().max():.4f}")
        print(f"Min: {rolling_mad.dropna().min():.4f}")
        print(f"Std: {rolling_mad.dropna().std():.4f}")
        
        return rolling_median, rolling_mad

    @staticmethod
    def calculate_trend_strength(df, ema_short_col='ema_short', ema_medium_col='ema_medium'):
        """Calculate trend strength with validation and normalization"""
        # Verify columns exist
        if ema_short_col not in df.columns or ema_medium_col not in df.columns:
            raise ValueError(f"Missing required EMA columns: {ema_short_col} or {ema_medium_col}")
        
        # Calculate true range first
        if 'true_range' not in df.columns:
            df['true_range'] = pd.concat([
                df['high_price'] - df['low_price'],
                abs(df['high_price'] - df['close_price'].shift(1)),
                abs(df['low_price'] - df['close_price'].shift(1))
            ], axis=1).max(axis=1)
        
        # Calculate ATR if not present
        if 'atr' not in df.columns:
            atr_period = 24
            df['atr'] = df['true_range'].rolling(window=atr_period).mean()
        
        # Calculate absolute difference between EMAs
        ema_diff = abs(df[ema_short_col] - df[ema_medium_col])
        
        # Calculate percentage difference relative to medium EMA
        epsilon = 1e-10
        price_scale = df[ema_medium_col].abs() + epsilon
        
        # Calculate trend strength normalized by ATR
        atr_scale = df['atr'] + epsilon
        trend_strength = (ema_diff / price_scale) * (price_scale / atr_scale)
        
        # Clip extreme values
        trend_strength = trend_strength.clip(lower=0, upper=0.1)
        
        return trend_strength
    
    @staticmethod
    def calculate_rsi(df, period=14):
        """Calculate RSI technical indicator"""
        print(f"\nCalculating RSI with period {period}...")
        
        # Calculate price changes
        delta = df['close_price'].diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate initial average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)  # Add small constant to prevent division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Print diagnostics
        print("\nRSI Statistics:")
        print(f"Mean: {rsi.mean():.2f}")
        print(f"Max: {rsi.max():.2f}")
        print(f"Min: {rsi.min():.2f}")
        print(f"NaN count: {rsi.isna().sum()}")
        
        return rsi

    @staticmethod
    def add_rsi(df, custom_params=None):
        """Add RSI indicator to the dataframe"""
        df = df.copy()
        config = custom_params if custom_params is not None else BACKTEST_CONFIG['rsi']
        
        df['rsi'] = TechnicalIndicators.calculate_rsi(df, period=config['period'])
        return df

    @staticmethod
    def add_all_indicators(df, custom_params=None):
        """Add all technical indicators including RSI"""
        print("\nAdding indicators...")
        print(f"Input data shape: {df.shape}")
        
        df = df.copy()
        
        try:
            # Check if indicators already exist
            if all(col in df.columns for col in ['ema_short', 'ema_medium', 'trend_strength', 'atr', 'rsi']):
                print("Indicators already present, skipping recalculation")
                return df
                
            # Calculate all required indicators
            df['volatility'] = TechnicalIndicators.calculate_parkinson_volatility(df)
            df = TechnicalIndicators.add_adaptive_ema(df)
            df = TechnicalIndicators.add_rsi(df, custom_params)
            
            # Calculate ATR
            df['true_range'] = pd.concat([
                df['high_price'] - df['low_price'],
                abs(df['high_price'] - df['close_price'].shift(1)),
                abs(df['low_price'] - df['close_price'].shift(1))
            ], axis=1).max(axis=1)
            
            atr_period = 24
            if custom_params and 'ema' in custom_params:
                atr_period = custom_params['ema'].get('medium', atr_period)
            
            df['atr'] = df['true_range'].rolling(window=atr_period).mean()
            
            # Calculate trend strength
            df['trend_strength'] = TechnicalIndicators.calculate_trend_strength(df)
            
            return df
            
        except Exception as e:
            print(f"Error adding indicators: {str(e)}")
            raise