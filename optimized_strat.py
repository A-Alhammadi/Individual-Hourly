# optimized_strat.py

import os
import json
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

from config import BACKTEST_CONFIG
from indicators import TechnicalIndicators
from strategies import TradingStrategies

def optimize_strategy_parameters(df, symbol, start_date, end_date):
    """Optimize strategy parameters with adaptive volatility adjustments"""
    print(f"\nOptimizing parameters for {symbol}")
    print(f"Training period: {start_date} to {end_date}")
    
    # Calculate required warmup period (30 days in hours)
    warmup_hours = 30 * 24
    
    # Get warmup period start date
    train_start = pd.to_datetime(start_date)
    warmup_start = train_start - pd.Timedelta(hours=warmup_hours)
    train_end = pd.to_datetime(end_date)
    
    # Get data including warmup
    mask = (df.index >= warmup_start) & (df.index <= train_end)
    full_df = df[mask].copy()
    
    # Calculate all indicators ONCE at the beginning
    print("\nCalculating base indicators for optimization...")
    full_df = TechnicalIndicators.add_all_indicators(full_df)
    
    # Check if we have enough data
    if len(full_df) < BACKTEST_CONFIG['optimization']['min_training_days']:
        raise ValueError(f"Insufficient training data for {symbol}")
        
    # Add flags for diagnostics
    full_df.is_optimization = True
    full_df.save_diagnostics = True
    
    # Calculate volatility with diagnostics
    full_df['volatility'] = TechnicalIndicators.calculate_parkinson_volatility(full_df)
    
    # Remove flags
    del full_df.is_optimization
    del full_df.save_diagnostics
    
    # Check volatility calculations
    print("\nVolatility Statistics:")
    print(f"Mean: {full_df['volatility'].mean():.4f}")
    print(f"Max: {full_df['volatility'].max():.4f}")
    print(f"Min: {full_df['volatility'].min():.4f}")
    print(f"Std: {full_df['volatility'].std():.4f}")
    
    if full_df['volatility'].std() < 0.1:
        print("\nWARNING: Volatility values appear too small!")
    elif full_df['volatility'].std() > 1.0:
        print("\nWARNING: Volatility values appear too large!")

    # Initialize results structure with trade tracking
    optimization_results = {
        'optimization_id': f"opt_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'symbol': symbol,
        'training_period': {'start': start_date, 'end': end_date},
        'market_conditions': {
            'high_volatility': {
                'ema': {'parameters': None, 'sharpe_ratio': -np.inf, 'trades': 0},
                'rsi': {'parameters': None, 'sharpe_ratio': -np.inf, 'trades': 0}
            },
            'normal': {
                'ema': {'parameters': None, 'sharpe_ratio': -np.inf, 'trades': 0},
                'rsi': {'parameters': None, 'sharpe_ratio': -np.inf, 'trades': 0}
            }
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Get parameter ranges
    ema_ranges = BACKTEST_CONFIG['optimization']['parameter_ranges']['ema']
    vol_ranges = BACKTEST_CONFIG['optimization']['parameter_ranges']['volatility']
    rsi_ranges = BACKTEST_CONFIG['optimization']['parameter_ranges']['rsi']
    # Debug: Print parameter ranges
    print("\nParameter Ranges:")
    print("EMA parameters:", ema_ranges)
    print("Volatility parameters:", vol_ranges)
    
    # Generate all parameter combinations
    ema_params = [
        ema_ranges['short'],
        ema_ranges['medium'],
        ema_ranges['volatility_window'],
        #ema_ranges['volatility_threshold'],
        ema_ranges['min_trend_strength']
    ]
    
    vol_params = [
        vol_ranges['annualization_factor'],
        vol_ranges['baseline_window_multiplier'],
        vol_ranges['baseline_lookback_gap'],
        vol_ranges['min_periods_multiplier']
    ]

    rsi_params = [
    rsi_ranges['period'],
    rsi_ranges['overbought'],
    rsi_ranges['oversold'],
    rsi_ranges['weight']
    ]
    
    ema_combinations = list(itertools.product(*ema_params))
    vol_combinations = list(itertools.product(*vol_params))
    rsi_combinations = list(itertools.product(*rsi_params))

    total_combinations = len(ema_combinations) * len(vol_combinations) * len(rsi_combinations)

    print(f"\nTesting {total_combinations} total parameter combinations...")
    print(f"Number of EMA combinations: {len(ema_combinations)}")
    print(f"Number of volatility combinations: {len(vol_combinations)}")
    
    best_score = -np.inf
    combinations_tested = 0
    valid_combinations = 0
    
    # Track best parameters for each regime
    best_params = {
        'high_volatility': {'params': None, 'sharpe': -np.inf, 'trades': 0},
        'normal': {'params': None, 'sharpe': -np.inf, 'trades': 0}
    }
    
    # Iterate over all parameter combinations
    for ema_combo in ema_combinations:
        ema_dict = {
            'short': ema_combo[0],
            'medium': ema_combo[1],
            'volatility_window': ema_combo[2],
            #'volatility_threshold': ema_combo[3],
            'min_trend_strength': ema_combo[3]
        }
        
        # Skip invalid EMA combinations
        if ema_dict['short'] >= ema_dict['medium']:
            continue
            
        for vol_combo in vol_combinations:
            vol_dict = {
                'annualization_factor': vol_combo[0],
                'baseline_window_multiplier': vol_combo[1],
                'baseline_lookback_gap': vol_combo[2],
                'min_periods_multiplier': vol_combo[3]
            }

            # Add RSI combinations loop
            for rsi_combo in rsi_combinations:
                rsi_dict = {
                    'period': rsi_combo[0],
                    'overbought': rsi_combo[1],
                    'oversold': rsi_combo[2],
                    'weight': rsi_combo[3],
                    'signal_threshold': BACKTEST_CONFIG['rsi']['signal_threshold']
                }

                # Update how parameters are combined
                full_params = {
                    'ema': ema_dict,
                    'volatility': vol_dict,
                    'rsi': rsi_dict
                }

                try:
                    # Use the pre-calculated indicators in full_df
                    test_strategy = OptimizedStrategy({})
                    regimes = pd.Series(index=full_df.index, dtype='object')
                    
                    # Detect regimes for each bar in the entire (warmup + training) set
                    for i in range(len(full_df)):
                        regimes.iloc[i] = test_strategy.detect_market_regime(
                            full_df, 
                            i, 
                            full_params
                        )
                    
                    # Only use the actual training period (exclude warmup)
                    train_mask = (full_df.index >= train_start) & (full_df.index <= train_end)
                    train_data = full_df[train_mask].copy()
                    train_regimes = regimes[train_mask]
                    
                    # Split data into regimes
                    high_vol_mask = (train_regimes == 'high_volatility')
                    normal_mask = ~high_vol_mask
                    
                    # Optional debug: regime distribution
                    high_vol_count = high_vol_mask.sum()
                    normal_count = normal_mask.sum()
                    print(f"\nRegime distribution for current combination:")
                    print(f"High volatility periods: {high_vol_count}")
                    print(f"Normal periods: {normal_count}")
                    
                    total_sharpe = 0
                    regime_count = 0
                    
                    # Evaluate each regime separately
                    for regime, rmask in [('high_volatility', high_vol_mask), ('normal', normal_mask)]:
                        regime_data = train_data[rmask].copy()
                        if len(regime_data) < 20:
                            print(f"Skipping {regime} regime due to insufficient data: {len(regime_data)} periods")
                            continue
                        
                        # Use the pre-calculated indicators for backtest
                        sharpe = test_strategy.test_parameters(
                            df_indicators=regime_data,  # Pre-calculated
                            train_mask=rmask,
                            regime=regime,
                            full_params=full_params
                        )
                        
                        if sharpe is not None:
                            # Generate signals to count trades - Use full_params instead of ema_dict
                            signals = TradingStrategies.ema_strategy(regime_data, full_params)
                            num_trades = len(signals[signals != 0])
                            
                            if num_trades > 0:  # Only consider valid if we have trades
                                print(f"{regime} regime - Current Sharpe: {sharpe:.4f}, Trades: {num_trades}")
                                
                                # Check if this is a new best for that regime
                                if sharpe > best_params[regime]['sharpe']:
                                    best_params[regime]['params'] = full_params.copy()
                                    best_params[regime]['sharpe'] = float(sharpe)
                                    best_params[regime]['trades'] = num_trades
                                    
                                    # Update optimization results
                                    optimization_results['market_conditions'][regime]['ema'].update({
                                        'parameters': full_params.copy(),
                                        'sharpe_ratio': float(sharpe),
                                        'trades': num_trades
                                    })
                                    optimization_results['market_conditions'][regime]['rsi'].update({
                                        'parameters': full_params['rsi'].copy(),
                                        'sharpe_ratio': float(sharpe),
                                        'trades': num_trades
                                    })
                                    print(f"New best parameters found for {regime} regime")
                                    print(f"Parameters: {full_params}")
                                    print(f"Sharpe Ratio: {sharpe:.4f}")
                                    print(f"Number of trades: {num_trades}")
                                
                                total_sharpe += sharpe
                                regime_count += 1
                                valid_combinations += 1
                    
                    # Track best overall parameters (across both regimes)
                    if regime_count > 0:
                        avg_sharpe = total_sharpe / regime_count
                        if avg_sharpe > best_score:
                            best_score = avg_sharpe
                            # Optionally update best across both regimes if desired
                            for rgm in ['high_volatility', 'normal']:
                                if rgm not in optimization_results['market_conditions']:
                                    optimization_results['market_conditions'][rgm] = {
                                        'ema': {'parameters': full_params.copy(), 'sharpe_ratio': -np.inf, 'trades': 0}
                                    }
                
                except Exception as e:
                    print(f"Error testing parameters {full_params}: {str(e)}")
                    continue
            
    print(f"\nOptimization complete:")
    print(f"Total combinations tested: {combinations_tested}")
    print(f"Valid combinations found: {valid_combinations}")
    print(f"Best overall Sharpe ratio: {best_score:.4f}")
    
    # Save results
    results_dir = BACKTEST_CONFIG['results_dir']
    symbol_dir = os.path.join(results_dir, symbol.replace('/', '_'))
    os.makedirs(symbol_dir, exist_ok=True)
    
    results_file = os.path.join(symbol_dir, f"optimization_{optimization_results['optimization_id']}.json")
    optimization_results['results_file'] = results_file
    
    print("\nOptimization Results:")
    for regime in ['high_volatility', 'normal']:
        print(f"\n{regime.title()} Regime:")
        params = optimization_results['market_conditions'][regime]['ema']['parameters']
        sharpe = optimization_results['market_conditions'][regime]['ema']['sharpe_ratio']
        trades = optimization_results['market_conditions'][regime]['ema'].get('trades', 0)
        if params:
            print(f"  EMA Parameters: {params['ema']}")
            print(f"  Volatility Parameters: {params['volatility']}")
            print(f"  Sharpe Ratio: {sharpe:.4f}")
            print(f"  Number of trades: {trades}")
        else:
            print("  No valid parameters found")
    
    with open(results_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    return optimization_results


class OptimizedStrategy:
    def __init__(self, optimization_results):
        self.optimization_results = optimization_results
        self.best_params = {}
        self.market_condition_params = {}
        self.parameter_changes = None
        self.base_performance = None
        self.reversion_count = 0
        self.last_change_time = None
        self.recent_changes = []
        
        # Add state tracking for regime detection
        self.last_regime = 'normal'
        self.regime_duration = 0
        self.min_regime_duration = 24  # Minimum hours before allowing regime switch
        
        # Extract parameters for each market regime
        if 'market_conditions' in optimization_results:
            self.market_condition_params = optimization_results['market_conditions']
            if ('normal' in self.market_condition_params and 
                self.market_condition_params['normal']['ema']['parameters'] is not None):
                self.best_params = self.market_condition_params['normal']['ema']['parameters']
            else:
                self.best_params = {
                    'ema': BACKTEST_CONFIG['ema'].copy(),
                    'volatility': BACKTEST_CONFIG['volatility'].copy()
                }
    
    def detect_market_regime(self, df, i, params=None):
        """
        Detects 'normal' vs 'high_volatility' using a z-score approach.
        This method requires that df['volatility'] exists (Parkinson volatility).
        We also assume we have rolling_median and rolling_mad cached, or
        else we compute them on the fly.
        """
        # If no regime is set yet, default to normal:
        if not hasattr(self, 'last_regime'):
            self.last_regime = 'normal'
        
        # Ensure we have a minimum of 'i' bars to do anything.
        if i < 10:
            return 'normal'  # Not enough data yet for any meaningful classification

        # 1) If not cached yet, compute rolling median & MAD for 'volatility'.
        #    We only do this once, then store in self._cached_stats
        if not hasattr(self, '_cached_stats') or self._cached_stats is None:
            rolling_median, rolling_mad = TechnicalIndicators.calculate_volatility_stats(df['volatility'])
            self._cached_stats = (rolling_median, rolling_mad)
        else:
            rolling_median, rolling_mad = self._cached_stats

        # 2) Get current volatility and corresponding median/MAD
        current_vol = df['volatility'].iloc[i]
        med_val = rolling_median.iloc[i]
        mad_val = rolling_mad.iloc[i]

        # Safety check: If MAD is zero or NaN, we can't compute z-score, so return last regime
        if np.isnan(med_val) or np.isnan(mad_val) or (mad_val == 0):
            print(f"⚠ Warning at bar {i}: Invalid MAD ({mad_val}) or Median ({med_val}), keeping regime: {self.last_regime}")
            return self.last_regime  

        # 3) Calculate z-score = how many MADs away current_vol is from median
        zscore = abs((current_vol - med_val) / (mad_val + 1e-10))

        # **DEBUG LOGGING**
        if i % 200 == 0:  # Print every 200 bars for tracking
            print(f"🔍 Bar {i}: vol={current_vol:.4f}, median={med_val:.4f}, mad={mad_val:.4f}, zscore={zscore:.3f}")

        # 4) Set z-score thresholds for "high volatility" vs. "normal."
        high_zscore_threshold = 2.0   # More conservative threshold for entering high vol
        exit_zscore_threshold = 1.5   # More conservative threshold for exiting high vol

        # Track how long we've been in the current regime
        if not hasattr(self, 'regime_duration'):
            self.regime_duration = 0

        # Increase the duration by 1 bar
        self.regime_duration += 1

        # Minimum periods before allowing regime switches
        if not hasattr(self, 'min_regime_duration'):
            self.min_regime_duration = 4  # 4 hours minimum before any regime switch

        # Minimum time to stay in high volatility
        min_high_vol_duration = 8  # 8 hours minimum before switching back to normal

        new_regime = self.last_regime

        # 5) Use hysteresis logic:
        if self.last_regime == 'normal':
            # Switch to high_volatility if zscore is above high_zscore_threshold
            # and we've spent enough bars in the current regime
            if zscore > high_zscore_threshold and self.regime_duration >= self.min_regime_duration:
                new_regime = 'high_volatility'
                self.regime_duration = 0  # reset
                print(f" Switching to HIGH VOLATILITY at bar {i}, z-score={zscore:.3f}")
        else:
            # If we're currently in high_volatility, exit to normal if zscore < exit_zscore_threshold
            # and we've been in high vol for at least min_high_vol_duration
            if zscore < exit_zscore_threshold and self.regime_duration >= max(self.min_regime_duration, min_high_vol_duration):
                new_regime = 'normal'
                self.regime_duration = 0
                print(f"⬇ Switching to NORMAL at bar {i}, z-score={zscore:.3f}")

        self.last_regime = new_regime
        return new_regime

    def test_parameters(self, df_indicators, train_mask, regime, full_params):
        """Test parameters using pre-calculated indicators"""
        if len(df_indicators) < 20:
            return None
        
        # Generate signals using existing indicators
        signals = TradingStrategies.ema_strategy(df_indicators, full_params)
        
        # Calculate returns
        strategy_returns = df_indicators['close_price'].pct_change() * signals.shift(1)
        strategy_returns.fillna(0, inplace=True)
        
        # Calculate metrics
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            annualization_factor = full_params['volatility']['annualization_factor']
            sharpe = np.sqrt(annualization_factor) * (
                strategy_returns.mean() / strategy_returns.std()
            )
            return sharpe
        
        return None

    def run_dynamic(self, df):
        """Generate signals using adaptive EMAs and dynamic position sizing"""
        print("\nRunning Adaptive EMA Strategy")
        
        # Make a copy of the dataframe
        df = df.copy()
        
        # Add all indicators once
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Initialize signals series and regime tracking
        final_signals = pd.Series(0.0, index=df.index)
        regime_changes = pd.Series(index=df.index, dtype='object', name='regime')       
    
        # Add adaptive indicators
        df = TechnicalIndicators.add_adaptive_ema(df)
        
        # Track regime changes for debugging
        regime_transitions = 0
        last_regime = None
        
        # Track filtered trades and parameter changes
        filtered_trades = 0
        potential_trades = 0
        self.parameter_changes = pd.Series(0, index=df.index)
        
        # Performance monitoring variables
        performance_window = BACKTEST_CONFIG['adaptive']['performance_window']
        reversion_threshold = BACKTEST_CONFIG['adaptive']['reversion_threshold']
        min_reversion_count = BACKTEST_CONFIG['adaptive']['min_reversion_count']
        
        # Track entry and exit positions separately
        current_position = 0
        entry_price = None
        was_above = None  # Track previous crossing state
        
        # Track base performance for self-healing
        self.base_performance = None
        self.reversion_count = 0
        self.last_change_time = None
        self.recent_changes = []
        
        print("\nGenerating trading signals...")
        
        # Generate signals for each bar
        for i in range(1, len(df)):
            # Get appropriate parameters for regime detection
            if ('high_volatility' in self.market_condition_params and 
                self.market_condition_params['high_volatility']['ema']['parameters'] is not None):
                regime_params = self.market_condition_params['high_volatility']['ema']['parameters']
            else:
                regime_params = self.best_params
                
            regime = self.detect_market_regime(df, i, regime_params)
            regime_changes.iloc[i] = regime
            
            if regime != last_regime:
                regime_transitions += 1
                last_regime = regime
                # Record parameter change time for dead zone calculation
                self.recent_changes.append(df.index[i])
                # Clean up old changes
                self.recent_changes = [t for t in self.recent_changes 
                                    if t > df.index[i] - pd.Timedelta(days=BACKTEST_CONFIG['adaptive']['lookback_periods'])]
            
            # Get regime-specific parameters
            if (regime in self.market_condition_params and 
                self.market_condition_params[regime]['ema']['parameters'] is not None):
                params = self.market_condition_params[regime]['ema']['parameters']['ema'].copy()
            else:
                params = BACKTEST_CONFIG['ema'].copy()
            
            # Performance monitoring and self-healing
            if i >= performance_window:
                current_performance = (
                    df['close_price'].iloc[i] / df['close_price'].iloc[i-performance_window] - 1
                )
                
                if self.base_performance is None:
                    self.base_performance = current_performance
                elif current_performance < self.base_performance * reversion_threshold:
                    self.reversion_count += 1
                    if self.reversion_count >= min_reversion_count:
                        # Revert to base parameters
                        params = BACKTEST_CONFIG['ema'].copy()
                        self.reversion_count = 0
                        self.base_performance = None
                        self.parameter_changes.iloc[i] = 1
                else:
                    self.reversion_count = 0
                    self.base_performance = current_performance
            
            # Track EMA crossovers
            is_above = df['ema_short'].iloc[i] > df['ema_medium'].iloc[i]
            
            # Initialize was_above on first iteration
            if was_above is None:
                was_above = is_above
                continue
            
            # Use pre-calculated trend strength
            trend_strength = df['trend_strength'].iloc[i]
            
            # Add debug logging for trend strength
            if i % 1000 == 0:
                print(f"\nTrend Strength at {df.index[i]}:")
                print(f"Current value: {trend_strength:.6f}")
                print(f"Min required: {params['min_trend_strength']:.6f}")
            
            # Get position scale based on volatility
            scale = df['position_scale'].iloc[i]
            
            potential_trades += 1
            
            if trend_strength >= params['min_trend_strength']:
                if current_position == 0:  # No position
                    if is_above and not was_above:  # Bullish crossover
                        final_signals.iloc[i] = 1.0 * scale
                        current_position = 1
                        entry_price = df['close_price'].iloc[i]
                        print(f"Long entry at {df.index[i]}, price: {entry_price:.2f}")
                    elif not is_above and was_above:  # Bearish crossover
                        final_signals.iloc[i] = -1.0 * scale
                        current_position = -1
                        entry_price = df['close_price'].iloc[i]
                        print(f"Short entry at {df.index[i]}, price: {entry_price:.2f}")
                else:  # In a position
                    if current_position == 1 and not is_above and was_above:  # Exit long
                        final_signals.iloc[i] = -1.0 * scale
                        current_position = 0
                        exit_price = df['close_price'].iloc[i]
                        print(f"Long exit at {df.index[i]}, price: {exit_price:.2f}, P/L: {((exit_price/entry_price)-1)*100:.2f}%")
                        entry_price = None
                    elif current_position == -1 and is_above and not was_above:  # Exit short
                        final_signals.iloc[i] = 1.0 * scale
                        current_position = 0
                        exit_price = df['close_price'].iloc[i]
                        print(f"Short exit at {df.index[i]}, price: {exit_price:.2f}, P/L: {((entry_price/exit_price)-1)*100:.2f}%")
                        entry_price = None
                    else:
                        # Update position size based on current scale
                        final_signals.iloc[i] = current_position * scale
            else:
                filtered_trades += 1
                if is_above != was_above:
                    print(f"Filtered crossover at {df.index[i]}, trend_strength: {trend_strength:.6f} vs min: {params['min_trend_strength']}")
                # Maintain current position with updated scale
                final_signals.iloc[i] = current_position * scale
                
            # Update crossing state
            was_above = is_above
        
        # Print debug info
        print(f"\nStrategy Summary:")
        print(f"Total regime transitions: {regime_transitions}")
        num_entries = (
            ((final_signals > 0) & (final_signals.shift(1) <= 0)).sum() +
            ((final_signals < 0) & (final_signals.shift(1) >= 0)).sum()
        )
        num_exits = (
            ((final_signals == 0) & (final_signals.shift(1) != 0)).sum()
        )
        print(f"Total entries: {num_entries}")
        print(f"Total exits: {num_exits}")
        print(f"Potential trades: {potential_trades}")
        print(f"Filtered trades: {filtered_trades}")
        print(f"Trades filtered by trend strength: {filtered_trades}/{potential_trades} ({filtered_trades/potential_trades*100:.1f}% filtered)")
        
        # Store regime changes and parameter changes for later analysis
        final_signals.regime_changes = regime_changes
        final_signals.parameter_changes = self.parameter_changes
        
        return final_signals