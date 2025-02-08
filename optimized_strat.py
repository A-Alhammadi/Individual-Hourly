# optimized_strat.py

import os
import json
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

from config import BACKTEST_CONFIG
from indicators import TechnicalIndicators
from strategies import TradingStrategies  # for the base signals (EMA, MACD, RSI, etc.)


def optimize_strategy_parameters(df, symbol, start_date, end_date):
    """Optimize EMA and volatility parameters for each market regime"""
    print(f"\nOptimizing parameters for {symbol}")
    print(f"Training period: {start_date} to {end_date}")
    
    # Filter training data
    mask = (df.index >= start_date) & (df.index <= end_date)
    train_df = df[mask].copy()
    if len(train_df) < BACKTEST_CONFIG['optimization']['min_training_days']:
        raise ValueError(f"Insufficient training data for {symbol}")
    
    # Initialize results structure
    optimization_results = {
        'optimization_id': f"opt_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'symbol': symbol,
        'training_period': {'start': start_date, 'end': end_date},
        'market_conditions': {
            'high_volatility': {'ema': {'parameters': None, 'sharpe_ratio': -np.inf}},
            'normal': {'ema': {'parameters': None, 'sharpe_ratio': -np.inf}}
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Get parameter ranges
    ema_ranges = BACKTEST_CONFIG['optimization']['parameter_ranges']['ema']
    vol_ranges = BACKTEST_CONFIG['optimization']['parameter_ranges']['volatility']
    
    # Generate all parameter combinations
    ema_params = [
        ema_ranges['short'],
        ema_ranges['medium'],
        ema_ranges['volatility_window'],
        ema_ranges['volatility_threshold']
    ]
    
    vol_params = [
        vol_ranges['annualization_factor'],
        vol_ranges['baseline_window_multiplier'],
        vol_ranges['baseline_lookback_gap'],
        vol_ranges['min_periods_multiplier']
    ]
    
    # Create all combinations
    ema_combinations = list(itertools.product(*ema_params))
    vol_combinations = list(itertools.product(*vol_params))
    
    print(f"Testing {len(ema_combinations) * len(vol_combinations)} total parameter combinations...")
    best_score = -np.inf
    
    # Test each combination
    for ema_combo in ema_combinations:
        ema_dict = {
            'short': ema_combo[0],
            'medium': ema_combo[1],
            'volatility_window': ema_combo[2],
            'volatility_threshold': ema_combo[3]
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
            
            # Combine parameters
            full_params = {
                'ema': ema_dict,
                'volatility': vol_dict
            }
            
            try:
                # Add indicators with current parameters
                df_indicators = TechnicalIndicators.add_all_indicators(
                    train_df.copy(),
                    custom_params=full_params
                )
                
                # Create test instance for regime detection
                test_strategy = OptimizedStrategy({})
                regimes = pd.Series(index=df_indicators.index, dtype='object')
                
                # Detect regimes for each bar
                for i in range(len(df_indicators)):
                    regimes.iloc[i] = test_strategy.detect_market_regime(df_indicators, i, full_params)
                
                # Split data into regimes
                high_vol_mask = (regimes == 'high_volatility')
                normal_mask = ~high_vol_mask
                
                # Test parameters in each regime
                total_sharpe = 0
                regime_count = 0
                
                for regime, mask in [('high_volatility', high_vol_mask), ('normal', normal_mask)]:
                    regime_data = df_indicators[mask]
                    if len(regime_data) < 20:  # Skip if too few samples
                        continue
                    
                    # Generate signals
                    signals = TradingStrategies.ema_strategy(regime_data, ema_dict)
                    
                    # Calculate returns
                    strategy_returns = regime_data['close_price'].pct_change() * signals.shift(1)
                    strategy_returns.fillna(0, inplace=True)
                    
                    # Calculate metrics
                    if len(strategy_returns) > 0 and strategy_returns.std() > 0:
                        sharpe = np.sqrt(vol_dict['annualization_factor']) * (
                            strategy_returns.mean() / strategy_returns.std()
                        )
                        
                        # Update best parameters if improved
                        if sharpe > optimization_results['market_conditions'][regime]['ema']['sharpe_ratio']:
                            optimization_results['market_conditions'][regime]['ema'].update({
                                'parameters': full_params.copy(),
                                'sharpe_ratio': float(sharpe)
                            })
                        
                        total_sharpe += sharpe
                        regime_count += 1
                
                # Track best overall parameters
                if regime_count > 0:
                    avg_sharpe = total_sharpe / regime_count
                    if avg_sharpe > best_score:
                        best_score = avg_sharpe
                        for regime in ['high_volatility', 'normal']:
                            if regime not in optimization_results['market_conditions']:
                                optimization_results['market_conditions'][regime] = {
                                    'ema': {'parameters': full_params.copy(), 'sharpe_ratio': -np.inf}
                                }
            
            except Exception as e:
                print(f"Error testing parameters {full_params}: {str(e)}")
                continue
    
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
        if params:
            print(f"  EMA Parameters: {params['ema']}")
            print(f"  Volatility Parameters: {params['volatility']}")
            print(f"  Sharpe Ratio: {sharpe:.4f}")
    
    with open(results_file, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    return optimization_results

class OptimizedStrategy:
    def __init__(self, optimization_results):
        self.optimization_results = optimization_results
        self.best_params = {}
        self.market_condition_params = {}
        
        # Extract parameters for each market regime
        if 'market_conditions' in optimization_results:
            self.market_condition_params = optimization_results['market_conditions']
    
    def detect_market_regime(self, df, i, params):
        """Detect if we're in high volatility or normal regime using configurable parameters"""
        vol_window = params['ema'].get('volatility_window', BACKTEST_CONFIG['ema']['volatility_window'])
        vol_threshold = params['ema'].get('volatility_threshold', BACKTEST_CONFIG['ema']['volatility_threshold'])
        
        # Get volatility configuration
        vol_config = params.get('volatility', BACKTEST_CONFIG['volatility'])
        annual_factor = vol_config['annualization_factor']
        baseline_multiplier = vol_config['baseline_window_multiplier']
        lookback_gap = vol_config['baseline_lookback_gap']
        min_periods = int(vol_window * vol_config['min_periods_multiplier'])
        
        # Need enough data for both current and baseline windows
        baseline_window = int(vol_window * baseline_multiplier)
        total_required = baseline_window + lookback_gap + vol_window
        if i < total_required:
            return 'normal'
        
        # Calculate returns
        returns = df['close_price'].pct_change()
        
        # Current volatility window
        current_start = i - vol_window
        current_end = i
        current_returns = returns.iloc[current_start:current_end]
        if len(current_returns) >= min_periods:
            current_vol = current_returns.std() * np.sqrt(annual_factor)
        else:
            return 'normal'
        
        # Baseline volatility window
        baseline_start = i - (baseline_window + vol_window + lookback_gap)
        baseline_end = i - (vol_window + lookback_gap)
        baseline_returns = returns.iloc[baseline_start:baseline_end]
        if len(baseline_returns) >= min_periods:
            baseline_vol = baseline_returns.std() * np.sqrt(annual_factor)
        else:
            return 'normal'
        
        if pd.isna(current_vol) or pd.isna(baseline_vol) or baseline_vol == 0:
            return 'normal'
        
        if current_vol > baseline_vol * vol_threshold:
            return 'high_volatility'
        return 'normal'

    def run_dynamic(self, df):
        """Generate signals using regime-specific parameters and track regime changes"""
        print("\nRunning Regime-Based EMA Strategy")
        
        # Make a copy of the dataframe
        df = df.copy()
        
        # Initialize signals series and regime tracking
        final_signals = pd.Series(0.0, index=df.index)
        regime_changes = pd.Series(index=df.index, dtype='object')
        
        # Get parameters from the optimization results
        if 'high_volatility' in self.market_condition_params:
            regime_params = self.market_condition_params['high_volatility']['ema']['parameters']
        else:
            regime_params = BACKTEST_CONFIG['ema']
        
        # Track regime changes for debugging
        regime_transitions = 0
        last_regime = None
        
        # Generate signals for each bar
        for i in range(len(df)):
            # Detect current market regime using parameters
            regime = self.detect_market_regime(df, i, regime_params)
            regime_changes.iloc[i] = regime
            
            if regime != last_regime:
                regime_transitions += 1
                last_regime = regime
            
            # Get regime-specific parameters
            if regime in self.market_condition_params:
                params = self.market_condition_params[regime]['ema']['parameters']['ema']  # Note the extra ['ema']
            else:
                params = BACKTEST_CONFIG['ema']
            
            # Calculate EMAs up to current bar using current regime's parameters
            current_data = df.iloc[:i+1]
            short_ema = current_data['close_price'].ewm(span=params['short'], adjust=False).mean().iloc[-1]
            medium_ema = current_data['close_price'].ewm(span=params['medium'], adjust=False).mean().iloc[-1]
            
            # Generate signal based on EMAs
            if short_ema > medium_ema:
                final_signals.iloc[i] = 1
            elif short_ema < medium_ema:
                final_signals.iloc[i] = -1
            else:
                final_signals.iloc[i] = 0
        
        # Print debug info
        print(f"Total regime transitions: {regime_transitions}")
        num_buys = (final_signals == 1).sum()
        num_sells = (final_signals == -1).sum()
        print(f"Total signals: {num_buys + num_sells}")
        print(f"  Buys: {num_buys}")
        print(f"  Sells: {num_sells}")
        
        # Store regime changes for later analysis
        final_signals.regime_changes = regime_changes
        
        return final_signals
