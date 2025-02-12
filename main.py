# main.py

import pandas as pd
import numpy as np
from database import DatabaseHandler
from indicators import TechnicalIndicators
from strategies import TradingStrategies
from backtester import Backtester
from config import BACKTEST_CONFIG
import importlib
import optimized_strat
importlib.reload(optimized_strat)
from optimized_strat import OptimizedStrategy  # Ensure fresh import
from optimized_strat import optimize_strategy_parameters, OptimizedStrategy
from TimeUtils import TimeUtils

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
from datetime import datetime, timedelta

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_strategy_metrics(portfolio_df, trades_df, period_start, period_end, initial_capital):
    """Calculate strategy metrics for a specific period using hourly data"""
    mask = (portfolio_df.index >= period_start) & (portfolio_df.index <= period_end)
    period_data = portfolio_df[mask]
    
    if len(period_data) == 0:
        return None

    # Calculate returns
    period_return = (
        period_data['total_value'].iloc[-1] - period_data['total_value'].iloc[0]
    ) / period_data['total_value'].iloc[0]
    
    # Calculate annual return using hours
    hours_in_period = len(period_data)
    hours_per_year = 365 * 24
    annual_return = period_return * (hours_per_year / hours_in_period)

    # Calculate Sharpe ratio using hourly returns
    hourly_returns = period_data['total_value'].pct_change()
    sharpe_ratio = np.sqrt(hours_per_year) * (
        hourly_returns.mean() / hourly_returns.std()
    ) if hourly_returns.std() != 0 else 0

    # Calculate drawdown
    peak = period_data['total_value'].expanding(min_periods=1).max()
    drawdown = (period_data['total_value'] - peak) / peak
    max_drawdown = drawdown.min()

    # Calculate trade statistics
    if trades_df is not None and not trades_df.empty:
        period_trades = trades_df[trades_df['date'].between(period_start, period_end)]
        num_trades = len(period_trades)
        
        # Calculate win rate using paired trades
        if num_trades > 1:
            paired_trades = period_trades.iloc[::2]  # Entry trades
            exit_trades = period_trades.iloc[1::2]   # Exit trades
            if len(paired_trades) > 0 and len(exit_trades) > 0:
                min_length = min(len(paired_trades), len(exit_trades))  # Ensure equal length
                wins = sum(
                    exit_trades['value'].reset_index(drop=True).iloc[:min_length].to_numpy()
                    >
                    paired_trades['value'].reset_index(drop=True).iloc[:min_length].to_numpy()
                )
            else:
                wins = 0  # Default to 0 if no valid trades

            win_rate = wins / len(paired_trades)
        else:
            win_rate = 0
            
        total_fees = period_trades['fee'].sum() if 'fee' in period_trades.columns else 0
    else:
        num_trades = 0
        win_rate = 0
        total_fees = 0

    return {
        'Total Return': f"{period_return * 100:.2f}%",
        'Annual Return': f"{annual_return * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown * 100:.2f}%",
        'Number of Trades': num_trades,
        'Win Rate': f"{win_rate * 100:.2f}%",
        'Trading Fees': f"${total_fees:.2f}"
    }

def save_results_to_file(results_dict, symbol, output_dir, combined_results, optimization_results=None, train_end=None):
    """Save backtesting results to file with hourly metrics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_').replace('\\', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    
    summary_file = os.path.join(symbol_dir, f'summary_{timestamp}.txt')
    
    with open(summary_file, 'w') as f:
        f.write("=== Backtest Configuration ===\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Training Period: {BACKTEST_CONFIG['optimization']['training_start']} to {BACKTEST_CONFIG['optimization']['training_end']}\n")
        f.write(f"Testing Period: {BACKTEST_CONFIG['optimization']['testing_start']} to {BACKTEST_CONFIG['optimization']['testing_end']}\n")
        f.write(f"Initial Capital: ${BACKTEST_CONFIG['initial_capital']}\n\n")
        
        # Write optimization results if available
        if optimization_results and 'results_file' in optimization_results:
            f.write("=== Optimization Results (Training Period) ===\n")
            try:
                with open(optimization_results['results_file'], 'r') as opt_f:
                    f.write(opt_f.read())
                f.write("\n")
            except Exception as e:
                f.write(f"Error reading optimization results: {str(e)}\n\n")
        
        f.write("\n=== Testing Period Performance ===\n")
        f.write(f"Period: {BACKTEST_CONFIG['optimization']['testing_start']} to {BACKTEST_CONFIG['optimization']['testing_end']}\n\n")
        
        # Calculate testing period metrics for each strategy
        testing_metrics = []

        first_result = next(iter(results_dict.values()))
        df = first_result['portfolio']
        test_mask = (df.index >= BACKTEST_CONFIG['optimization']['testing_start']) & (df.index <= BACKTEST_CONFIG['optimization']['testing_end'])
        test_df = df[test_mask]
        
        if len(test_df) > 0:
            # Buy and Hold metrics using hourly data
            initial_price = float(test_df['close'].iloc[0])
            final_price = float(test_df['close'].iloc[-1])
            test_return = (final_price - initial_price) / initial_price
            
            # Calculate annual return using hours
            hours_in_period = len(test_df)
            hours_per_year = 365 * 24
            test_annual_return = test_return * (hours_per_year / hours_in_period)
            
            # Calculate Sharpe ratio for Buy and Hold
            bh_returns = test_df['close'].pct_change()
            bh_sharpe = np.sqrt(hours_per_year) * (
                bh_returns.mean() / bh_returns.std()
            ) if bh_returns.std() != 0 else 0
            
            # Calculate volatility
            bh_volatility = bh_returns.std() * np.sqrt(hours_per_year) * 100

            buy_hold_metrics = {
                'Strategy': 'Buy and Hold',
                'Total Return': f"{test_return * 100:.2f}%",
                'Annual Return': f"{test_annual_return * 100:.2f}%",
                'Sharpe Ratio': f"{bh_sharpe:.2f}",
                'Volatility': f"{bh_volatility:.2f}%",
                'Number of Trades': 1,
                'Trading Fees': f"${BACKTEST_CONFIG['initial_capital'] * BACKTEST_CONFIG['trading_fee']:.2f}"
            }
            
            # Metrics for each strategy
            for strategy_name, result in results_dict.items():
                metrics = calculate_strategy_metrics(
                    result['portfolio'],
                    result['trades'],
                    BACKTEST_CONFIG['optimization']['testing_start'],
                    BACKTEST_CONFIG['optimization']['testing_end'],
                    BACKTEST_CONFIG['initial_capital']
                )
                if metrics:
                    metrics['Strategy'] = strategy_name
                    testing_metrics.append(metrics)

            # Add buy and hold metrics
            testing_metrics.append(buy_hold_metrics)
            
            # Write testing period metrics table
            metrics_df = pd.DataFrame(testing_metrics)
            f.write(tabulate(metrics_df, headers="keys", tablefmt="grid", numalign="right"))
            
            # Additional details if "Optimized" is present
            if 'Optimized' in results_dict:
                f.write("\n\n=== Optimized Strategy Testing Period Details ===\n")
                opt_result = results_dict['Optimized']
                
                # Hourly returns resampled to monthly
                test_portfolio = opt_result['portfolio'][test_mask]
                monthly_returns = test_portfolio['total_value'].resample('M').last().pct_change()
                
                f.write("\nMonthly Returns:\n")
                f.write(monthly_returns.to_string())
                
                # Trade analysis
                if not opt_result['trades'].empty:
                    test_trades = opt_result['trades'][
                        opt_result['trades']['date'].between(
                            BACKTEST_CONFIG['optimization']['testing_start'], 
                            BACKTEST_CONFIG['optimization']['testing_end']
                        )
                    ]
                    f.write("\n\nTrade Analysis:\n")
                    trade_stats = {
                        'Total Trades': len(test_trades),
                        'Average Trade Duration': (
                            f"{test_trades['date'].diff().mean().total_seconds() / 3600:.1f} hours"
                            if len(test_trades) > 1 else "N/A"
                        ),
                        'Average Trade Size': (
                            f"${test_trades['value'].mean():.2f}" if not test_trades.empty else "N/A"
                        ),
                        'Largest Trade': (
                            f"${test_trades['value'].max():.2f}" if not test_trades.empty else "N/A"
                        ),
                        'Smallest Trade': (
                            f"${test_trades['value'].min():.2f}" if not test_trades.empty else "N/A"
                        ),
                        'Total Trading Fees': (
                            f"${test_trades['fee'].sum():.2f}" if not test_trades.empty else "$0.00"
                        )
                    }
                    
                    for stat, value in trade_stats.items():
                        f.write(f"{stat}: {value}\n")
        else:
            f.write("No data available for testing period\n")
    
    return summary_file

def save_test_period_results(results_dict, symbol, output_dir, train_end):
    """Save test period results with corrected regime statistics and hourly metrics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    
    test_results_file = os.path.join(symbol_dir, f'test_period_results_{timestamp}.txt')
    
    with open(test_results_file, 'w') as f:
        f.write(f"=== Test Period Performance for {symbol} ===\n")
        f.write(f"Test Period: {BACKTEST_CONFIG['optimization']['testing_start']} to {BACKTEST_CONFIG['optimization']['testing_end']}\n\n")
        
        # Get test period data only
        portfolio = results_dict['Optimized']['portfolio']
        test_mask = (
            (portfolio.index >= BACKTEST_CONFIG['optimization']['testing_start']) & 
            (portfolio.index <= BACKTEST_CONFIG['optimization']['testing_end'])
        )
        test_portfolio = portfolio[test_mask]
        
        print(f"\nAnalyzing test period data:")
        print(f"Test period start: {test_portfolio.index.min()}")
        print(f"Test period end: {test_portfolio.index.max()}")
        print(f"Number of hours: {len(test_portfolio)}")
        
        if len(test_portfolio) > 0:
            # Calculate strategy metrics using hourly data
            strategy_return = (
                test_portfolio['total_value'].iloc[-1] - test_portfolio['total_value'].iloc[0]
            ) / test_portfolio['total_value'].iloc[0]
            
            # Calculate annualized return using hours
            hours_in_period = len(test_portfolio)
            hours_per_year = 365 * 24
            strategy_annual_return = strategy_return * (hours_per_year / hours_in_period)
            
            # Calculate Sharpe ratio using hourly returns
            hourly_returns = test_portfolio['total_value'].pct_change()
            sharpe = np.sqrt(hours_per_year) * (
                hourly_returns.mean() / hourly_returns.std()
            ) if hourly_returns.std() != 0 else 0
            
            # Calculate volatility
            ann_volatility = hourly_returns.std() * np.sqrt(hours_per_year) * 100
            
            # Calculate max drawdown
            max_drawdown = (
                (test_portfolio['total_value'].cummax() - test_portfolio['total_value']) / 
                test_portfolio['total_value'].cummax()
            ).max()
            
            # Get trades for test period only
            trades = results_dict['Optimized']['trades']
            test_trades = trades[trades['date'].between(
                BACKTEST_CONFIG['optimization']['testing_start'],
                BACKTEST_CONFIG['optimization']['testing_end']
            )] if not trades.empty else pd.DataFrame()
            
            # Calculate win rate using paired trades
            num_trades = len(test_trades)
            if num_trades > 1:
                paired_trades = test_trades.iloc[::2]  # Entry trades
                exit_trades = test_trades.iloc[1::2]   # Exit trades
                if len(paired_trades) > 0 and len(exit_trades) > 0:
                    min_length = min(len(paired_trades), len(exit_trades))  # Ensure equal length
                    wins = sum(
                        exit_trades['value'].reset_index(drop=True).iloc[:min_length].to_numpy()
                        >
                        paired_trades['value'].reset_index(drop=True).iloc[:min_length].to_numpy()
                    )
                else:
                    wins = 0  # Default to 0 if no valid trades

                win_rate = wins / len(paired_trades)
            else:
                win_rate = 0
            
            # Calculate regime statistics for test period only
            if 'regime_changes' in results_dict['Optimized']:
                test_regimes = results_dict['Optimized']['regime_changes'][test_mask]
                regime_hours = {'high_volatility': 0, 'normal': 0}
                
                # Calculate hours in each regime
                current_regime = test_regimes.iloc[0]
                current_start = test_regimes.index[0]
                
                for idx, regime in test_regimes.items():
                    if regime != current_regime:
                        hours = (idx - current_start).total_seconds() / 3600
                        regime_hours[current_regime] += hours
                        current_regime = regime
                        current_start = idx
                
                # Add the last period
                hours = (test_regimes.index[-1] - current_start).total_seconds() / 3600
                regime_hours[current_regime] += hours
                
                total_hours = sum(regime_hours.values())
                high_vol_pct = (regime_hours['high_volatility'] / total_hours) * 100
                
                print(f"\nRegime Statistics for Test Period:")
                print(f"Total hours: {total_hours:.1f}")
                print(f"High volatility hours: {regime_hours['high_volatility']:.1f}")
                print(f"Normal hours: {regime_hours['normal']:.1f}")
            
            # Write results
            f.write("Regime-Based EMA Strategy Metrics:\n")
            f.write(f"Total Return: {strategy_return*100:.2f}%\n")
            f.write(f"Annualized Return: {strategy_annual_return*100:.2f}%\n")
            f.write(f"Annualized Volatility: {ann_volatility:.2f}%\n")
            f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
            f.write(f"Max Drawdown: {max_drawdown*100:.2f}%\n")
            f.write(f"Number of Trades: {num_trades}\n")
            f.write(f"Win Rate: {win_rate*100:.1f}%\n")
            
            if len(test_portfolio) > 0:
                # Grab the first and last close in the test window
                initial_price = test_portfolio['close'].iloc[0]
                final_price = test_portfolio['close'].iloc[-1]
                
                # Basic total return
                bh_return = (final_price - initial_price) / initial_price
                
                # Annualized B/H (using hours)
                hours_in_test = len(test_portfolio)
                hours_per_year = 365 * 24
                bh_annual_return = bh_return * (hours_per_year / hours_in_test)
                
                # Compute Sharpe ratio for B/H using same hourly returns logic
                bh_hourly_returns = test_portfolio['close'].pct_change()
                bh_sharpe = 0
                if bh_hourly_returns.std() > 0:
                    bh_sharpe = (bh_hourly_returns.mean() / bh_hourly_returns.std()) * np.sqrt(hours_per_year)
                
                # Print B/H in the file
                f.write("\nBuy & Hold Metrics:\n")
                f.write(f"  Total Return: {bh_return*100:.2f}%\n")
                f.write(f"  Annualized Return: {bh_annual_return*100:.2f}%\n")
                f.write(f"  Sharpe Ratio: {bh_sharpe:.2f}\n\n")
            else:
                f.write("\nNo data available for Buy & Hold comparison.\n")
            # Add average trade metrics if trades exist
            if num_trades > 0:
                f.write("\nTrade Statistics:\n")
                avg_duration = test_trades['date'].diff().mean().total_seconds() / 3600
                f.write(f"Average Trade Duration: {avg_duration:.1f} hours\n")
                f.write(f"Average Trade Size: ${test_trades['value'].mean():.2f}\n")
                f.write(f"Largest Trade: ${test_trades['value'].max():.2f}\n")
                f.write(f"Total Trading Fees: ${test_trades['fee'].sum():.2f}\n")
            
            # Add regime statistics
            if 'regime_changes' in results_dict['Optimized']:
                f.write("\nRegime Statistics:\n")
                f.write(f"High Volatility: {regime_hours['high_volatility']:.1f} hours ({high_vol_pct:.1f}%)\n")
                f.write(f"Normal Volatility: {regime_hours['normal']:.1f} hours ({100-high_vol_pct:.1f}%)\n")
                
                # Calculate regime transitions
                transitions = (test_regimes != test_regimes.shift(1)).sum()
                avg_regime_duration = total_hours / transitions if transitions > 0 else total_hours
                f.write(f"Regime Transitions: {transitions}\n")
                f.write(f"Average Regime Duration: {avg_regime_duration:.1f} hours\n")
        else:
            f.write("No data available for test period\n")
    
    return test_results_file

def analyze_strategy_correlations(results_dict, df, symbol, output_dir):
    """Analyze strategy performance by regime using hourly data"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    
    analysis_file = os.path.join(symbol_dir, f'regime_analysis_{timestamp}.txt')
    
    with open(analysis_file, 'w') as f:
        f.write(f"=== Regime Analysis for {symbol} ===\n\n")
        
        if 'regime_changes' in results_dict['Optimized']:
            regimes = results_dict['Optimized']['regime_changes']
            portfolio = results_dict['Optimized']['portfolio']
            
            # Analyze performance in each regime
            for regime in ['high_volatility', 'normal']:
                regime_mask = (regimes == regime)
                regime_returns = portfolio['total_value'].pct_change()[regime_mask]
                
                if len(regime_returns) > 0:
                    # Calculate annualized metrics using hourly data
                    hours_per_year = 365 * 24
                    ann_return = regime_returns.mean() * hours_per_year
                    volatility = regime_returns.std() * np.sqrt(hours_per_year)
                    sharpe = ann_return / volatility if volatility != 0 else 0
                    
                    f.write(f"\n{regime.title()} Regime Performance:\n")
                    f.write(f"Number of Hours: {regime_mask.sum()}\n")
                    f.write(f"Duration: {TimeUtils.hours_to_human_readable(regime_mask.sum())}\n")
                    f.write(f"Annualized Return: {ann_return*100:.2f}%\n")
                    f.write(f"Annualized Volatility: {volatility*100:.2f}%\n")
                    f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
                    
                    # Regime transition analysis
                    if len(regimes) > 1:
                        transitions = (regimes != regimes.shift(1)).sum()
                        f.write(f"\nRegime Transitions: {transitions}\n")
                        avg_duration_hours = len(regimes) / transitions if transitions > 0 else len(regimes)
                        f.write(f"Average Regime Duration: {TimeUtils.hours_to_human_readable(avg_duration_hours)}\n")
        else:
            f.write("No regime data available for analysis\n")
    
    return analysis_file

def calculate_plot_metrics(values):
    """Calculate metrics for plot labels using hourly data"""
    returns = values.pct_change()
    hours_per_year = 365 * 24
    ann_return = returns.mean() * hours_per_year
    volatility = returns.std() * np.sqrt(hours_per_year)
    sharpe = ann_return / volatility if volatility != 0 else 0
    
    # Calculate drawdown
    drawdown = (values.cummax() - values) / values.cummax()
    max_dd = drawdown.min()
    
    return {
        'final_value': values.iloc[-1],
        'ann_return': ann_return,
        'sharpe': sharpe,
        'max_dd': max_dd
    }

def calculate_hourly_metrics(values):
    """Calculate metrics using hourly data"""
    hourly_returns = values.pct_change()
    hours_per_year = 365 * 24
    
    # Calculate annualized return
    ann_return = hourly_returns.mean() * hours_per_year
    
    # Calculate annualized volatility
    volatility = hourly_returns.std() * np.sqrt(hours_per_year)
    
    # Calculate Sharpe ratio
    sharpe = ann_return / volatility if volatility != 0 else 0
    
    return ann_return, volatility, sharpe

def plot_results(results_dict, symbol, output_dir):
    """Plot full period results showing Regime-Based EMA vs Buy & Hold"""
    print(f"\nPlotting full period results for {symbol}")
    
    plt.style.use('default')
    strategy_colors = {
        'Regime-Based EMA': '#1E90FF',  # Blue
        'Buy and Hold': '#000000'       # Black
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Get data
    portfolio = results_dict['Optimized']['portfolio']
    initial_capital = float(BACKTEST_CONFIG['initial_capital'])
    initial_price = float(portfolio['close'].iloc[0])
    buy_hold_units = initial_capital / initial_price
    buy_hold_values = portfolio['close'] * buy_hold_units
    
    # Calculate metrics for Buy & Hold
    bh_ann_return, bh_vol, bh_sharpe = calculate_hourly_metrics(buy_hold_values)
    final_bh_value = buy_hold_values.iloc[-1]
    
    # Calculate metrics for Strategy
    strat_ann_return, strat_vol, strat_sharpe = calculate_hourly_metrics(portfolio['total_value'])
    final_value = portfolio['total_value'].iloc[-1]
    
    # Plot portfolio values with metrics
    ax1.plot(portfolio.index, buy_hold_values,
             label=f'Buy & Hold (${final_bh_value:.0f}, Return: {bh_ann_return*100:.1f}%, SR: {bh_sharpe:.2f})', 
             linewidth=1.5, color=strategy_colors['Buy and Hold'],
             linestyle='--', alpha=0.7)
    
    ax1.plot(portfolio.index, portfolio['total_value'],
            label=f'Regime-Based EMA (${final_value:.0f}, Return: {strat_ann_return*100:.1f}%, SR: {strat_sharpe:.2f})',
            linewidth=2.0, color=strategy_colors['Regime-Based EMA'])

    ax1.set_title(f'{symbol} - Portfolio Value', fontsize=12, pad=20)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot drawdowns
    bh_dd = (buy_hold_values.cummax() - buy_hold_values) / buy_hold_values.cummax()
    max_bh_dd = bh_dd.min()
    ax2.plot(portfolio.index, bh_dd, 
             label=f'Buy & Hold (Max DD: {max_bh_dd:.1%}, Vol: {bh_vol*100:.1f}%)',
             linewidth=1.5, color=strategy_colors['Buy and Hold'],
             linestyle='--', alpha=0.7)

    drawdown = (portfolio['total_value'].cummax() - portfolio['total_value']) / portfolio['total_value'].cummax()
    max_dd = drawdown.min()
    ax2.plot(portfolio.index, drawdown,
            label=f'Regime-Based EMA (Max DD: {max_dd:.1%}, Vol: {strat_vol*100:.1f}%)',
            linewidth=2.0, color=strategy_colors['Regime-Based EMA'])

    ax2.set_title('Drawdowns', fontsize=12, pad=20)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(fontsize=10, loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add regime changes as background shading
    if 'regime_changes' in results_dict['Optimized']:
        regimes = results_dict['Optimized']['regime_changes']
        last_idx = regimes.index[0]
        last_regime = regimes.iloc[0]
        
        regime_hours = {'high_volatility': 0, 'normal': 0}
        current_start = regimes.index[0]
        
        for i, (idx, regime) in enumerate(regimes.items()):
            if regime != last_regime:
                hours = (idx - current_start).total_seconds() / 3600
                regime_hours[last_regime] += hours
                
                if last_regime == 'high_volatility':
                    ax1.axvspan(last_idx, idx, alpha=0.2, color='red', 
                              label='High Volatility' if i==1 else "")
                    ax2.axvspan(last_idx, idx, alpha=0.2, color='red')
                
                last_idx = idx
                last_regime = regime
                current_start = idx
        
        # Handle the last period
        hours = (regimes.index[-1] - current_start).total_seconds() / 3600
        regime_hours[last_regime] += hours
        
        if last_regime == 'high_volatility':
            ax1.axvspan(last_idx, regimes.index[-1], alpha=0.2, color='red')
            ax2.axvspan(last_idx, regimes.index[-1], alpha=0.2, color='red')
        
        # Add regime distribution to title
        total_hours = sum(regime_hours.values())
        high_vol_pct = (regime_hours['high_volatility'] / total_hours) * 100
        ax1.set_title(f'{symbol} - Portfolio Value (High Vol: {high_vol_pct:.1f}% of time)',
                     fontsize=12, pad=20)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    plot_file = os.path.join(symbol_dir, f'full_period_plot_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Full period plot saved to {plot_file}")

def plot_test_period_results(results_dict, symbol, output_dir, test_start, test_end):
    """Plot test period results showing only Regime-Based EMA vs Buy & Hold"""
    print(f"\nPlotting test period results for {symbol}")
    print(f"Test period: {test_start} to {test_end}")
    
    plt.style.use('default')
    strategy_colors = {
        'Regime-Based EMA': '#1E90FF',  # Blue
        'Buy and Hold': '#000000'       # Black
    }

    safe_symbol = symbol.replace('/', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    os.makedirs(symbol_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Get test period data
    df = results_dict['Optimized']['portfolio']
    test_mask = (df.index >= test_start) & (df.index <= test_end)
    test_df = df[test_mask]

    if test_df.empty:
        print(f"No data available for test period {test_start} to {test_end}")
        return

    # Calculate values starting from initial capital
    initial_capital = float(BACKTEST_CONFIG['initial_capital'])
    
    # Buy & Hold starting from test period
    initial_price = float(test_df['close'].iloc[0])
    buy_hold_units = initial_capital / initial_price
    buy_hold_values = test_df['close'] * buy_hold_units

    # Strategy values starting from initial capital
    strategy_start_value = test_df['total_value'].iloc[0]
    scaling_factor = initial_capital / strategy_start_value
    strategy_values = test_df['total_value'] * scaling_factor

    # Calculate metrics for Buy & Hold
    bh_ann_return, bh_vol, bh_sharpe = calculate_hourly_metrics(buy_hold_values)
    final_bh_value = buy_hold_values.iloc[-1]
    
    # Calculate metrics for Strategy
    strat_ann_return, strat_vol, strat_sharpe = calculate_hourly_metrics(strategy_values)
    final_value = strategy_values.iloc[-1]
    
    # Plot portfolio values with metrics
    ax1.plot(test_df.index, buy_hold_values,
             label=f'Buy & Hold (${final_bh_value:.0f}, Return: {bh_ann_return*100:.1f}%, SR: {bh_sharpe:.2f})', 
             linewidth=1.5, color=strategy_colors['Buy and Hold'],
             linestyle='--', alpha=0.7)
    
    ax1.plot(test_df.index, strategy_values,
             label=f'Regime-Based EMA (${final_value:.0f}, Return: {strat_ann_return*100:.1f}%, SR: {strat_sharpe:.2f})',
             linewidth=2.0, color=strategy_colors['Regime-Based EMA'])

    ax1.set_title(f'{symbol} - Portfolio Value (Testing Period)', fontsize=12, pad=20)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot drawdowns
    bh_dd = (buy_hold_values.cummax() - buy_hold_values) / buy_hold_values.cummax()
    max_bh_dd = bh_dd.min()
    ax2.plot(test_df.index, bh_dd, 
             label=f'Buy & Hold (Max DD: {max_bh_dd:.1%}, Vol: {bh_vol*100:.1f}%)',
             linewidth=1.5, color=strategy_colors['Buy and Hold'],
             linestyle='--', alpha=0.7)

    drawdown = (strategy_values.cummax() - strategy_values) / strategy_values.cummax()
    max_dd = drawdown.min()
    ax2.plot(test_df.index, drawdown,
             label=f'Regime-Based EMA (Max DD: {max_dd:.1%}, Vol: {strat_vol*100:.1f}%)',
             linewidth=2.0, color=strategy_colors['Regime-Based EMA'])

    ax2.set_title('Drawdowns (Testing Period)', fontsize=12, pad=20)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(fontsize=10, loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Calculate regime distribution
    if 'regime_changes' in results_dict['Optimized']:
        regimes = results_dict['Optimized']['regime_changes'][test_mask]
        regime_hours = {'high_volatility': 0, 'normal': 0}
        current_start = regimes.index[0]
        current_regime = regimes.iloc[0]
        last_idx = regimes.index[0]
        last_regime = regimes.iloc[0]
        
        for i, (idx, regime) in enumerate(regimes.items()):
            if regime != current_regime:
                hours = (idx - current_start).total_seconds() / 3600
                regime_hours[current_regime] += hours
                current_regime = regime
                current_start = idx
            
            if regime != last_regime:
                if last_regime == 'high_volatility':
                    ax1.axvspan(last_idx, idx, alpha=0.2, color='red',
                              label='High Volatility' if i==1 else "")
                    ax2.axvspan(last_idx, idx, alpha=0.2, color='red')
                last_idx = idx
                last_regime = regime
        
        # Handle the last period
        hours = (regimes.index[-1] - current_start).total_seconds() / 3600
        regime_hours[current_regime] += hours
        
        if last_regime == 'high_volatility':
            ax1.axvspan(last_idx, regimes.index[-1], alpha=0.2, color='red')
            ax2.axvspan(last_idx, regimes.index[-1], alpha=0.2, color='red')
        
        # Update title with regime distribution
        total_hours = sum(regime_hours.values())
        high_vol_pct = (regime_hours['high_volatility'] / total_hours) * 100
        ax1.set_title(f'{symbol} - Portfolio Value (Testing Period, High Vol: {high_vol_pct:.1f}% of time)',
                     fontsize=12, pad=20)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(symbol_dir, f'testing_period_plot_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Test period plot saved to {plot_file}")
    
def save_combined_results(combined_results, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = os.path.join(output_dir, f'combined_results_{timestamp}.txt')
    
    with open(combined_file, 'w') as f:
        f.write("=== Combined Backtest Results ===\n")
        f.write(f"Period: {BACKTEST_CONFIG['start_date']} to {BACKTEST_CONFIG['end_date']}\n")
        f.write(f"Initial Capital per Symbol: ${BACKTEST_CONFIG['initial_capital']}\n\n")
        
        for symbol, results in combined_results.items():
            f.write(f"\n=== {symbol} Results ===\n")
            all_results = results['strategies'] + [results['buy_hold']]
            f.write(tabulate(all_results, headers="keys", tablefmt="grid", numalign="right"))
            f.write("\n" + "="*80 + "\n")
    
    return combined_file

def main():
    try:
        print("Starting backtesting process...")
        db = DatabaseHandler()
        output_dir = BACKTEST_CONFIG['results_dir']
        ensure_directory(output_dir)

        time_utils = TimeUtils()
        
        for symbol in BACKTEST_CONFIG['symbols']:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}...")
            print(f"{'='*50}")
            
            # Get configuration values
            test_start = BACKTEST_CONFIG['optimization']['testing_start']
            test_end = BACKTEST_CONFIG['optimization']['testing_end']
            training_hours = BACKTEST_CONFIG['optimization']['training_days']
            warmup_hours = 720  # 30 days * 24 hours for indicators
            
            # Calculate all required periods using instance methods
            fetch_period = time_utils.calculate_fetch_period(
                test_start, test_end, training_hours, warmup_hours
            )
            
            training_period = time_utils.calculate_training_period(
                test_start, training_hours
            )
            
            print(f"\nPeriod Calculations:")
            print(f"Training Start: {training_period['train_start']}")
            print(f"Training End: {training_period['train_end']}")
            print(f"Test Start: {training_period['test_start']}")
            print(f"Test End: {test_end}")
            print(f"Required Hours: {time_utils.hours_to_human_readable(fetch_period['total_hours'])}")
            # Fetch and prepare data
            data = db.get_historical_data(
                symbol, 
                fetch_period['fetch_start'],
                fetch_period['fetch_end']
            )
            
            # Validate data
            time_utils.validate_data_periods(data, fetch_period['total_hours'], symbol)
            
            print(f"\nData Validation:")
            print(f"Date Range: {data.index.min()} to {data.index.max()}")
            print(f"Total Hours: {len(data)}")
            
            # Add technical indicators to full dataset
            data_with_indicators = TechnicalIndicators.add_all_indicators(data.copy())
            
            # Optimize strategy parameters
            optimization_results = optimize_strategy_parameters(
                data_with_indicators.copy(),  # Pass a copy
                symbol,
                training_period['train_start'],
                training_period['train_end']
            )
            
            # Initialize and run the regime-based EMA strategy
            regime_strategy = OptimizedStrategy(optimization_results)
            dynamic_signals = regime_strategy.run_dynamic(data_with_indicators.copy())
            
            # Create results dictionary
            results = {}
            backtester = Backtester(data_with_indicators, "Optimized", lambda df: dynamic_signals)
            results['Optimized'] = backtester.run()
            results['Optimized']['regime_changes'] = dynamic_signals.regime_changes
            
            # Save test period results
            test_period_file = save_test_period_results(
                results, 
                symbol, 
                output_dir, 
                training_period['train_end']
            )
            print(f"✓ Test period results saved to {test_period_file}")
            
            # Generate plots if enabled
            if BACKTEST_CONFIG['save_plots']:
                try:
                    plot_results(results, symbol, output_dir)
                    print("✓ Full period plot saved")

                    plot_test_period_results(
                        results, 
                        symbol, 
                        output_dir, 
                        training_period['test_start'],
                        test_end
                    )
                    print("✓ Test period plot saved")
                except Exception as plot_error:
                    print(f"Warning: Error generating plots: {str(plot_error)}")
            
            print(f"\nProcessing complete for {symbol}")
            print(f"Results saved to: {output_dir}/{symbol.replace('/', '_')}")
                
    except Exception as e:
        print(f"\nError during backtesting: {str(e)}")
        raise
    finally:
        print("\nClosing database connection...")
        db.close()

if __name__ == "__main__":
    main()
