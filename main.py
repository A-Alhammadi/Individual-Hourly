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

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os
from datetime import datetime, timedelta

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def calculate_training_period(test_start, training_days):
    """Calculate training period end and start dates based on test start date"""
    test_start_date = pd.to_datetime(test_start)
    train_end = test_start_date - timedelta(days=1)  # Day before test start
    train_start = train_end - timedelta(days=training_days)
    return train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d')

def calculate_strategy_metrics(portfolio_df, trades_df, period_start, period_end, initial_capital):
    """Calculate strategy metrics for a specific period."""
    mask = (portfolio_df.index >= period_start) & (portfolio_df.index <= period_end)
    period_data = portfolio_df[mask]
    
    if len(period_data) == 0:
        return None

    period_return = (
        period_data['total_value'].iloc[-1] - period_data['total_value'].iloc[0]
    ) / period_data['total_value'].iloc[0]
    period_length = (period_data.index[-1] - period_data.index[0]).days / 365.0
    annual_return = period_return / period_length if period_length > 0 else 0

    daily_returns = period_data['total_value'].pct_change()
    sharpe_ratio = np.sqrt(365) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0

    peak = period_data['total_value'].expanding(min_periods=1).max()
    drawdown = (period_data['total_value'] - peak) / peak
    max_drawdown = drawdown.min()

    if trades_df is not None and not trades_df.empty:
        period_trades = trades_df[trades_df['date'].between(period_start, period_end)]
        num_trades = len(period_trades)
        win_rate = (period_trades['value'] > period_trades['value'].shift(1)).sum() / (num_trades // 2) if num_trades > 0 else 0
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
            # Buy and Hold metrics
            initial_price = float(test_df['close'].iloc[0])
            final_price = float(test_df['close'].iloc[-1])
            test_return = (final_price - initial_price) / initial_price
            test_period = (test_df.index[-1] - test_df.index[0]).days / 365.0
            test_annual_return = test_return / test_period if test_period > 0 else 0

            buy_hold_metrics = {
                'Strategy': 'Buy and Hold',
                'Total Return': f"{test_return * 100:.2f}%",
                'Annual Return': f"{test_annual_return * 100:.2f}%",
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
                
                # Monthly returns
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
                        'Largest Winning Trade': (
                            f"${test_trades['value'].max():.2f}" if not test_trades.empty else "N/A"
                        ),
                        'Largest Losing Trade': (
                            f"${test_trades['value'].min():.2f}" if not test_trades.empty else "N/A"
                        ),
                    }
                    
                    for stat, value in trade_stats.items():
                        f.write(f"{stat}: {value}\n")
        else:
            f.write("No data available for testing period\n")
    
    return summary_file

def save_test_period_results(results_dict, symbol, output_dir, train_end):
    """Simplified version focusing on EMA regime strategy"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_')
    symbol_dir = os.path.join(output_dir, safe_symbol)
    ensure_directory(symbol_dir)
    
    test_results_file = os.path.join(symbol_dir, f'test_period_results_{timestamp}.txt')
    
    with open(test_results_file, 'w') as f:
        f.write(f"=== Test Period Performance for {symbol} ===\n")
        f.write(f"Test Period: {BACKTEST_CONFIG['optimization']['testing_start']} to {BACKTEST_CONFIG['optimization']['testing_end']}\n\n")
        
        # Get test period data
        df = results_dict['Optimized']['portfolio']
        test_mask = (df.index >= BACKTEST_CONFIG['optimization']['testing_start']) & (df.index <= BACKTEST_CONFIG['optimization']['testing_end'])
        test_df = df[test_mask]
        
        if len(test_df) > 0:
            # Calculate metrics for Regime-Based EMA
            portfolio = results_dict['Optimized']['portfolio'][test_mask]
            trades = results_dict['Optimized']['trades'][
                results_dict['Optimized']['trades']['date'] >= train_end
            ] if not results_dict['Optimized']['trades'].empty else pd.DataFrame()
            
            strategy_return = (
                portfolio['total_value'].iloc[-1] - portfolio['total_value'].iloc[0]
            ) / portfolio['total_value'].iloc[0]
            
            test_days = (test_df.index[-1] - test_df.index[0]).days
            strategy_annual_return = strategy_return * (365 / test_days) if test_days > 0 else 0
            
            daily_returns = portfolio['total_value'].pct_change()
            sharpe = np.sqrt(365) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
            max_drawdown = ((portfolio['total_value'].cummax() - portfolio['total_value']) / portfolio['total_value'].cummax()).max()
            
            # Buy and Hold metrics
            initial_price = float(test_df['close'].iloc[0])
            final_price = float(test_df['close'].iloc[-1])
            bh_return = (final_price - initial_price) / initial_price
            bh_annual_return = bh_return * (365 / test_days) if test_days > 0 else 0
            
            # Write results
            f.write("Regime-Based EMA Strategy Metrics:\n")
            f.write(f"Total Return: {strategy_return*100:.2f}%\n")
            f.write(f"Annual Return: {strategy_annual_return*100:.2f}%\n")
            f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
            f.write(f"Max Drawdown: {max_drawdown*100:.2f}%\n")
            f.write(f"Number of Trades: {len(trades)}\n")
            if len(trades) > 0:
                win_rate = len(trades[trades['value'] > trades['value'].shift(1)]) / (len(trades)//2)
                f.write(f"Win Rate: {win_rate*100:.2f}%\n")
            
            f.write("\nBuy and Hold Metrics:\n")
            f.write(f"Total Return: {bh_return*100:.2f}%\n")
            f.write(f"Annual Return: {bh_annual_return*100:.2f}%\n")
            
            # Strategy vs Buy and Hold comparison
            f.write("\nStrategy vs Buy and Hold:\n")
            outperformance = strategy_return - bh_return
            f.write(f"Outperformance: {outperformance*100:+.2f}%\n")
            
            # Add regime statistics
            if 'regime_changes' in results_dict['Optimized']:
                regimes = results_dict['Optimized']['regime_changes']
                high_vol_periods = (regimes == 'high_volatility').sum()
                total_periods = len(regimes)
                f.write(f"\nRegime Statistics:\n")
                f.write(f"High Volatility Periods: {high_vol_periods} ({high_vol_periods/total_periods*100:.1f}%)\n")
                f.write(f"Normal Volatility Periods: {total_periods-high_vol_periods} ({(1-high_vol_periods/total_periods)*100:.1f}%)\n")
        else:
            f.write("No data available for test period\n")
    
    return test_results_file

def analyze_strategy_correlations(results_dict, df, symbol, output_dir):
    """Simplified version focusing on regime analysis"""
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
                    ann_return = regime_returns.mean() * 252
                    volatility = regime_returns.std() * np.sqrt(252)
                    sharpe = ann_return / volatility if volatility != 0 else 0
                    
                    f.write(f"\n{regime.title()} Regime Performance:\n")
                    f.write(f"Number of Periods: {regime_mask.sum()}\n")
                    f.write(f"Annualized Return: {ann_return*100:.2f}%\n")
                    f.write(f"Annualized Volatility: {volatility*100:.2f}%\n")
                    f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
                    
                    # Regime transition analysis
                    if len(regimes) > 1:
                        transitions = (regimes != regimes.shift(1)).sum()
                        f.write(f"\nRegime Transitions: {transitions}\n")
                        avg_duration = len(regimes) / transitions if transitions > 0 else len(regimes)
                        f.write(f"Average Regime Duration: {avg_duration:.1f} periods\n")
        else:
            f.write("No regime data available for analysis\n")
    
    return analysis_file

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
    
    # Plot portfolio values
    final_bh_value = buy_hold_values.iloc[-1]
    ax1.plot(portfolio.index, buy_hold_values,
             label=f'Buy and Hold (${final_bh_value:.0f})', 
             linewidth=1.5, color=strategy_colors['Buy and Hold'],
             linestyle='--', alpha=0.7)
    
    # Plot Regime-Based EMA strategy
    final_value = portfolio['total_value'].iloc[-1]
    ax1.plot(portfolio.index, portfolio['total_value'],
            label=f'Regime-Based EMA (${final_value:.0f})',
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
             label=f'Buy and Hold (Max DD: {max_bh_dd:.1%})',
             linewidth=1.5, color=strategy_colors['Buy and Hold'],
             linestyle='--', alpha=0.7)

    drawdown = (portfolio['total_value'].cummax() - portfolio['total_value']) / portfolio['total_value'].cummax()
    max_dd = drawdown.min()
    ax2.plot(portfolio.index, drawdown,
            label=f'Regime-Based EMA (Max DD: {max_dd:.1%})',
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
        
        for i, (idx, regime) in enumerate(regimes.items()):
            if regime != last_regime:
                if last_regime == 'high_volatility':
                    ax1.axvspan(last_idx, idx, alpha=0.2, color='red', 
                              label='High Volatility' if i==1 else "")
                    ax2.axvspan(last_idx, idx, alpha=0.2, color='red')
                last_idx = idx
                last_regime = regime
        
        # Handle the last period
        if last_regime == 'high_volatility':
            ax1.axvspan(last_idx, regimes.index[-1], alpha=0.2, color='red')
            ax2.axvspan(last_idx, regimes.index[-1], alpha=0.2, color='red')

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

    # Plot portfolio values
    final_bh_value = buy_hold_values.iloc[-1]
    ax1.plot(test_df.index, buy_hold_values,
             label=f'Buy and Hold (${final_bh_value:.0f})', 
             linewidth=1.5, color=strategy_colors['Buy and Hold'],
             linestyle='--', alpha=0.7)

    final_value = strategy_values.iloc[-1]
    ax1.plot(test_df.index, strategy_values,
            label=f'Regime-Based EMA (${final_value:.0f})',
            linewidth=2.0, color=strategy_colors['Regime-Based EMA'])

    ax1.set_title(f'{symbol} - Portfolio Value (Testing Period)', fontsize=12, pad=20)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot drawdowns for test period
    bh_dd = (buy_hold_values.cummax() - buy_hold_values) / buy_hold_values.cummax()
    max_bh_dd = bh_dd.min()
    ax2.plot(test_df.index, bh_dd, 
             label=f'Buy and Hold (Max DD: {max_bh_dd:.1%})',
             linewidth=1.5, color=strategy_colors['Buy and Hold'],
             linestyle='--', alpha=0.7)

    drawdown = (strategy_values.cummax() - strategy_values) / strategy_values.cummax()
    max_dd = drawdown.min()
    ax2.plot(test_df.index, drawdown,
            label=f'Regime-Based EMA (Max DD: {max_dd:.1%})',
            linewidth=2.0, color=strategy_colors['Regime-Based EMA'])

    ax2.set_title('Drawdowns (Testing Period)', fontsize=12, pad=20)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(fontsize=10, loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Add regime changes as background shading for test period only
    if 'regime_changes' in results_dict['Optimized']:
        regimes = results_dict['Optimized']['regime_changes'][test_mask]
        last_idx = regimes.index[0]
        last_regime = regimes.iloc[0]
        
        for i, (idx, regime) in enumerate(regimes.items()):
            if regime != last_regime:
                if last_regime == 'high_volatility':
                    ax1.axvspan(last_idx, idx, alpha=0.2, color='red',
                              label='High Volatility' if i==1 else "")
                    ax2.axvspan(last_idx, idx, alpha=0.2, color='red')
                last_idx = idx
                last_regime = regime
        
        # Handle the last period
        if last_regime == 'high_volatility':
            ax1.axvspan(last_idx, regimes.index[-1], alpha=0.2, color='red')
            ax2.axvspan(last_idx, regimes.index[-1], alpha=0.2, color='red')

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
        
        for symbol in BACKTEST_CONFIG['symbols']:
            print(f"\n{'='*50}")
            print(f"Processing {symbol}...")
            print(f"{'='*50}")

            # Fetch and prepare data
            data = db.get_historical_data(symbol, BACKTEST_CONFIG['start_date'], BACKTEST_CONFIG['end_date'])
            if len(data) == 0:
                print(f"No data found for {symbol}")
                continue
                
            # Add technical indicators
            data_with_indicators = TechnicalIndicators.add_all_indicators(data.copy())
            
            # Get training/testing periods
            test_start = BACKTEST_CONFIG['optimization']['testing_start']
            test_end = BACKTEST_CONFIG['optimization']['testing_end']
            train_start, train_end = calculate_training_period(
                test_start, 
                BACKTEST_CONFIG['optimization']['training_days']
            )
            
            print(f"\nTraining period: {train_start} to {train_end}")
            print(f"Testing period: {test_start} to {test_end}")
            
            # Optimize strategy parameters
            optimization_results = optimize_strategy_parameters(
                data_with_indicators, symbol, train_start, train_end
            )
            
            # Initialize and run the regime-based EMA strategy
            regime_strategy = OptimizedStrategy(optimization_results)
            dynamic_signals = regime_strategy.run_dynamic(data_with_indicators)
            
            # Create results dictionary with only optimized strategy
            results = {}
            backtester = Backtester(data_with_indicators, "Optimized", lambda df: dynamic_signals)
            results['Optimized'] = backtester.run()

            results['Optimized']['regime_changes'] = dynamic_signals.regime_changes

            
            # Save test period results
            test_period_file = save_test_period_results(results, symbol, output_dir, train_end)
            print(f"✓ Test period results saved to {test_period_file}")
            
            # Plot test period results
            if BACKTEST_CONFIG['save_plots']:

                # Full period plot
                plot_results(results, symbol, output_dir)
                print("✓ Full period plot saved")

                plot_test_period_results(results, symbol, output_dir, test_start, test_end)
                print("✓ Test period plot saved")
                
    except Exception as e:
        print(f"\nError during backtesting: {str(e)}")
        raise
    finally:
        print("\nClosing database connection...")
        db.close()

if __name__ == "__main__":
    main()
