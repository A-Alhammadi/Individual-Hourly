# ema.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import psycopg2
import os

# Configuration
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "cryptocurrencies",
    "user": "myuser",
    "password": "mypassword"
}

# Strategy Parameters
STRATEGY_PARAMS = {
    "market_conditions": {
        "high_volatility": {
            "ema": {
                "parameters": {
                    "ema": {
                        "short": 5,
                        "medium": 15,
                        "volatility_window": 60,
                        "volatility_threshold": 2.0,
                        "min_trend_strength": 0.01
                    },
                    "volatility": {
                        "annualization_factor": 8760,
                        "baseline_window_multiplier": 2,
                        "baseline_lookback_gap": 5,
                        "min_periods_multiplier": 0.75
                    }
                }
            }
        },
        "normal": {
            "ema": {
                "parameters": {
                    "ema": {
                        "short": 15,
                        "medium": 30,
                        "volatility_window": 40,
                        "volatility_threshold": 1.2,
                        "min_trend_strength": 0.02
                    },
                    "volatility": {
                        "annualization_factor": 8760,
                        "baseline_window_multiplier": 1.5,
                        "baseline_lookback_gap": 5,
                        "min_periods_multiplier": 0.75
                    }
                }
            }
        }
    }
}

# Backtest Configuration
BACKTEST_CONFIG = {
    "symbol": "BTC/USD",  # Change this to test different symbols
    "start_date": "2024-01-01",  # Change this to set test period start
    "end_date": "2024-12-01",    # Change this to set test period end
    "initial_capital": 10000,
    "position_size": 1.0,
    "trading_fee": 0.001
}

class EMAStrategy:
    def __init__(self, df, strategy_params):
        self.df = df.copy()
        self.params = strategy_params
    
    def verify_signal(self, signal, price):
        """Verify that a signal is valid"""
        if signal not in [-1, 0, 1]:
            print(f"Warning: Invalid signal {signal} detected!")
            return 0
        if not isinstance(price, (int, float)) or pd.isna(price):
            print(f"Warning: Invalid price {price} detected!")
            return 0
        return signal
        
    def detect_market_regime(self, i, params):
        """Detect if we're in high volatility or normal regime"""
        vol_window = params['ema']['volatility_window']
        vol_threshold = params['ema']['volatility_threshold']
        
        # Get volatility configuration
        vol_config = params['volatility']
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
        returns = self.df['close_price'].pct_change()
        
        # Current volatility window
        current_start = i - vol_window + 1  # Include current period
        current_end = i + 1
        current_returns = returns.iloc[current_start:current_end]
        if len(current_returns) >= min_periods:
            current_vol = current_returns.std() * np.sqrt(annual_factor)
        else:
            return 'normal'
        
        # Baseline volatility window
        baseline_start = i - (baseline_window + vol_window + lookback_gap) + 1
        baseline_end = i - (vol_window + lookback_gap) + 1
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

    def generate_signals(self):
        """Generate trading signals based on regime-specific EMAs"""
        signals = pd.Series(0.0, index=self.df.index)
        regime_changes = pd.Series(index=self.df.index, dtype='object')
        
        filtered_trades = 0
        potential_trades = 0
        position = 0  # Track current position
        
        print("\nGenerating signals...")
        
        # Debug counters
        crossovers_normal = 0
        crossovers_high_vol = 0
        strength_filtered = 0
        position_filtered = 0
        signals_generated = 0
        
        for i in range(len(self.df)):
            # Get the normal regime parameters for regime detection
            normal_params = self.params['market_conditions']['normal']['ema']['parameters']
            
            # Detect current regime using normal parameters
            regime = self.detect_market_regime(i, normal_params)
            regime_changes.iloc[i] = regime
            
            # Get regime-specific parameters for trading
            params = self.params['market_conditions'][regime]['ema']['parameters']['ema']
            
            # Calculate EMAs up to current bar
            current_data = self.df.iloc[:i+1]['close_price']
            short_ema = current_data.ewm(span=params['short'], adjust=False).mean()
            medium_ema = current_data.ewm(span=params['medium'], adjust=False).mean()
            
            # Calculate trend strength
            trend_strength = abs(short_ema.iloc[-1] - medium_ema.iloc[-1]) / medium_ema.iloc[-1]
            
            # Check for potential signal
            potential_signal = 0
            
            if short_ema.iloc[-1] > medium_ema.iloc[-1]:
                potential_signal = 1
                if regime == 'high_volatility':
                    crossovers_high_vol += 1
                else:
                    crossovers_normal += 1
            elif short_ema.iloc[-1] < medium_ema.iloc[-1]:
                potential_signal = -1
                if regime == 'high_volatility':
                    crossovers_high_vol += 1
                else:
                    crossovers_normal += 1
                    
            if potential_signal != 0:
                potential_trades += 1
                
                # Print debug info every hour
                if i % 24 == 0:
                    print(f"\nBar {i} - {self.df.index[i]}:")
                    print(f"Regime: {regime}")
                    print(f"EMAs - Short: {short_ema.iloc[-1]:.2f}, Medium: {medium_ema.iloc[-1]:.2f}")
                    print(f"Trend Strength: {trend_strength:.4f} (Min Required: {params['min_trend_strength']:.4f})")
                    print(f"Position: {position}, Potential Signal: {potential_signal}")
                    print(f"Close Price: {self.df['close_price'].iloc[i]:.2f}")
                
                # Only generate signal if trend is strong enough and would change position
                if trend_strength >= params['min_trend_strength']:
                    if (potential_signal == 1 and position <= 0) or (potential_signal == -1 and position >= 0):
                        verified_signal = self.verify_signal(potential_signal, self.df['close_price'].iloc[i])
                        signals.iloc[i] = verified_signal
                        if verified_signal != 0:
                            position = verified_signal
                            signals_generated += 1
                        
                            print(f"\nSignal Generated at {self.df.index[i]}:")
                            print(f"Type: {'BUY' if potential_signal == 1 else 'SELL'}")
                            print(f"Price: {self.df['close_price'].iloc[i]:.2f}")
                            print(f"Trend Strength: {trend_strength:.4f}")
                            print(f"Regime: {regime}")
                    else:
                        position_filtered += 1
                else:
                    strength_filtered += 1
                    filtered_trades += 1
        
        print(f"\nSignal Generation Detailed Statistics:")
        print(f"Normal Regime Crossovers: {crossovers_normal}")
        print(f"High Volatility Crossovers: {crossovers_high_vol}")
        print(f"Signals Filtered by Trend Strength: {strength_filtered}")
        print(f"Signals Filtered by Position: {position_filtered}")
        print(f"Final Signals Generated: {signals_generated}")
        print(f"  Buy Signals: {(signals == 1).sum()}")
        print(f"  Sell Signals: {(signals == -1).sum()}")
        
        # Verify final signal counts match
        total_signals = (signals == 1).sum() + (signals == -1).sum()
        if total_signals != signals_generated:
            print(f"Warning: Signal count mismatch! Generated: {signals_generated}, Total: {total_signals}")
        
        signals.regime_changes = regime_changes
        return signals

def backtest_strategy(df, signals, config):
    """Run backtest and calculate metrics - Long only positions"""
    portfolio = pd.DataFrame(index=df.index)
    portfolio['holdings'] = 0.0
    portfolio['cash'] = float(config['initial_capital'])
    portfolio['close'] = df['close_price']
    portfolio['total_value'] = portfolio['cash']  # Initialize total value
    
    position = 0
    trades = []
    
    print("\nExecuting backtest...")
    
    for i in range(len(portfolio)):
        if i > 0:
            # Carry forward previous values
            portfolio.loc[portfolio.index[i], 'cash'] = portfolio.loc[portfolio.index[i-1], 'cash']
            portfolio.loc[portfolio.index[i], 'holdings'] = portfolio.loc[portfolio.index[i-1], 'holdings']
            
            # Update holdings value before processing any new trades
            current_price = float(portfolio.loc[portfolio.index[i], 'close'])
            portfolio.loc[portfolio.index[i], 'holdings_value'] = portfolio.loc[portfolio.index[i], 'holdings'] * current_price
        
        signal = signals.iloc[i]
        
        if signal == 1 and position == 0:  # Buy signal - only enter when not in position
            # Open long position
            available_capital = float(portfolio.loc[portfolio.index[i], 'cash'])
            position_value = available_capital * config['position_size']
            units = position_value / float(portfolio.loc[portfolio.index[i], 'close'])
            
            fee = position_value * config['trading_fee']
            
            portfolio.loc[portfolio.index[i], 'holdings'] = float(units)
            portfolio.loc[portfolio.index[i], 'cash'] = float(available_capital - (position_value + fee))
            position = 1
            
            trades.append({
                'date': portfolio.index[i],
                'type': 'BUY',
                'price': float(portfolio.loc[portfolio.index[i], 'close']),
                'units': float(units),
                'value': float(position_value),
                'fee': float(fee)
            })
            
            print(f"\nBUY Trade at {portfolio.index[i]}:")
            print(f"Price: ${portfolio.loc[portfolio.index[i], 'close']:.2f}")
            print(f"Units: {units:.6f}")
            print(f"Value: ${position_value:.2f}")
            print(f"Fee: ${fee:.2f}")
            
        elif signal == -1 and position == 1:  # Sell signal - only exit when in position
            # Close long position
            units = float(portfolio.loc[portfolio.index[i], 'holdings'])
            position_value = units * float(portfolio.loc[portfolio.index[i], 'close'])
            fee = position_value * config['trading_fee']
            
            portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
            portfolio.loc[portfolio.index[i], 'cash'] = float(portfolio.loc[portfolio.index[i], 'cash'] + position_value - fee)
            position = 0
            
            trades.append({
                'date': portfolio.index[i],
                'type': 'SELL',
                'price': float(portfolio.loc[portfolio.index[i], 'close']),
                'units': float(units),
                'value': float(position_value),
                'fee': float(fee)
            })
            
            print(f"\nSELL Trade at {portfolio.index[i]}:")
            print(f"Price: ${portfolio.loc[portfolio.index[i], 'close']:.2f}")
            print(f"Units: {units:.6f}")
            print(f"Value: ${position_value:.2f}")
            print(f"Fee: ${fee:.2f}")
        
        # Update total value
        current_price = float(portfolio.loc[portfolio.index[i], 'close'])
        portfolio.loc[portfolio.index[i], 'holdings_value'] = portfolio.loc[portfolio.index[i], 'holdings'] * current_price
        portfolio.loc[portfolio.index[i], 'total_value'] = (
            portfolio.loc[portfolio.index[i], 'cash'] + 
            portfolio.loc[portfolio.index[i], 'holdings_value']
        )
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    if not trades_df.empty:
        print(f"\nBacktest completed with {len(trades_df)} trades")
    else:
        print("\nBacktest completed with no trades")
    
    return portfolio, trades_df

def plot_results(portfolio_df, signals, symbol, output_dir='results'):
    """Plot strategy results"""
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Calculate Buy & Hold
    initial_capital = float(BACKTEST_CONFIG['initial_capital'])  # Make sure we use initial capital
    initial_price = portfolio_df['close'].iloc[0]
    buy_hold_units = initial_capital / initial_price
    buy_hold_values = portfolio_df['close'] * buy_hold_units
    
    # Debug prints
    print("\nPlotting Debug Info:")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Initial Price: ${initial_price:.2f}")
    print(f"Buy & Hold Units: {buy_hold_units:.6f}")
    print(f"Strategy Initial Value: ${portfolio_df['total_value'].iloc[0]:.2f}")
    print(f"Strategy Final Value: ${portfolio_df['total_value'].iloc[-1]:.2f}")
    
    # Plot portfolio values
    ax1.plot(portfolio_df.index, buy_hold_values,
             label=f'Buy & Hold (${buy_hold_values.iloc[-1]:,.0f})',
             linewidth=1.5, color='black', linestyle='--', alpha=0.7)
    
    ax1.plot(portfolio_df.index, portfolio_df['total_value'],
             label=f'EMA Strategy (${portfolio_df["total_value"].iloc[-1]:,.0f})',
             linewidth=2.0, color='blue')
    
    ax1.set_title(f'{symbol} - Portfolio Value', fontsize=12, pad=20)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot drawdowns
    bh_dd = (buy_hold_values.cummax() - buy_hold_values) / buy_hold_values.cummax()
    strategy_dd = (portfolio_df['total_value'].cummax() - portfolio_df['total_value']) / portfolio_df['total_value'].cummax()
    
    ax2.plot(portfolio_df.index, bh_dd * 100,  # Convert to percentage
             label=f'Buy & Hold (Max DD: {bh_dd.min()*100:.1f}%)',
             linewidth=1.5, color='black', linestyle='--', alpha=0.7)
    
    ax2.plot(portfolio_df.index, strategy_dd * 100,  # Convert to percentage
             label=f'EMA Strategy (Max DD: {strategy_dd.min()*100:.1f}%)',
             linewidth=2.0, color='blue')
    
    ax2.set_title('Drawdowns', fontsize=12, pad=20)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(fontsize=10, loc='lower left')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add regime changes as background shading
    regimes = signals.regime_changes
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
    
    if last_regime == 'high_volatility':
        ax1.axvspan(last_idx, regimes.index[-1], alpha=0.2, color='red')
        ax2.axvspan(last_idx, regimes.index[-1], alpha=0.2, color='red')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_')
    plot_file = os.path.join(output_dir, f'{safe_symbol}_results_{timestamp}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults plot saved to: {plot_file}")

# Add this new function to save detailed results:
def save_detailed_results(portfolio_df, trades_df, signals, symbol, output_dir='results'):
    """Save detailed backtest results to a text file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace('/', '_')
    results_file = os.path.join(output_dir, f'{safe_symbol}_detailed_results_{timestamp}.txt')
    
    initial_capital = float(BACKTEST_CONFIG['initial_capital'])
    
    with open(results_file, 'w') as f:
        f.write(f"=== EMA Strategy Backtest Results ===\n")
        f.write(f"Symbol: {symbol}\n")
        f.write(f"Period: {BACKTEST_CONFIG['start_date']} to {BACKTEST_CONFIG['end_date']}\n")
        f.write(f"Initial Capital: ${initial_capital:,.2f}\n\n")
        
        # Calculate and write returns
        total_return = (portfolio_df['total_value'].iloc[-1] - initial_capital) / initial_capital
        bh_return = (portfolio_df['close'].iloc[-1] - portfolio_df['close'].iloc[0]) / portfolio_df['close'].iloc[0]
        
        f.write("=== Performance Metrics ===\n")
        f.write(f"Final Portfolio Value: ${portfolio_df['total_value'].iloc[-1]:,.2f}\n")
        f.write(f"Total Return: {total_return:.2%}\n")
        f.write(f"Buy & Hold Return: {bh_return:.2%}\n")
        
        # Daily returns and risk metrics
        daily_returns = portfolio_df['total_value'].pct_change()
        sharpe = np.sqrt(365) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
        volatility = daily_returns.std() * np.sqrt(365)
        
        f.write(f"Annualized Volatility: {volatility:.2%}\n")
        f.write(f"Sharpe Ratio: {sharpe:.2f}\n")
        
        # Drawdown analysis
        drawdown = (portfolio_df['total_value'].cummax() - portfolio_df['total_value']) / portfolio_df['total_value'].cummax()
        f.write(f"Maximum Drawdown: {drawdown.max():.2%}\n\n")
        
        # Trade analysis
        if not trades_df.empty:
            f.write("=== Trading Statistics ===\n")
            num_trades = len(trades_df)
            num_buys = len(trades_df[trades_df['type'] == 'BUY'])
            num_sells = len(trades_df[trades_df['type'] == 'SELL'])
            
            f.write(f"Total Number of Trades: {num_trades}\n")
            f.write(f"Number of Buys: {num_buys}\n")
            f.write(f"Number of Sells: {num_sells}\n")
            
            if num_trades > 1:
                win_rate = len(trades_df[trades_df['value'] > trades_df['value'].shift(1)]) / (num_trades // 2)
                f.write(f"Win Rate: {win_rate:.2%}\n")
            
            f.write(f"Average Trade Size: ${trades_df['value'].mean():,.2f}\n")
            f.write(f"Total Trading Fees: ${trades_df['fee'].sum():,.2f}\n")
            
            if num_trades > 1:
                avg_duration = trades_df['date'].diff().mean()
                f.write(f"Average Trade Duration: {avg_duration}\n\n")
        
        # Regime analysis
        f.write("=== Regime Statistics ===\n")
        regimes = signals.regime_changes
        high_vol_periods = (regimes == 'high_volatility').sum()
        total_periods = len(regimes)
        
        f.write(f"Total Trading Periods: {total_periods}\n")
        f.write(f"High Volatility Periods: {high_vol_periods} ({high_vol_periods/total_periods:.1%})\n")
        f.write(f"Normal Volatility Periods: {total_periods-high_vol_periods} ({1-high_vol_periods/total_periods:.1%})\n")
        
        # Parameters used
        f.write("\n=== Strategy Parameters ===\n")
        f.write("High Volatility Regime:\n")
        f.write(f"  EMA Parameters: {STRATEGY_PARAMS['market_conditions']['high_volatility']['ema']['parameters']['ema']}\n")
        f.write(f"  Volatility Parameters: {STRATEGY_PARAMS['market_conditions']['high_volatility']['ema']['parameters']['volatility']}\n\n")
        
        f.write("Normal Regime:\n")
        f.write(f"  EMA Parameters: {STRATEGY_PARAMS['market_conditions']['normal']['ema']['parameters']['ema']}\n")
        f.write(f"  Volatility Parameters: {STRATEGY_PARAMS['market_conditions']['normal']['ema']['parameters']['volatility']}\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    return results_file


def get_data_from_db(symbol, start_date, end_date):
    """Fetch historical data from database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        query = """
            SELECT 
                date_time,
                open_price,
                high_price,
                low_price,
                close_price,
                volume_crypto,
                volume_usd
            FROM crypto_data_hourly
            WHERE symbol = %s
            AND date_time BETWEEN %s AND %s
            ORDER BY date_time ASC
        """
        
        df = pd.read_sql_query(
            query,
            conn,
            params=(symbol, start_date, end_date),
            parse_dates=['date_time']
        )
        
        df.set_index('date_time', inplace=True)
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        if conn:
            conn.close()
        raise

def calculate_metrics(portfolio_df, trades_df, initial_capital):
    """Calculate and print strategy metrics"""
    # Returns
    total_return = (portfolio_df['total_value'].iloc[-1] - initial_capital) / initial_capital
    
    # Buy & Hold return
    bh_return = (portfolio_df['close'].iloc[-1] - portfolio_df['close'].iloc[0]) / portfolio_df['close'].iloc[0]
    
    # Daily metrics
    daily_returns = portfolio_df['total_value'].pct_change()
    sharpe = np.sqrt(365) * (daily_returns.mean() / daily_returns.std()) if daily_returns.std() != 0 else 0
    
    # Drawdown
    drawdown = (portfolio_df['total_value'].cummax() - portfolio_df['total_value']) / portfolio_df['total_value'].cummax()
    max_drawdown = drawdown.max()
    
    # Print metrics
    print("\nStrategy Metrics:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Buy & Hold Return: {bh_return:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    
    if not trades_df.empty:
        num_trades = len(trades_df)
        win_rate = len(trades_df[trades_df['value'] > trades_df['value'].shift(1)]) / (num_trades // 2) if num_trades > 1 else 0
        total_fees = trades_df['fee'].sum()
        
        print(f"\nTrading Statistics:")
        print(f"Number of Trades: {num_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Trading Fees: ${total_fees:.2f}")
        print(f"Average Trade Size: ${trades_df['value'].mean():.2f}")
        
        if num_trades > 1:
            avg_duration = trades_df['date'].diff().mean()
            print(f"Average Trade Duration: {avg_duration}")

def main():
    try:
        print(f"Starting backtest for {BACKTEST_CONFIG['symbol']}")
        print(f"Period: {BACKTEST_CONFIG['start_date']} to {BACKTEST_CONFIG['end_date']}")
        
        # Get data
        df = get_data_from_db(
            BACKTEST_CONFIG['symbol'],
            BACKTEST_CONFIG['start_date'],
            BACKTEST_CONFIG['end_date']
        )
        
        if len(df) == 0:
            print("No data found for the specified period")
            return
            
        print(f"\nData loaded: {len(df)} periods")
        
        # Initialize strategy
        strategy = EMAStrategy(df, STRATEGY_PARAMS)
        
        # Generate signals
        signals = strategy.generate_signals()
        
        # Run backtest
        portfolio, trades = backtest_strategy(df, signals, BACKTEST_CONFIG)
        
        # Calculate metrics
        calculate_metrics(portfolio, trades, BACKTEST_CONFIG['initial_capital'])
        
        # Save detailed results
        save_detailed_results(portfolio, trades, signals, BACKTEST_CONFIG['symbol'])
        
        # Plot results
        plot_results(portfolio, signals, BACKTEST_CONFIG['symbol'])
        
    except Exception as e:
        print(f"\nError during backtest: {str(e)}")
        raise

if __name__ == "__main__":
    main()
