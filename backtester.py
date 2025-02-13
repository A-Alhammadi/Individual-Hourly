# backtester.py

import pandas as pd
import numpy as np
from config import BACKTEST_CONFIG
from indicators import TechnicalIndicators

class Backtester:
    def __init__(self, df, strategy_name, strategy_func):
        self.df = df.copy()
        self.strategy_name = strategy_name
        self.strategy_func = strategy_func
        self.initial_capital = float(BACKTEST_CONFIG['initial_capital'])
        
        # This is your *base* position size; you will multiply it by position_scale
        self.position_size = float(BACKTEST_CONFIG['position_size'])
        
        self.trading_fee = float(BACKTEST_CONFIG['trading_fee'])
        
        # Map price column names
        self.price_columns = {
            'close': 'close_price' if 'close_price' in df.columns else 'close',
            'high': 'high_price' if 'high_price' in df.columns else 'high',
            'low': 'low_price' if 'low_price' in df.columns else 'low',
            'open': 'open_price' if 'open_price' in df.columns else 'open'
        }

    def run(self):
        """Run backtest with the selected strategy"""
        print("\nRunning backtest...")
        print(f"Initial capital: ${self.initial_capital}")
        
        # Get trading signals
        df_copy = self.df.copy()  # Make an explicit copy
        df_copy = TechnicalIndicators.add_all_indicators(df_copy)  # Add indicators here
        signals = self.strategy_func(df_copy)        

        # Initialize portfolio metrics with float dtype
        portfolio = pd.DataFrame(index=self.df.index)
        portfolio['holdings'] = 0.0
        portfolio['cash'] = float(self.initial_capital)
        portfolio['signal'] = signals

        # Copy price data to portfolio
        for col_type, col_name in self.price_columns.items():
            portfolio[col_type] = self.df[col_name]

        # Also copy position_scale if it exists (created in add_adaptive_ema)
        if 'position_scale' in self.df.columns:
            portfolio['position_scale'] = self.df['position_scale']
        else:
            # Default to 1.0 if not present
            portfolio['position_scale'] = 1.0
        
        # Ensure float dtype for numerical columns
        portfolio = portfolio.astype({
            'holdings': 'float64',
            'cash': 'float64',
            'close': 'float64',
            'high': 'float64',
            'low': 'float64',
            'open': 'float64',
            'position_scale': 'float64'
        })

        position = 0
        trades = []

        for i in range(len(portfolio)):

            price = float(portfolio.loc[portfolio.index[i], 'close'])

            if i > 0:
                portfolio.loc[portfolio.index[i], 'cash'] = portfolio.loc[portfolio.index[i-1], 'cash']
                portfolio.loc[portfolio.index[i], 'holdings'] = portfolio.loc[portfolio.index[i-1], 'holdings']

            signal = portfolio.iloc[i, portfolio.columns.get_loc('signal')]

            # We fetch the position scale for this bar:
            scale_factor = portfolio.iloc[i, portfolio.columns.get_loc('position_scale')]
            
            # 1) If the signal is > 0, we treat that as "BUY or COVER"
            if signal > 0:
                # If position == 0, we open a new long position
                if position == 0:  # Open new long
                    available_capital = float(portfolio.loc[portfolio.index[i], 'cash'])

                    # Ensure position_value does not exceed available capital
                    position_value = min(available_capital, available_capital * self.position_size * scale_factor)

                    units = position_value / float(portfolio.loc[portfolio.index[i], 'close'])
                    fee = position_value * self.trading_fee

                    portfolio.loc[portfolio.index[i], 'holdings'] = float(units)
                    portfolio.loc[portfolio.index[i], 'cash'] = float(available_capital - (position_value + fee))
                    position = 1

                    trades.append({
                        'date': portfolio.index[i],
                        'type': 'BUY',
                        'price': price,
                        'units': float(units),
                        'value': float(position_value),
                        'fee': float(fee)
                    })

                # If position == -1, then signal > 0 means "COVER" (close short)
                elif position == -1:
                    units = float(portfolio.loc[portfolio.index[i], 'holdings'])  # negative
                    price = float(portfolio.loc[portfolio.index[i], 'close'])
                    position_value = abs(units) * price

                    fee = position_value * self.trading_fee

                    portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
                    portfolio.loc[portfolio.index[i], 'cash'] = float(
                        portfolio.loc[portfolio.index[i], 'cash'] - position_value - fee
                    )

                    position = 0

                    trades.append({
                        'date': portfolio.index[i],
                        'type': 'COVER',
                        'price': price,
                        'units': float(units),  # negative means we were short
                        'value': float(position_value),
                        'fee': float(fee)
                    })

            # 2) If the signal is < 0, we treat that as "SELL or SHORT"
            elif signal < 0:
                # If position == 0, we open a new short
                if position == 0:  # Open new short
                    available_capital = float(portfolio.loc[portfolio.index[i], 'cash'])

                    # Ensure position_value does not exceed available capital
                    position_value = min(available_capital, available_capital * self.position_size * scale_factor)

                    units = position_value / float(portfolio.loc[portfolio.index[i], 'close'])
                    fee = position_value * self.trading_fee

                    portfolio.loc[portfolio.index[i], 'holdings'] = float(-units)  # Negative for shorts
                    portfolio.loc[portfolio.index[i], 'cash'] = float(available_capital + position_value - fee)
                    position = -1

                    trades.append({
                        'date': portfolio.index[i],
                        'type': 'SHORT',
                        'price': price,
                        'units': float(-units),
                        'value': float(position_value),
                        'fee': float(fee)
                    })

                # If position == 1, then signal < 0 means "SELL" (close long)
                elif position == 1:
                    units = float(portfolio.loc[portfolio.index[i], 'holdings'])
                    price = float(portfolio.loc[portfolio.index[i], 'close'])
                    position_value = units * price

                    fee = position_value * self.trading_fee

                    portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
                    portfolio.loc[portfolio.index[i], 'cash'] = float(
                        portfolio.loc[portfolio.index[i], 'cash'] + position_value - fee
                    )
                    position = 0

                    trades.append({
                        'date': portfolio.index[i],
                        'type': 'SELL',
                        'price': price,
                        'units': float(units),
                        'value': float(position_value),
                        'fee': float(fee)
                    })

        # Calculate portfolio value
        portfolio['holdings_value'] = portfolio['holdings'] * portfolio['close'].astype(float)
        portfolio['total_value'] = portfolio['cash'] + portfolio['holdings_value']

        # Calculate hourly returns
        portfolio['hourly_returns'] = portfolio['total_value'].pct_change()

        # Calculate metrics
        total_return = (portfolio['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        buy_and_hold_return = (
            float(self.df[self.price_columns['close']].iloc[-1]) -
            float(self.df[self.price_columns['close']].iloc[0])
        ) / float(self.df[self.price_columns['close']].iloc[0])

        # Calculate annualized metrics using hourly data
        hours_in_period = len(portfolio)
        hours_per_year = 365 * 24
        annual_return = total_return * (hours_per_year / hours_in_period)

        # Calculate Sharpe ratio using hourly data
        if portfolio['hourly_returns'].std() > 0:
            hourly_sharpe = np.sqrt(hours_per_year) * (
                portfolio['hourly_returns'].mean() / portfolio['hourly_returns'].std()
            )
        else:
            hourly_sharpe = 0

        max_drawdown = (
            (portfolio['total_value'].cummax() - portfolio['total_value']) /
            portfolio['total_value'].cummax()
        ).max()

        trades_df = pd.DataFrame(trades)
        num_trades = len(trades_df)

        # Calculate win rate
        if num_trades > 1:
            paired_trades = trades_df.iloc[::2]  # Entry trades (every other row)
            exit_trades = trades_df.iloc[1::2]   # Exit trades

            # Ensure both lists have the same length before comparing
            min_length = min(len(paired_trades), len(exit_trades))

            if min_length > 0:
                wins = sum(
                    exit_trades['value'].reset_index(drop=True).iloc[:min_length].to_numpy()
                    >
                    paired_trades['value'].reset_index(drop=True).iloc[:min_length].to_numpy()
                )
                win_rate = wins / min_length
            else:
                win_rate = 0  # If no valid trades, win rate is 0
        else:
            win_rate = 0


        # Prepare overall metrics
        metrics = {
            'Strategy': self.strategy_name,
            'Total Return': f"{total_return * 100:.2f}%",
            'Buy and Hold Return': f"{buy_and_hold_return * 100:.2f}%",
            'Annual Return': f"{annual_return * 100:.2f}%",
            'Sharpe Ratio': f"{hourly_sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown * 100:.2f}%",
            'Trading Statistics': {
                'Number of Trades': num_trades,
                'Win Rate': f"{win_rate * 100:.2f}%",
                'Average Trade Duration': (
                    f"{trades_df['date'].diff().mean().total_seconds() / 3600:.1f} hours"
                    if num_trades > 1 else "N/A"
                ),
                'Average Trade Size': (
                    f"${trades_df['value'].mean():.2f}"
                    if num_trades > 0 else "N/A"
                ),
                'Total Trading Fees': (
                    f"${trades_df['fee'].sum():.2f}"
                    if num_trades > 0 else "$0.00"
                )
            }
        }
        # Add trade analysis by regime if regime changes are available
        if hasattr(signals, 'regime_changes'):
            trades_df = trades_df.reset_index(drop=True)  # Just reset_index, no rename
            # do NOT rename the 'index' column to 'date' now

            regime_trades = pd.merge(
                trades_df,
                signals.regime_changes, 
                left_on='date',          # The trades_df already has 'date' from the dictionary
                right_index=True, 
                how='left'
            )

            for regime in ['high_volatility', 'normal']:
                regime_stats = regime_trades[regime_trades['regime'] == regime]
                if not regime_stats.empty:
                    paired_regime_trades = regime_stats.iloc[::2]  # Entry trades
                    exit_regime_trades = regime_stats.iloc[1::2]   # Exit trades
                    if len(paired_regime_trades) > 0 and len(exit_regime_trades) > 0:
                        min_length = min(len(paired_regime_trades), len(exit_regime_trades))  # Ensure equal length
                        regime_wins = sum(
                            exit_regime_trades['value'].reset_index(drop=True).iloc[:min_length].to_numpy()
                            >
                            paired_regime_trades['value'].reset_index(drop=True).iloc[:min_length].to_numpy()
                        )
                        regime_win_rate = regime_wins / min_length  # Use min_length to avoid division errors
                    else:
                        regime_win_rate = 0  # Default to 0 if no valid trades

                    metrics[f'{regime.title()} Regime'] = {
                        'Number of Trades': len(regime_stats),
                        'Win Rate': f"{regime_win_rate * 100:.2f}%",
                        'Average Trade Size': f"${regime_stats['value'].mean():.2f}"
                    }

        return {
            'metrics': metrics,
            'portfolio': portfolio,
            'trades': trades_df if num_trades > 0 else pd.DataFrame()
        }
