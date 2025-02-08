#backtester.py

import pandas as pd
import numpy as np
from config import BACKTEST_CONFIG

class Backtester:
    def __init__(self, df, strategy_name, strategy_func):
        self.df = df.copy()
        self.strategy_name = strategy_name
        self.strategy_func = strategy_func
        self.initial_capital = float(BACKTEST_CONFIG['initial_capital'])
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
        # Get trading signals
        signals = self.strategy_func(self.df)
        
        # Initialize portfolio metrics with float dtype
        portfolio = pd.DataFrame(index=self.df.index)
        portfolio['holdings'] = 0.0
        portfolio['cash'] = float(self.initial_capital)
        portfolio['signal'] = signals
        
        # Copy price data to portfolio
        for col_type, col_name in self.price_columns.items():
            portfolio[col_type] = self.df[col_name]
        
        # Ensure float dtype for numerical columns
        portfolio = portfolio.astype({
            'holdings': 'float64', 
            'cash': 'float64',
            'close': 'float64',
            'high': 'float64',
            'low': 'float64',
            'open': 'float64'
        })
        
        position = 0
        trades = []
        
        for i in range(len(portfolio)):
            if i > 0:
                portfolio.loc[portfolio.index[i], 'cash'] = portfolio.loc[portfolio.index[i-1], 'cash']
                portfolio.loc[portfolio.index[i], 'holdings'] = portfolio.loc[portfolio.index[i-1], 'holdings']
            
            signal = portfolio.iloc[i, portfolio.columns.get_loc('signal')]
            
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size
                available_capital = float(portfolio.loc[portfolio.index[i], 'cash'])
                position_value = available_capital * self.position_size
                units = position_value / float(portfolio.loc[portfolio.index[i], 'close'])
                
                # Apply trading fee
                fee = position_value * self.trading_fee
                
                # Update portfolio using loc
                portfolio.loc[portfolio.index[i], 'holdings'] = float(units)
                portfolio.loc[portfolio.index[i], 'cash'] = float(available_capital - (position_value + fee))
                position = 1
                
                # Record trade
                trades.append({
                    'date': portfolio.index[i],
                    'type': 'BUY',
                    'price': float(portfolio.loc[portfolio.index[i], 'close']),
                    'units': float(units),
                    'value': float(position_value),
                    'fee': float(fee)
                })
                
            elif signal == -1 and position == 1:  # Sell signal
                # Calculate position value
                units = float(portfolio.loc[portfolio.index[i], 'holdings'])
                position_value = units * float(portfolio.loc[portfolio.index[i], 'close'])
                
                # Apply trading fee
                fee = position_value * self.trading_fee
                
                # Update portfolio using loc
                portfolio.loc[portfolio.index[i], 'holdings'] = 0.0
                portfolio.loc[portfolio.index[i], 'cash'] = float(portfolio.loc[portfolio.index[i], 'cash'] + position_value - fee)
                position = 0
                
                # Record trade
                trades.append({
                    'date': portfolio.index[i],
                    'type': 'SELL',
                    'price': float(portfolio.loc[portfolio.index[i], 'close']),
                    'units': float(units),
                    'value': float(position_value),
                    'fee': float(fee)
                })
        
        # Calculate portfolio value
        portfolio['holdings_value'] = portfolio['holdings'] * self.df['close_price'].astype(float)
        portfolio['total_value'] = portfolio['cash'] + portfolio['holdings_value']
        
        # Calculate metrics
        total_return = (portfolio['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        buy_and_hold_return = (float(self.df['close_price'].iloc[-1]) - float(self.df['close_price'].iloc[0])) / float(self.df['close_price'].iloc[0])
        
        # Calculate daily returns and metrics
        portfolio['daily_returns'] = portfolio['total_value'].pct_change()
        
        annual_return = total_return * (365 / len(portfolio))
        sharpe_ratio = np.sqrt(365) * (portfolio['daily_returns'].mean() / portfolio['daily_returns'].std())
        max_drawdown = ((portfolio['total_value'].cummax() - portfolio['total_value']) / portfolio['total_value'].cummax()).max()
        
        trades_df = pd.DataFrame(trades)
        num_trades = len(trades)
        win_rate = len(trades_df[trades_df['value'] > trades_df['value'].shift(1)]) / (num_trades // 2) if num_trades > 0 else 0
        
        # Calculate metrics with regime-specific focus
        metrics = {
            'Strategy': self.strategy_name,
            'Total Return': f"{total_return * 100:.2f}%",
            'Buy and Hold Return': f"{buy_and_hold_return * 100:.2f}%",
            'Annual Return': f"{annual_return * 100:.2f}%",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
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
                'Total Trading Fees': f"${trades_df['fee'].sum():.2f}" if num_trades > 0 else "$0.00"
            }
        }

        # Add trade analysis by regime if regime changes are available
        if hasattr(self.strategy_func, 'regime_changes'):
            regime_trades = pd.merge(
                trades_df,
                self.strategy_func.regime_changes,
                left_on='date',
                right_index=True,
                how='left'
            )
            
            for regime in ['high_volatility', 'normal']:
                regime_stats = regime_trades[regime_trades['regime'] == regime]
                if not regime_stats.empty:
                    metrics[f'{regime.title()} Regime'] = {
                        'Number of Trades': len(regime_stats),
                        'Win Rate': f"{(len(regime_stats[regime_stats['value'] > regime_stats['value'].shift(1)]) / (len(regime_stats)//2) * 100):.2f}%",
                        'Average Trade Size': f"${regime_stats['value'].mean():.2f}"
                    }
        
        return {
            'metrics': metrics,
            'portfolio': portfolio,
            'trades': trades_df if num_trades > 0 else pd.DataFrame()
        }