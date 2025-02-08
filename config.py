# config.py

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "dbname": "cryptocurrencies",
    "user": "myuser",
    "password": "mypassword"
}

BACKTEST_CONFIG = {
    # Date range
    "start_date": "2022-11-01",
    "end_date": "2024-12-31",
    
    # Trading pairs to test
    "symbols": ["LTC/USD", "BTC/USD", "ETH/USD", "XRP/USD"],
    #"symbols": ["LTC/USD", "BTC/USD", "ETH/USD", "XRP/USD"],
    # Initial capital for each currency
    "initial_capital": 10000,
    
    # Position size (percentage of capital)
    "position_size": 1,  # 95% of capital
    
    # Optimization configuration
    "optimization": {
        "training_days": 365,  # Number of days to use for training before test period
        "testing_start": "2024-01-01",   # Start date for testing period
        "testing_end": "2024-06-01",     # End date for testing period
        "min_training_days": 30,         # Minimum days required for training
        
        # Parameter ranges for optimization
        "parameter_ranges": {
            "ema": {
                "short": range(5, 16, 5),     # 3 values [5,10,15]
                "medium": range(15, 46, 15),  # 3 values [15,30,45]
                "volatility_window": [20, 40, 60],  # 3 values
                "volatility_threshold": [1.2, 1.5, 2.0],  # 3 values
                "min_trend_strength": [0.002, 0.005, 0.01, 0.015, 0.02]  # 0.2% to 2% minimum trend strength
            },
            "volatility": {
                "annualization_factor": [365*24],  # 1 value (stick with hourly)
                "baseline_window_multiplier": [1.5, 2],  # 2 values
                "baseline_lookback_gap": [0, 5],  # 2 values
                "min_periods_multiplier": [0.75]  # 1 value
            }
        }
    },
    
    # Base parameters (default values)
    "ema": {
        "short": 9,
        "medium": 21,
        "volatility_window": 30,
        "volatility_threshold": 1.5
    },
    
    "volatility": {
        "annualization_factor": 365 * 24,  # Default for hourly data
        "baseline_window_multiplier": 2,   # Default: baseline window = 2 * volatility_window
        "baseline_lookback_gap": 0,        # Default: no gap between current and baseline windows
        "min_periods_multiplier": 0.75     # Default: need 75% of window size for calculation
    },
    
    # Trading fees
    "trading_fee": 0,  # 0.001 = 0.1%
    
    # Output configuration
    "results_dir": "backtest_results",  # Directory to save results
    "save_trades": True,  # Whether to save detailed trade information
    "save_plots": True   # Whether to save plots as PNG files
}