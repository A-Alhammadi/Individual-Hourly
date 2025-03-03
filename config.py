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
    # Date range (dates will be converted to hour start)
    "start_date": "2024-09-01",
    "end_date": "2024-12-31",
    
    # Trading pairs to test
    "symbols": ["BTC/USD"],
    
    # Initial capital and position size
    "initial_capital": 10000,
    "position_size": 1,
    
    # Adaptive parameters (all in hours)
    "adaptive": {
        "volatility_window": 720,        # 30 days * 24 hours
        "epsilon": 1e-10,
        "base_dead_zone": 0.5,
        "lookback_periods": 2400,        # 100 days * 24 hours
        "performance_window": 24,         # 1 day
        "reversion_threshold": 0.95,
        "min_reversion_count": 3
    },

    "rsi": {
        "period": 14,
        "overbought": 70,
        "oversold": 30,
        "weight": 0.5, # Weight for combining RSI signal with EMA signal
        "signal_threshold": 0.3  # Minimum combined signal strength for trade entry
    },
    
    # Optimization configuration (all in hours)
    "optimization": {
        "training_days": 1440,           # 1440 = 60 days * 24 hours
        "min_training_days": 720,        # 30 days * 24 hours
        "testing_start": "2024-11-01",   # Will be converted to hour start
        "testing_end": "2024-12-05",     # Will be converted to hour start
        
        # Parameter ranges (all in hours)
        "parameter_ranges": {
            "rsi": {
            "period": [7, 14, 21, 28],
            "overbought": [65, 70, 75, 80],
            "oversold": [20, 25, 30, 35],
            "weight": [0.3, 0.4, 0.5, 0.6, 0.7]
            },
            "ema": {
                "short": [12],    # range(12, 96, 6),      # [12, 24, 36, 48, 60, 72] hours
                "medium": [24],       # range(24, 240, 12),    # [24, 48, 72, 96, 120, 144] hours
                "volatility_window": [720],            # [360, 720, 1440],  # [15, 30, 60] days in hours
                #"volatility_threshold": [1.2, 1.6, 4.2, 4.8],   #[1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2], #[1.5, 2.0, 2.5]
                "min_trend_strength": [0.001, 0.002, 0.004]  
            },
            "volatility": {
                "annualization_factor": [8760],  # 365 days * 24 hours
                "baseline_window_multiplier": [1.5], # [1.5, 2.0, 2.5]
                "baseline_lookback_gap": [24],  # [24, 48, 72]
                "min_periods_multiplier": [0.75] # [0.75, 0.9]
            }
        }
    },
    
    # Base parameters (all in hours)
    "ema": {
        "short": 48,                 # 2 days in hours
        "medium": 96,                # 4 days in hours
        "volatility_window": 720,    # 30 days in hours
        #"volatility_threshold": 1.5,
        "min_trend_strength": 0.002
    },
    
    "volatility": {
        "annualization_factor": 8760,  # 365 days * 24 hours
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