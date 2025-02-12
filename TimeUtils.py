import pandas as pd
from datetime import datetime, timedelta
from config import BACKTEST_CONFIG

class TimeUtils:
    def __init__(self):
        self.hours_per_day = 24
        self.hours_per_year = 365 * 24
    
    def parse_datetime(self, datetime_str):
        """Parse datetime string maintaining hour precision"""
        dt = pd.to_datetime(datetime_str)
        if len(datetime_str) <= 10:  # If only date is provided (YYYY-MM-DD)
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return dt.floor('H')  # Round to nearest hour for datetime strings with time
    
    def calculate_training_period(self, test_start, training_hours):
        """Calculate training period start/end dates based on test start date"""
        # Parse test start maintaining hour precision
        test_start_dt = self.parse_datetime(test_start)
        
        # Calculate training end (1 hour before test start)
        train_end = test_start_dt - timedelta(hours=1)
        
        # Calculate training start
        train_start = train_end - timedelta(hours=training_hours)
        
        training_period = {
            'train_start': train_start.strftime('%Y-%m-%d %H:00:00'),
            'train_end': train_end.strftime('%Y-%m-%d %H:00:00'),
            'test_start': test_start_dt.strftime('%Y-%m-%d %H:00:00')
        }
        
        print("\nTraining period details:")
        print(f"Train start: {training_period['train_start']}")
        print(f"Train end: {training_period['train_end']}")
        print(f"Test start: {training_period['test_start']}")
        
        return training_period
    
    def calculate_fetch_period(self, test_start, test_end, training_hours, warmup_hours):
        """Calculate the exact period needed for data fetching with hourly precision"""
        print("\nCalculating fetch period:")
        print(f"Training hours required: {training_hours} ({training_hours/self.hours_per_day:.1f} days)")
        print(f"Warmup hours required: {warmup_hours} ({warmup_hours/self.hours_per_day:.1f} days)")
        
        # Parse dates maintaining hour precision
        test_start_dt = self.parse_datetime(test_start)
        test_end_dt = self.parse_datetime(test_end)
        
        if test_start_dt >= test_end_dt:
            raise ValueError(f"Test start ({test_start}) must be before test end ({test_end})")
        
        # Calculate total hours needed before test start
        total_pre_hours = training_hours + warmup_hours
        
        # Calculate fetch start date
        fetch_start = test_start_dt - pd.Timedelta(hours=total_pre_hours)
        
        # Calculate total hours needed
        test_duration_hours = ((test_end_dt - test_start_dt).total_seconds() / 3600)
        total_hours = total_pre_hours + test_duration_hours + 1
        
        fetch_period = {
            'fetch_start': fetch_start.strftime('%Y-%m-%d %H:00:00'),
            'fetch_end': test_end_dt.strftime('%Y-%m-%d %H:00:00'),
            'total_hours': total_hours
        }
        
        print(f"\nFetch period details:")
        print(f"Fetch start: {fetch_period['fetch_start']}")
        print(f"Fetch end: {fetch_period['fetch_end']}")
        print(f"Total hours: {total_hours:.0f} ({total_hours/self.hours_per_day:.1f} days)")
        print(f"Test duration: {test_duration_hours:.0f} hours ({test_duration_hours/self.hours_per_day:.1f} days)")
        
        return fetch_period
    
    def validate_data_periods(self, data, required_hours, symbol):
        """Validate that we have sufficient data for the required periods"""
        actual_hours = len(data)
        margin_hours = required_hours * 0.1  # 10% margin
        
        print(f"\nValidating data periods for {symbol}:")
        print(f"Required hours: {self.hours_to_human_readable(required_hours)}")
        print(f"Actual hours: {self.hours_to_human_readable(actual_hours)}")
        
        if actual_hours < required_hours:
            raise ValueError(
                f"Insufficient data for {symbol}. "
                f"Need {self.hours_to_human_readable(required_hours)}, "
                f"got {self.hours_to_human_readable(actual_hours)}"
            )
        
        if actual_hours > required_hours + margin_hours:
            raise ValueError(
                f"Too much data fetched for {symbol}. "
                f"Expected ~{self.hours_to_human_readable(required_hours)}, "
                f"got {self.hours_to_human_readable(actual_hours)}. "
                "Check date filtering."
            )
        
        # Validate data continuity
        date_diff = data.index.to_series().diff().iloc[1:]
        expected_diff = pd.Timedelta(hours=1)
        if not (date_diff == expected_diff).all():
            gaps = date_diff[date_diff != expected_diff]
            if not gaps.empty:
                print("\nWarning: Found gaps in data:")
                for idx, gap in gaps.items():
                    print(f"Gap at {idx}: {gap}")
        
        return True
    
    def hours_to_human_readable(self, hours):
        """Convert hours to a human-readable format"""
        days = hours // self.hours_per_day
        remaining_hours = hours % self.hours_per_day
        return f"{int(days)} days, {int(remaining_hours)} hours"