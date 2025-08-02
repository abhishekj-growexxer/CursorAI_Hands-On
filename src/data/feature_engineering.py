"""
Feature engineering module for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
import yaml

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    """Feature engineering for time series data."""
    
    def __init__(self, config_path: str = "configs/training_configs/default.yaml"):
        """Initialize feature engineer."""
        logger.info(f"Initializing FeatureEngineer with config: {config_path}")
        self.config_path = config_path
        self.config = self._load_config()
        logger.debug(f"Loaded configuration: {self.config}")
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.debug(f"Successfully loaded config from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
            
    def create_lag_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        lag_periods: List[int]
    ) -> pd.DataFrame:
        """Create lag features for time series data."""
        logger.info(f"Creating lag features for {target_col}")
        try:
            df = data.copy()
            
            # Create lag features for each store
            for store in df['store_nbr'].unique():
                store_mask = df['store_nbr'] == store
                store_data = df.loc[store_mask, target_col]
                
                for lag in lag_periods:
                    lag_name = f'{target_col}_lag_{lag}'
                    df.loc[store_mask, lag_name] = store_data.shift(lag)
                    
            logger.debug(f"Created {len(lag_periods)} lag features")
            return df
            
        except Exception as e:
            logger.error(f"Error creating lag features: {str(e)}")
            raise
            
    def create_rolling_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        windows: List[int],
        statistics: List[str]
    ) -> pd.DataFrame:
        """Create rolling window features."""
        logger.info(f"Creating rolling features for {target_col}")
        try:
            df = data.copy()
            
            # Define statistic functions
            stat_funcs = {
                'mean': np.mean,
                'std': np.std,
                'min': np.min,
                'max': np.max,
                'median': np.median
            }
            
            # Create rolling features for each store
            for store in df['store_nbr'].unique():
                store_mask = df['store_nbr'] == store
                store_data = df.loc[store_mask, target_col]
                
                for window in windows:
                    for stat in statistics:
                        if stat not in stat_funcs:
                            logger.warning(f"Unknown statistic: {stat}")
                            continue
                            
                        feature_name = f'{target_col}_rolling_{stat}_{window}'
                        df.loc[store_mask, feature_name] = store_data.rolling(
                            window=window,
                            min_periods=1
                        ).apply(stat_funcs[stat])
                        
            logger.debug(f"Created {len(windows) * len(statistics)} rolling features")
            return df
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {str(e)}")
            raise
            
    def create_trend_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        windows: List[int]
    ) -> pd.DataFrame:
        """Create trend indicator features."""
        logger.info(f"Creating trend features for {target_col}")
        try:
            df = data.copy()
            
            # Create trend features for each store
            for store in df['store_nbr'].unique():
                store_mask = df['store_nbr'] == store
                store_data = df.loc[store_mask, target_col]
                
                for window in windows:
                    # Calculate moving average
                    ma = store_data.rolling(window=window, min_periods=1).mean()
                    
                    # Create trend indicator (1 if above MA, 0 if below)
                    feature_name = f'{target_col}_trend_{window}'
                    df.loc[store_mask, feature_name] = (store_data > ma).astype(int)
                    
            logger.debug(f"Created {len(windows)} trend features")
            return df
            
        except Exception as e:
            logger.error(f"Error creating trend features: {str(e)}")
            raise
            
    def create_cyclical_features(
        self,
        data: pd.DataFrame,
        col: str,
        max_val: int
    ) -> pd.DataFrame:
        """Create cyclical features using sine and cosine transformations."""
        logger.info(f"Creating cyclical features for {col}")
        try:
            df = data.copy()
            
            # Convert to radians
            values = 2 * np.pi * df[col] / max_val
            
            # Create sine and cosine features
            df[f'{col}_sin'] = np.sin(values)
            df[f'{col}_cos'] = np.cos(values)
            
            logger.debug(f"Created cyclical features for {col}")
            return df
            
        except Exception as e:
            logger.error(f"Error creating cyclical features: {str(e)}")
            raise
            
    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create all time series features."""
        logger.info("Creating all features")
        try:
            df = data.copy()
            target_col = 'sales'
            
            # Create lag features
            lag_periods = [1, 7, 14, 30]
            df = self.create_lag_features(df, target_col, lag_periods)
            
            # Create rolling features
            rolling_windows = [7, 14, 30]
            rolling_stats = ['mean', 'std', 'min', 'max']
            df = self.create_rolling_features(df, target_col, rolling_windows, rolling_stats)
            
            # Create trend features
            trend_windows = [7, 14, 30]
            df = self.create_trend_features(df, target_col, trend_windows)
            
            # Create cyclical features
            df = self.create_cyclical_features(df, 'day_of_week', 7)
            df = self.create_cyclical_features(df, 'month', 12)
            df = self.create_cyclical_features(df, 'day', 31)
            
            logger.info("Feature creation completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering process: {str(e)}")
            raise 