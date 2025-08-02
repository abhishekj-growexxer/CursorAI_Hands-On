"""
Data processing module for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union, Tuple
from datetime import datetime
import yaml
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataProcessor:
    """Data processor for time series data."""
    
    def __init__(self, config_path: str = "configs/training_configs/default.yaml"):
        """Initialize data processor."""
        logger.info(f"Initializing DataProcessor with config: {config_path}")
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
            
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Process all data sources and merge them."""
        logger.info("Starting data preprocessing")
        try:
            # Process individual datasets
            logger.debug("Processing sales data")
            processed_sales = self._process_sales_data(data['train'])
            
            logger.debug("Processing oil data")
            processed_oil = self._process_oil_data(data['oil'])
            
            logger.debug("Processing holidays data")
            processed_holidays = self._process_holidays_data(data['holidays'])
            
            logger.debug("Processing transactions data")
            processed_transactions = self._process_transactions_data(data['transactions'])
            
            logger.debug("Merging features")
            # Merge all processed data
            final_data = self._merge_features({
                'sales': processed_sales,
                'oil': processed_oil,
                'holidays': processed_holidays,
                'transactions': processed_transactions
            })
            
            logger.info("Data preprocessing completed successfully")
            return final_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def _process_sales_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process sales data."""
        logger.debug("Processing sales data")
        try:
            df = data.copy()
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Extract time-based features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            logger.debug(f"Processed sales data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing sales data: {str(e)}")
            raise
            
    def _process_oil_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process oil price data."""
        logger.debug("Processing oil data")
        try:
            df = data.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            logger.debug(f"Processed oil data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing oil data: {str(e)}")
            raise
            
    def _process_holidays_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process holidays data."""
        logger.debug("Processing holidays data")
        try:
            df = data.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            # Create dummy variables for holiday types and locales
            df = pd.get_dummies(df, columns=['type', 'locale'])
            
            logger.debug(f"Processed holidays data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing holidays data: {str(e)}")
            raise
            
    def _process_transactions_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process transactions data."""
        logger.debug("Processing transactions data")
        try:
            df = data.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            # Handle missing values
            df['transactions'].fillna(df['transactions'].mean(), inplace=True)
            
            # Add rolling statistics
            df['transactions_rolling_mean'] = df.groupby('store_nbr')['transactions'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
            
            logger.debug(f"Processed transactions data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error processing transactions data: {str(e)}")
            raise
            
    def _merge_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all processed features."""
        logger.info("Merging processed features")
        try:
            # Start with sales data as base
            final_df = data_dict['sales'].copy()
            
            # Merge oil prices
            final_df = pd.merge(
                final_df,
                data_dict['oil'][['date', 'dcoilwtico']],
                on='date',
                how='left'
            )
            
            # Fill missing oil prices with forward fill then backward fill
            final_df['dcoilwtico'].fillna(method='ffill', inplace=True)
            final_df['dcoilwtico'].fillna(method='bfill', inplace=True)
            
            # Merge holidays
            holiday_cols = [col for col in data_dict['holidays'].columns
                          if col not in ['date', 'description', 'locale_name', 'transferred']]
            final_df = pd.merge(
                final_df,
                data_dict['holidays'][['date'] + holiday_cols],
                on='date',
                how='left'
            )
            
            # Fill holiday indicators with 0
            for col in holiday_cols:
                final_df[col].fillna(0, inplace=True)
                
            # Merge transactions
            final_df = pd.merge(
                final_df,
                data_dict['transactions'][['date', 'store_nbr', 'transactions', 'transactions_rolling_mean']],
                on=['date', 'store_nbr'],
                how='left'
            )
            
            # Fill missing transactions with store average
            final_df['transactions'].fillna(
                final_df.groupby('store_nbr')['transactions'].transform('mean'),
                inplace=True
            )
            final_df['transactions_rolling_mean'].fillna(
                final_df['transactions'],
                inplace=True
            )
            
            logger.info(f"Final merged data shape: {final_df.shape}")
            return final_df
            
        except Exception as e:
            logger.error(f"Error merging features: {str(e)}")
            raise
            
    def create_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models."""
        logger.info("Creating sequences")
        try:
            sequence_length = sequence_length or self.config['data']['sequence_length']
            
            # Sort data by date and store
            data = data.sort_values(['store_nbr', 'date'])
            
            # Create sequences for each store
            X, y = [], []
            for store in data['store_nbr'].unique():
                store_data = data[data['store_nbr'] == store]
                
                # Create sequences
                for i in range(len(store_data) - sequence_length):
                    X.append(store_data.iloc[i:i + sequence_length].values)
                    y.append(store_data.iloc[i + sequence_length]['sales'])
                    
            X = np.array(X)
            y = np.array(y)
            
            logger.debug(f"Created sequences with shape X: {X.shape}, y: {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise
            
    def split_data(
        self,
        data: pd.DataFrame,
        test_size: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        logger.info("Splitting data")
        try:
            test_size = test_size or self.config['data']['test_size']
            
            # Sort by date
            data = data.sort_values('date')
            
            # Calculate split point
            split_idx = int(len(data) * (1 - test_size))
            split_date = data.iloc[split_idx]['date']
            
            # Split data
            train_data = data[data['date'] < split_date]
            test_data = data[data['date'] >= split_date]
            
            logger.debug(f"Split data - Train shape: {train_data.shape}, Test shape: {test_data.shape}")
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise 