"""
Tests for the data processing module.
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.data_processor import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample sales data
    sales_data = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', end='2021-01-10'),
        'store_nbr': [1] * 10,
        'family': ['GROCERY'] * 10,
        'sales': np.random.rand(10) * 100,
        'onpromotion': np.random.randint(0, 2, 10)
    })
    
    # Create sample oil data
    oil_data = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', end='2021-01-10'),
        'dcoilwtico': np.random.rand(10) * 50
    })
    
    # Create sample holidays data
    holidays_data = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', end='2021-01-10'),
        'type': ['Holiday'] * 5 + ['Transfer'] * 5,
        'locale': ['National'] * 5 + ['Regional'] * 5,
        'locale_name': ['Ecuador'] * 10,
        'description': ['New Year'] * 10,
        'transferred': [False] * 10
    })
    
    # Create sample transactions data
    transactions_data = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', end='2021-01-10'),
        'store_nbr': [1] * 10,
        'transactions': np.random.randint(100, 1000, 10)
    })
    
    # Create sample stores data
    stores_data = pd.DataFrame({
        'store_nbr': [1],
        'city': ['Quito'],
        'state': ['Pichincha'],
        'type': ['A'],
        'cluster': [1]
    })
    
    return {
        'train': sales_data,
        'oil': oil_data,
        'holidays': holidays_data,
        'transactions': transactions_data,
        'stores': stores_data
    }

@pytest.fixture
def data_processor():
    """Create a DataProcessor instance."""
    return DataProcessor()

def test_load_config(data_processor):
    """Test configuration loading."""
    config = data_processor.config
    assert isinstance(config, dict)
    assert 'data' in config
    assert 'training' in config

def test_process_sales_data(data_processor, sample_data):
    """Test sales data processing."""
    processed = data_processor._process_sales_data(sample_data['train'])
    
    # Check if time-based features are created
    assert 'year' in processed.columns
    assert 'month' in processed.columns
    assert 'day' in processed.columns
    assert 'day_of_week' in processed.columns
    assert 'week_of_year' in processed.columns
    
    # Check if missing values are handled
    assert processed['sales'].isna().sum() == 0
    assert processed['onpromotion'].isna().sum() == 0

def test_process_oil_data(data_processor, sample_data):
    """Test oil data processing."""
    # Add some missing values
    sample_data['oil'].loc[2:4, 'dcoilwtico'] = np.nan
    
    processed = data_processor._process_oil_data(sample_data['oil'])
    
    # Check if missing values are handled
    assert processed['dcoilwtico'].isna().sum() == 0

def test_process_holidays_data(data_processor, sample_data):
    """Test holidays data processing."""
    processed = data_processor._process_holidays_data(sample_data['holidays'])
    
    # Check if dummy variables are created
    assert 'holiday_type_Holiday' in processed.columns
    assert 'holiday_type_Transfer' in processed.columns
    assert 'locale_National' in processed.columns
    assert 'locale_Regional' in processed.columns

def test_process_transactions_data(data_processor, sample_data):
    """Test transactions data processing."""
    # Add some missing values
    sample_data['transactions'].loc[2:4, 'transactions'] = np.nan
    
    processed = data_processor._process_transactions_data(sample_data['transactions'])
    
    # Check if missing values are handled
    assert processed['transactions'].isna().sum() == 0

def test_merge_features(data_processor, sample_data):
    """Test feature merging."""
    # Process individual datasets
    processed = {
        'sales': data_processor._process_sales_data(sample_data['train']),
        'oil': data_processor._process_oil_data(sample_data['oil']),
        'holidays': data_processor._process_holidays_data(sample_data['holidays']),
        'transactions': data_processor._process_transactions_data(sample_data['transactions'])
    }
    
    merged = data_processor._merge_features(processed)
    
    # Check if all features are present
    assert 'sales' in merged.columns
    assert 'dcoilwtico' in merged.columns
    assert 'transactions' in merged.columns
    assert 'holiday_type_Holiday' in merged.columns

def test_create_sequences(data_processor, sample_data):
    """Test sequence creation."""
    # Process and merge data
    processed = {
        'sales': data_processor._process_sales_data(sample_data['train']),
        'oil': data_processor._process_oil_data(sample_data['oil']),
        'holidays': data_processor._process_holidays_data(sample_data['holidays']),
        'transactions': data_processor._process_transactions_data(sample_data['transactions'])
    }
    merged = data_processor._merge_features(processed)
    
    # Create sequences
    X, y = data_processor.create_sequences(
        merged,
        sequence_length=3,
        prediction_horizon=2
    )
    
    # Check shapes
    assert len(X.shape) == 3  # (samples, sequence_length, features)
    assert len(y.shape) == 2  # (samples, prediction_horizon)
    assert X.shape[0] == y.shape[0]  # Same number of samples
    assert X.shape[1] == 3  # Sequence length
    assert y.shape[1] == 2  # Prediction horizon

def test_split_data(data_processor, sample_data):
    """Test data splitting."""
    processed = {
        'sales': data_processor._process_sales_data(sample_data['train']),
        'oil': data_processor._process_oil_data(sample_data['oil']),
        'holidays': data_processor._process_holidays_data(sample_data['holidays']),
        'transactions': data_processor._process_transactions_data(sample_data['transactions'])
    }
    merged = data_processor._merge_features(processed)
    
    # Split data
    train, test = data_processor.split_data(merged, test_size=0.2)
    
    # Check split sizes
    assert len(train) > len(test)
    assert abs(len(test) / len(merged) - 0.2) < 0.1  # Allow small deviation due to rounding

def test_scale_features(data_processor, sample_data):
    """Test feature scaling."""
    processed = {
        'sales': data_processor._process_sales_data(sample_data['train']),
        'oil': data_processor._process_oil_data(sample_data['oil']),
        'holidays': data_processor._process_holidays_data(sample_data['holidays']),
        'transactions': data_processor._process_transactions_data(sample_data['transactions'])
    }
    merged = data_processor._merge_features(processed)
    
    # Split data
    train, test = data_processor.split_data(merged, test_size=0.2)
    
    # Scale features
    feature_columns = ['sales', 'transactions', 'dcoilwtico']
    train_scaled, test_scaled = data_processor.scale_features(train, test, feature_columns)
    
    # Check if scaling is applied correctly
    for column in feature_columns:
        assert abs(train_scaled[column].mean()) < 1e-10  # Mean should be close to 0
        assert abs(train_scaled[column].std() - 1.0) < 1e-10  # Std should be close to 1

def test_end_to_end_processing(data_processor, sample_data):
    """Test the entire data processing pipeline."""
    # Load and preprocess data
    processed = data_processor.preprocess_data(sample_data)
    
    # Check if all steps are completed successfully
    assert 'final' in processed
    assert isinstance(processed['final'], pd.DataFrame)
    assert len(processed['final']) > 0
    
    # Check if all expected features are present
    expected_features = [
        'sales', 'onpromotion', 'year', 'month', 'day',
        'day_of_week', 'week_of_year', 'dcoilwtico',
        'transactions', 'holiday_type_Holiday', 'holiday_type_Transfer'
    ]
    for feature in expected_features:
        assert feature in processed['final'].columns 