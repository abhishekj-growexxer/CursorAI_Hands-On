"""
Tests for the feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.feature_engineering import FeatureEngineer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create date range
    dates = pd.date_range(start='2021-01-01', end='2021-01-31', freq='D')
    
    # Create sample data
    data = pd.DataFrame({
        'date': dates.repeat(2),  # Two stores
        'store_nbr': [1, 2] * len(dates),
        'sales': np.random.rand(len(dates) * 2) * 100,
        'onpromotion': np.random.randint(0, 2, len(dates) * 2),
        'transactions': np.random.randint(100, 1000, len(dates) * 2),
        'dcoilwtico': np.random.rand(len(dates) * 2) * 50,
        'day_of_week_sin': np.sin(2 * np.pi * np.arange(len(dates) * 2) / 7)
    })
    
    return data

@pytest.fixture
def feature_engineer():
    """Create a FeatureEngineer instance."""
    return FeatureEngineer()

def test_create_lag_features(feature_engineer, sample_data):
    """Test creation of lag features."""
    lag_periods = [1, 7]
    result = feature_engineer.create_lag_features(sample_data, 'sales', lag_periods)
    
    # Check if lag features are created
    assert 'sales_lag_1' in result.columns
    assert 'sales_lag_7' in result.columns
    
    # Check values
    store_1_data = result[result['store_nbr'] == 1]
    assert np.isclose(
        store_1_data['sales'].iloc[1:].values,
        store_1_data['sales_lag_1'].iloc[1:].dropna().values
    ).all()

def test_create_rolling_features(feature_engineer, sample_data):
    """Test creation of rolling features."""
    windows = [7]
    statistics = ['mean', 'std', 'min', 'max']
    result = feature_engineer.create_rolling_features(
        sample_data,
        'sales',
        windows,
        statistics
    )
    
    # Check if rolling features are created
    assert 'sales_rolling_mean_7' in result.columns
    assert 'sales_rolling_std_7' in result.columns
    assert 'sales_rolling_min_7' in result.columns
    assert 'sales_rolling_max_7' in result.columns
    
    # Check if values are reasonable
    assert (result['sales_rolling_min_7'] <= result['sales_rolling_max_7']).all()
    assert (result['sales_rolling_mean_7'] >= result['sales_rolling_min_7']).all()
    assert (result['sales_rolling_mean_7'] <= result['sales_rolling_max_7']).all()

def test_create_cyclical_features(feature_engineer, sample_data):
    """Test creation of cyclical features."""
    result = feature_engineer.create_cyclical_features(sample_data)
    
    # Check if cyclical features are created
    cyclical_features = [
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos',
        'day_of_month_sin', 'day_of_month_cos'
    ]
    
    for feature in cyclical_features:
        assert feature in result.columns
        
    # Check if values are in correct range (-1 to 1)
    for feature in cyclical_features:
        assert (result[feature] >= -1).all()
        assert (result[feature] <= 1).all()

def test_create_interaction_features(feature_engineer, sample_data):
    """Test creation of interaction features."""
    feature_pairs = [
        ('onpromotion', 'day_of_week_sin'),
        ('transactions', 'dcoilwtico')
    ]
    
    result = feature_engineer.create_interaction_features(sample_data, feature_pairs)
    
    # Check if interaction features are created
    assert 'onpromotion_day_of_week_sin_interaction' in result.columns
    assert 'transactions_dcoilwtico_interaction' in result.columns
    
    # Check if values are correctly calculated
    for col1, col2 in feature_pairs:
        interaction_col = f'{col1}_{col2}_interaction'
        expected = sample_data[col1] * sample_data[col2]
        assert (result[interaction_col] == expected).all()

def test_create_trend_features(feature_engineer, sample_data):
    """Test creation of trend features."""
    windows = [7]
    result = feature_engineer.create_trend_features(sample_data, 'sales', windows)
    
    # Check if trend features are created
    assert 'sales_trend_7' in result.columns
    assert 'sales_trend_strength_7' in result.columns
    
    # Check if trend indicator is binary
    assert set(result['sales_trend_7'].unique()).issubset({0, 1})

def test_create_all_features(feature_engineer, sample_data):
    """Test creation of all features."""
    result = feature_engineer.create_all_features(sample_data)
    
    # Check if all feature types are created
    feature_types = [
        '_lag_', '_rolling_', '_sin', '_cos',
        '_trend_', '_interaction'
    ]
    
    for feature_type in feature_types:
        assert any(feature_type in col for col in result.columns)
        
    # Check if number of features increased
    assert len(result.columns) > len(sample_data.columns)

def test_get_feature_importance_correlation(feature_engineer, sample_data):
    """Test feature importance calculation using correlation method."""
    importance = feature_engineer.get_feature_importance(
        sample_data,
        'sales',
        method='correlation'
    )
    
    # Check structure
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    
    # Check if importance scores are between 0 and 1
    assert (importance['importance'] >= 0).all()
    assert (importance['importance'] <= 1).all()
    
    # Check if sorted in descending order
    assert (importance['importance'].diff().fillna(0) <= 0).all()

def test_get_feature_importance_mutual_info(feature_engineer, sample_data):
    """Test feature importance calculation using mutual information method."""
    importance = feature_engineer.get_feature_importance(
        sample_data,
        'sales',
        method='mutual_info'
    )
    
    # Check structure
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    
    # Check if importance scores are non-negative
    assert (importance['importance'] >= 0).all()
    
    # Check if sorted in descending order
    assert (importance['importance'].diff().fillna(0) <= 0).all()

def test_invalid_feature_importance_method(feature_engineer, sample_data):
    """Test error handling for invalid feature importance method."""
    with pytest.raises(ValueError):
        feature_engineer.get_feature_importance(
            sample_data,
            'sales',
            method='invalid_method'
        )

def test_handle_missing_values(feature_engineer, sample_data):
    """Test handling of missing values in feature creation."""
    # Add some missing values
    sample_data.loc[1:5, 'sales'] = np.nan
    
    # Try to create features
    result = feature_engineer.create_all_features(sample_data)
    
    # Check if result is not empty
    assert len(result) > 0
    
    # Check if missing values are handled in derived features
    assert not result.isnull().all().any()

def test_feature_engineering_with_single_store(feature_engineer):
    """Test feature engineering with data from a single store."""
    # Create sample data with single store
    dates = pd.date_range(start='2021-01-01', end='2021-01-31', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'store_nbr': [1] * len(dates),
        'sales': np.random.rand(len(dates)) * 100,
        'onpromotion': np.random.randint(0, 2, len(dates)),
        'transactions': np.random.randint(100, 1000, len(dates)),
        'dcoilwtico': np.random.rand(len(dates)) * 50
    })
    
    # Create features
    result = feature_engineer.create_all_features(data)
    
    # Check if features are created correctly
    assert len(result) == len(data)
    assert not result.isnull().all().any() 