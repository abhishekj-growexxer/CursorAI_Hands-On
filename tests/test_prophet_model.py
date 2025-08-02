"""
Tests for Prophet model implementation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from prophet import Prophet

from src.models.prophet_model import ProphetModel

@pytest.fixture
def prophet_model():
    """Create Prophet model instance."""
    return ProphetModel()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create date range
    dates = pd.date_range(start='2021-01-01', end='2021-12-31', freq='D')
    
    # Create sample data
    data = pd.DataFrame({
        'date': dates,
        'store_nbr': [1 if i < len(dates)/2 else 2 for i in range(len(dates))],
        'sales': np.random.uniform(0, 50, len(dates)),
        'onpromotion': np.random.randint(0, 2, len(dates)),
        'transactions': np.random.uniform(0, 100, len(dates)),
        'dcoilwtico': np.random.uniform(0, 50, len(dates)),
        'holiday_type_Holiday': np.random.randint(0, 2, len(dates))
    })
    
    return data

def test_model_initialization(prophet_model):
    """Test model initialization."""
    assert prophet_model.name == "prophet"
    assert isinstance(prophet_model.models, dict)
    assert prophet_model.seasonality_mode == "multiplicative"
    assert prophet_model.yearly_seasonality == "auto"
    assert prophet_model.weekly_seasonality == "auto"
    assert prophet_model.daily_seasonality is False

def test_create_prophet_df(prophet_model, sample_data):
    """Test creation of Prophet DataFrame."""
    # Create Prophet DataFrame for first store
    prophet_df = prophet_model._create_prophet_df(sample_data, 1)
    
    # Check DataFrame structure
    assert isinstance(prophet_df, pd.DataFrame)
    assert 'ds' in prophet_df.columns
    assert 'y' in prophet_df.columns
    assert 'holiday' in prophet_df.columns
    assert 'onpromotion' in prophet_df.columns
    assert 'transactions' in prophet_df.columns
    assert 'dcoilwtico' in prophet_df.columns
    
    # Check data types
    assert isinstance(prophet_df['ds'].iloc[0], pd.Timestamp)
    assert isinstance(prophet_df['y'].iloc[0], (int, float))
    assert isinstance(prophet_df['holiday'].iloc[0], (int, float))

def test_create_model(prophet_model):
    """Test Prophet model creation."""
    # Create model for a store
    model = prophet_model._create_model(1)
    
    # Check model configuration
    assert model.seasonality_mode == "multiplicative"
    assert 'onpromotion' in model.extra_regressors
    assert 'dcoilwtico' in model.extra_regressors
    assert 'transactions' in model.extra_regressors

def test_preprocess_data(prophet_model, sample_data):
    """Test data preprocessing."""
    # Preprocess training data
    prophet_dfs = prophet_model.preprocess(sample_data, is_training=True)
    
    # Check output format
    assert isinstance(prophet_dfs, dict)
    assert len(prophet_dfs) == 2  # Two stores
    
    # Check first store's data
    store_1_df = prophet_dfs[1]
    assert isinstance(store_1_df, pd.DataFrame)
    assert len(store_1_df) > 0
    assert 'ds' in store_1_df.columns
    assert 'y' in store_1_df.columns

def test_model_training(prophet_model, sample_data):
    """Test model training."""
    # Split data into train and validation
    train_data = sample_data[sample_data['date'] < '2021-12-01']
    val_data = sample_data[sample_data['date'] >= '2021-12-01']
    
    # Train model
    metrics = prophet_model.train(train_data, val_data)
    
    # Check if models are trained
    assert len(prophet_model.models) == 2  # One model per store
    assert all(isinstance(model, Prophet) for model in prophet_model.models.values())
    
    # Check metrics
    assert 'store_1_mse' in metrics
    assert 'store_1_mae' in metrics
    assert 'store_1_mape' in metrics
    assert 'avg_mse' in metrics

def test_model_prediction(prophet_model, sample_data):
    """Test model prediction."""
    # Train model
    prophet_model.train(sample_data)
    
    # Generate predictions
    predictions = prophet_model.predict(sample_data, 7)
    
    # Check predictions shape
    assert predictions.shape == (7, 2)  # 7 days, 2 stores
    assert not np.any(np.isnan(predictions))

def test_plot_components(prophet_model, sample_data, tmp_path):
    """Test component plotting."""
    # Train model
    prophet_model.train(sample_data)
    
    # Plot components
    output_dir = str(tmp_path)
    prophet_model.plot_components(1, output_dir)
    
    # Check if plots are created
    assert os.path.exists(os.path.join(output_dir, 'components_store_1.png'))
    assert os.path.exists(os.path.join(output_dir, 'cv_metrics_store_1.png'))

def test_get_forecast_uncertainty(prophet_model, sample_data):
    """Test forecast uncertainty calculation."""
    # Train model
    prophet_model.train(sample_data)
    
    # Get forecast with uncertainty
    forecast = prophet_model.get_forecast_uncertainty(1, 7)
    
    # Check forecast format
    assert isinstance(forecast, dict)
    assert 'yhat_lower' in forecast
    assert 'yhat_upper' in forecast
    assert forecast['yhat_lower'].shape == (7,)
    assert forecast['yhat_upper'].shape == (7,)
    assert not np.any(np.isnan(forecast['yhat_lower']))
    assert not np.any(np.isnan(forecast['yhat_upper'])) 