"""
Tests for the ARIMA model implementation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.models.arima_model import ARIMAModel

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create date range
    dates = pd.date_range(start='2021-01-01', end='2021-03-31', freq='D')
    
    # Create sample data for two stores
    data = []
    for store in [1, 2]:
        # Generate synthetic sales data with trend and seasonality
        t = np.arange(len(dates))
        trend = 0.1 * t
        seasonality = 10 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
        noise = np.random.normal(0, 1, len(dates))
        sales = trend + seasonality + noise
        
        store_data = pd.DataFrame({
            'date': dates,
            'store_nbr': store,
            'sales': sales,
            'onpromotion': np.random.randint(0, 2, len(dates))
        })
        data.append(store_data)
        
    return pd.concat(data, ignore_index=True)

@pytest.fixture
def arima_model():
    """Create an ARIMAModel instance."""
    return ARIMAModel()

def test_model_initialization(arima_model):
    """Test model initialization."""
    assert arima_model.name == "arima"
    assert arima_model.model is None
    assert arima_model.order is None
    assert arima_model.seasonal_order is None

def test_check_stationarity(arima_model, sample_data):
    """Test stationarity check."""
    # Get data for one store
    store_data = sample_data[sample_data['store_nbr'] == 1]['sales']
    
    # Check stationarity
    is_stationary, p_value = arima_model._check_stationarity(store_data)
    
    assert isinstance(is_stationary, bool)
    assert isinstance(p_value, float)
    assert 0 <= p_value <= 1

def test_find_optimal_order(arima_model, sample_data):
    """Test finding optimal ARIMA order."""
    # Get data for one store
    store_data = sample_data[sample_data['store_nbr'] == 1]['sales']
    
    # Find optimal order
    order, seasonal_order = arima_model._find_optimal_order(
        store_data,
        max_p=2,
        max_d=1,
        max_q=2,
        seasonal=True
    )
    
    # Check order format
    assert isinstance(order, tuple)
    assert len(order) == 3
    assert all(isinstance(x, int) for x in order)
    
    if seasonal_order:
        assert isinstance(seasonal_order, tuple)
        assert len(seasonal_order) == 4
        assert all(isinstance(x, int) for x in seasonal_order)

def test_preprocess_data(arima_model, sample_data):
    """Test data preprocessing."""
    # Preprocess training data
    X_train, y_train = arima_model.preprocess(sample_data, is_training=True)
    
    # Check output format
    assert isinstance(X_train, pd.DataFrame)
    assert y_train is None
    assert set(X_train.columns) == {1, 2}  # Store numbers
    
    # Check if orders are found
    assert hasattr(arima_model, 'store_orders')
    assert hasattr(arima_model, 'store_seasonal_orders')
    assert set(arima_model.store_orders.keys()) == {1, 2}

def test_model_training(arima_model, sample_data):
    """Test model training."""
    # Split data into train and validation
    train_data = sample_data[sample_data['date'] < '2021-03-15']
    val_data = sample_data[sample_data['date'] >= '2021-03-15']
    
    # Train model
    metrics = arima_model.train(train_data, val_data)
    
    # Check if model is trained
    assert arima_model.model is not None
    assert isinstance(arima_model.model, dict)
    assert set(arima_model.model.keys()) == {1, 2}
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert 'avg_aic' in metrics
    assert 'avg_bic' in metrics
    assert all(isinstance(v, float) for v in metrics.values())

def test_model_prediction(arima_model, sample_data):
    """Test model prediction."""
    # Train model
    train_data = sample_data[sample_data['date'] < '2021-03-15']
    arima_model.train(train_data)
    
    # Make predictions
    predictions = arima_model.predict(train_data, prediction_steps=7)
    
    # Check predictions format
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (7, 2)  # 7 days, 2 stores
    assert not np.any(np.isnan(predictions))

def test_model_evaluation(arima_model, sample_data):
    """Test model evaluation."""
    # Split data
    train_data = sample_data[sample_data['date'] < '2021-03-15']
    test_data = sample_data[sample_data['date'] >= '2021-03-15']
    
    # Train model
    arima_model.train(train_data)
    
    # Evaluate model
    metrics = arima_model.evaluate(test_data)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'mape' in metrics
    assert all(isinstance(v, float) for v in metrics.values())

def test_get_diagnostics(arima_model, sample_data):
    """Test model diagnostics."""
    # Train model
    arima_model.train(sample_data)
    
    # Get diagnostics for first store
    diagnostics = arima_model.get_diagnostics(1)
    
    # Check diagnostics
    assert isinstance(diagnostics, dict)
    assert 'residuals' in diagnostics
    assert 'forecast' in diagnostics
    assert 'confidence_intervals' in diagnostics
    assert all(isinstance(v, pd.DataFrame) for v in diagnostics.values())

def test_decompose_series(arima_model, sample_data):
    """Test time series decomposition."""
    # Get data for one store
    store_data = sample_data[sample_data['store_nbr'] == 1]['sales']
    
    # Decompose series
    components = arima_model.decompose_series(store_data, period=7)
    
    # Check components
    assert isinstance(components, dict)
    assert 'trend' in components
    assert 'seasonal' in components
    assert 'residual' in components
    assert all(isinstance(v, pd.Series) for v in components.values())

def test_error_handling(arima_model):
    """Test error handling."""
    # Test prediction without training
    with pytest.raises(ValueError):
        arima_model.predict(pd.DataFrame(), 1)
    
    # Test diagnostics without training
    with pytest.raises(ValueError):
        arima_model.get_diagnostics(1)
    
    # Test with invalid data
    with pytest.raises(Exception):
        arima_model.train(None)

def test_model_persistence(arima_model, sample_data, tmp_path):
    """Test model saving and loading."""
    # Train model
    arima_model.train(sample_data)
    
    # Save model
    save_path = tmp_path / "arima_model.joblib"
    arima_model.save(str(save_path))
    
    # Create new model instance
    new_model = ARIMAModel()
    
    # Load model
    new_model.load(str(save_path))
    
    # Check if model is loaded correctly
    assert new_model.model is not None
    assert set(new_model.model.keys()) == {1, 2}
    
    # Compare predictions
    pred1 = arima_model.predict(sample_data, 7)
    pred2 = new_model.predict(sample_data, 7)
    np.testing.assert_array_almost_equal(pred1, pred2) 