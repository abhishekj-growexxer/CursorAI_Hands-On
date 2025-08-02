"""
Tests for the Prefect workflows.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from prefect import flow
from src.monitoring.prefect_workflows import (
    load_and_process_data,
    engineer_features,
    split_data,
    train_model,
    evaluate_model,
    compare_models,
    train_and_evaluate_models,
    retrain_best_model,
    generate_predictions
)
from src.monitoring.mlflow_tracking import MLflowTracker

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create date range
    dates = pd.date_range(start='2021-01-01', end='2021-12-31', freq='D')
    
    # Create sample data for two stores
    data = []
    for store in [1, 2]:
        # Generate synthetic sales data with trend and seasonality
        t = np.arange(len(dates))
        trend = 0.1 * t
        seasonality = 10 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
        yearly = 5 * np.sin(2 * np.pi * t / 365)  # Yearly seasonality
        noise = np.random.normal(0, 1, len(dates))
        sales = trend + seasonality + yearly + noise
        
        store_data = pd.DataFrame({
            'date': dates,
            'store_nbr': store,
            'sales': sales,
            'onpromotion': np.random.randint(0, 2, len(dates)),
            'transactions': np.random.randint(100, 1000, len(dates)),
            'dcoilwtico': np.random.rand(len(dates)) * 50,
            'day_of_week': dates.dayofweek,
            'month': dates.month
        })
        data.append(store_data)
        
    return pd.concat(data, ignore_index=True)

@pytest.fixture
def processed_data(sample_data):
    """Create processed data for testing."""
    return {'final': sample_data}

@pytest.fixture
def mlflow_tracker(tmp_path):
    """Create MLflowTracker instance with temporary tracking URI."""
    tracking_uri = str(tmp_path / "mlruns")
    return MLflowTracker(tracking_uri=tracking_uri)

def test_load_and_process_data(processed_data, tmp_path):
    """Test data loading and processing task."""
    # Create test data directory
    data_dir = tmp_path / "data"
    os.makedirs(data_dir)
    
    # Save sample data
    for name, df in processed_data.items():
        df.to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)
        
    # Create test flow
    @flow
    def test_flow():
        return load_and_process_data(str(data_dir))
        
    # Run flow
    result = test_flow()
    
    # Check result
    assert isinstance(result, dict)
    assert 'final' in result
    assert isinstance(result['final'], pd.DataFrame)

def test_engineer_features(processed_data):
    """Test feature engineering task."""
    # Create test flow
    @flow
    def test_flow():
        return engineer_features(processed_data['final'])
        
    # Run flow
    result = test_flow()
    
    # Check result
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(processed_data['final'])
    assert 'sales' in result.columns

def test_split_data(processed_data):
    """Test data splitting task."""
    # Create test flow
    @flow
    def test_flow():
        return split_data(processed_data['final'])
        
    # Run flow
    train, val, test = test_flow()
    
    # Check results
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(val) + len(test) == len(processed_data['final'])

def test_train_model(processed_data, mlflow_tracker):
    """Test model training task."""
    # Create test flow
    @flow
    def test_flow():
        train_data = processed_data['final'].iloc[:300]
        val_data = processed_data['final'].iloc[300:400]
        return train_model("arima", train_data, val_data, mlflow_tracker)
        
    # Run flow
    metrics = test_flow()
    
    # Check results
    assert isinstance(metrics, dict)
    assert len(metrics) > 0
    assert all(isinstance(v, float) for v in metrics.values())

def test_evaluate_model(processed_data, mlflow_tracker):
    """Test model evaluation task."""
    # Create test flow
    @flow
    def test_flow():
        # First train the model
        train_data = processed_data['final'].iloc[:300]
        val_data = processed_data['final'].iloc[300:400]
        train_model("arima", train_data, val_data, mlflow_tracker)
        
        # Then evaluate
        test_data = processed_data['final'].iloc[400:]
        return evaluate_model("arima", test_data, mlflow_tracker)
        
    # Run flow
    metrics = test_flow()
    
    # Check results
    assert isinstance(metrics, dict)
    assert len(metrics) > 0
    assert all(isinstance(v, float) for v in metrics.values())

def test_compare_models(mlflow_tracker):
    """Test model comparison task."""
    # Create test metrics
    metrics = {
        'arima': {'rmse': 0.5},
        'prophet': {'rmse': 0.3},
        'lstm': {'rmse': 0.4}
    }
    
    # Create test flow
    @flow
    def test_flow():
        return compare_models(metrics, mlflow_tracker)
        
    # Run flow
    best_model = test_flow()
    
    # Check result
    assert best_model == 'prophet'

def test_train_and_evaluate_models(processed_data, tmp_path):
    """Test main training workflow."""
    # Create test data directory
    data_dir = tmp_path / "data"
    os.makedirs(data_dir)
    
    # Save sample data
    for name, df in processed_data.items():
        df.to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)
        
    # Create test flow
    @flow
    def test_flow():
        return train_and_evaluate_models(
            str(data_dir),
            models=["arima"]  # Test with one model for speed
        )
        
    # Run flow
    best_model = test_flow()
    
    # Check result
    assert isinstance(best_model, str)
    assert best_model == "arima"

def test_retrain_best_model(processed_data, tmp_path):
    """Test model retraining workflow."""
    # Create test data directory
    data_dir = tmp_path / "data"
    os.makedirs(data_dir)
    
    # Save sample data
    for name, df in processed_data.items():
        df.to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)
        
    # First train models
    train_and_evaluate_models(str(data_dir), models=["arima"])
    
    # Create test flow
    @flow
    def test_flow():
        return retrain_best_model(str(data_dir))
        
    # Run flow
    test_flow()
    
    # Check if model files exist
    assert os.path.exists("models/arima")
    assert len(os.listdir("models/arima")) > 0

def test_generate_predictions(processed_data, tmp_path):
    """Test prediction generation workflow."""
    # Create test data directory
    data_dir = tmp_path / "data"
    os.makedirs(data_dir)
    
    # Save sample data
    for name, df in processed_data.items():
        df.to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)
        
    # First train models
    train_and_evaluate_models(str(data_dir), models=["arima"])
    
    # Create test flow
    @flow
    def test_flow():
        return generate_predictions(str(data_dir), prediction_days=7)
        
    # Run flow
    predictions = test_flow()
    
    # Check predictions
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == 7
    assert 'date' in predictions.columns
    assert 'predictions' in predictions.columns

def test_workflow_error_handling(tmp_path):
    """Test error handling in workflows."""
    # Test with nonexistent directory
    with pytest.raises(Exception):
        train_and_evaluate_models("nonexistent_dir")
        
    # Test with invalid model name
    with pytest.raises(ValueError):
        train_and_evaluate_models(models=["invalid_model"])
        
    # Test prediction without trained models
    with pytest.raises(Exception):
        generate_predictions()

def test_workflow_caching(processed_data, tmp_path):
    """Test task caching in workflows."""
    # Create test data directory
    data_dir = tmp_path / "data"
    os.makedirs(data_dir)
    
    # Save sample data
    for name, df in processed_data.items():
        df.to_csv(os.path.join(data_dir, f"{name}.csv"), index=False)
        
    # Create test flow
    @flow
    def test_flow():
        # Run same task twice
        data1 = load_and_process_data(str(data_dir))
        data2 = load_and_process_data(str(data_dir))
        
        # Results should be identical due to caching
        return data1, data2
        
    # Run flow
    data1, data2 = test_flow()
    
    # Check results are identical
    pd.testing.assert_frame_equal(data1['final'], data2['final']) 