"""
Tests for the training pipeline module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import mlflow
from src.training.train_pipeline import TrainingPipeline
from src.models.base_model import BaseTimeSeriesModel

class MockTimeSeriesModel(BaseTimeSeriesModel):
    """Mock model class for testing."""
    
    def __init__(self, name="mock_model"):
        super().__init__(name)
        self.is_trained = False
        
    def preprocess(self, data, is_training=True):
        X = data[['feature1', 'feature2']].values
        y = data['target'].values if is_training else None
        return X, y
        
    def train(self, train_data, validation_data=None):
        self.is_trained = True
        self.model = "mock_model_object"
        return {'loss': 0.1, 'accuracy': 0.9}
        
    def predict(self, data, prediction_steps):
        if not self.is_trained:
            raise ValueError("Model not trained")
        return np.zeros((len(data), prediction_steps))

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create date range
    dates = pd.date_range(start='2021-01-01', end='2021-01-31', freq='D')
    
    # Create sample data
    data = pd.DataFrame({
        'date': dates,
        'feature1': np.random.rand(len(dates)),
        'feature2': np.random.rand(len(dates)),
        'target': np.random.rand(len(dates))
    })
    
    return data

@pytest.fixture
def training_pipeline():
    """Create a TrainingPipeline instance."""
    return TrainingPipeline()

@pytest.fixture
def mock_model():
    """Create a MockTimeSeriesModel instance."""
    return MockTimeSeriesModel()

def test_pipeline_initialization(training_pipeline):
    """Test pipeline initialization."""
    assert training_pipeline.config is not None
    assert training_pipeline.experiment_name == "store_sales_forecasting"

def test_train_model(training_pipeline, mock_model, sample_data):
    """Test model training."""
    # Split data into train and validation
    train_data = sample_data.iloc[:20]
    val_data = sample_data.iloc[20:]
    
    # Train model
    metrics = training_pipeline.train_model(
        mock_model,
        train_data,
        val_data
    )
    
    # Check if model is trained
    assert mock_model.is_trained
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    
    # Check if model is saved
    model_dir = os.path.join("models", mock_model.name)
    assert os.path.exists(model_dir)
    assert len(os.listdir(model_dir)) > 0

def test_cross_validation(training_pipeline, mock_model, sample_data):
    """Test cross-validation."""
    # Perform cross-validation
    metrics = training_pipeline.cross_validate(
        mock_model,
        sample_data,
        n_splits=3
    )
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert 'avg_mse' in metrics
    assert 'std_mse' in metrics

def test_hyperparameter_search(training_pipeline, sample_data):
    """Test hyperparameter search."""
    # Define parameter grid
    param_grid = {
        'param1': [1, 2, 3],
        'param2': [0.1, 0.2, 0.3]
    }
    
    # Split data
    train_data = sample_data.iloc[:20]
    val_data = sample_data.iloc[20:]
    
    # Perform hyperparameter search
    results = training_pipeline.hyperparameter_search(
        MockTimeSeriesModel,
        param_grid,
        train_data,
        val_data,
        n_trials=2
    )
    
    # Check results
    assert isinstance(results, dict)
    assert 'best_params' in results
    assert 'best_value' in results

def test_mlflow_logging(training_pipeline, mock_model, sample_data):
    """Test MLflow logging."""
    # Train model with MLflow tracking
    training_pipeline.train_model(mock_model, sample_data)
    
    # Check MLflow experiment
    experiment = mlflow.get_experiment_by_name(
        training_pipeline.experiment_name
    )
    assert experiment is not None
    
    # Get latest run
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0
    
    # Check logged metrics
    latest_run = runs.iloc[0]
    assert 'loss' in latest_run.metrics
    assert 'accuracy' in latest_run.metrics

def test_model_evaluation(training_pipeline, mock_model, sample_data):
    """Test model evaluation."""
    # Train model
    training_pipeline.train_model(mock_model, sample_data)
    
    # Evaluate model
    metrics = mock_model.evaluate(sample_data)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'mape' in metrics

def test_error_handling(training_pipeline, mock_model):
    """Test error handling."""
    # Test with invalid data
    with pytest.raises(Exception):
        training_pipeline.train_model(mock_model, None)
    
    # Test with untrained model
    with pytest.raises(ValueError):
        mock_model.predict(pd.DataFrame(), 1)
    
    # Test with invalid config path
    with pytest.raises(Exception):
        TrainingPipeline(config_path="invalid_path.yaml")

def test_model_saving_loading(training_pipeline, mock_model, sample_data):
    """Test model saving and loading."""
    # Train and save model
    training_pipeline.train_model(mock_model, sample_data)
    
    # Get saved model path
    model_dir = os.path.join("models", mock_model.name)
    model_path = os.path.join(
        model_dir,
        os.listdir(model_dir)[0]
    )
    
    # Create new model instance
    new_model = MockTimeSeriesModel()
    
    # Load saved model
    new_model.load(model_path)
    
    # Check if model is loaded
    assert new_model.model == "mock_model_object"

def test_pipeline_with_test_data(training_pipeline, mock_model, sample_data):
    """Test pipeline with test data."""
    # Split data into train, validation, and test
    train_data = sample_data.iloc[:15]
    val_data = sample_data.iloc[15:25]
    test_data = sample_data.iloc[25:]
    
    # Train model with all splits
    metrics = training_pipeline.train_model(
        mock_model,
        train_data,
        val_data,
        test_data
    )
    
    # Check metrics
    assert 'test_mse' in metrics
    assert 'val_mse' in metrics
    assert 'loss' in metrics 