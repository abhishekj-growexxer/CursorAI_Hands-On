"""
Tests for the LSTM model implementation.
"""

import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import os
from src.models.lstm_model import LSTMTimeSeriesModel, TimeSeriesDataset, LSTMModel

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
def lstm_model():
    """Create an LSTMTimeSeriesModel instance."""
    return LSTMTimeSeriesModel()

def test_dataset_creation():
    """Test TimeSeriesDataset class."""
    # Create sample data
    features = np.random.rand(100, 5)
    targets = np.random.rand(100)
    sequence_length = 10
    
    # Create dataset
    dataset = TimeSeriesDataset(features, targets, sequence_length)
    
    # Check dataset properties
    assert len(dataset) == 91  # 100 - 10 + 1
    
    # Check item format
    sequence, target = dataset[0]
    assert isinstance(sequence, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert sequence.shape == (10, 5)
    assert target.shape == ()

def test_lstm_model_forward():
    """Test LSTM model forward pass."""
    # Create model
    model = LSTMModel(
        input_size=5,
        hidden_size=32,
        num_layers=2,
        output_size=1
    )
    
    # Create sample input
    batch_size = 16
    sequence_length = 10
    input_size = 5
    x = torch.randn(batch_size, sequence_length, input_size)
    
    # Forward pass
    output, hidden = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 1)
    assert len(hidden) == 2  # (h_n, c_n)
    assert hidden[0].shape == (2, batch_size, 32)  # h_n
    assert hidden[1].shape == (2, batch_size, 32)  # c_n

def test_model_initialization(lstm_model):
    """Test model initialization."""
    assert lstm_model.name == "lstm"
    assert lstm_model.hidden_size == 64
    assert lstm_model.num_layers == 2
    assert lstm_model.dropout == 0.2
    assert lstm_model.learning_rate == 0.001
    assert lstm_model.batch_size == 32
    assert lstm_model.num_epochs == 100
    assert lstm_model.sequence_length == 30

def test_preprocess_data(lstm_model, sample_data):
    """Test data preprocessing."""
    # Preprocess training data
    X_train, y_train = lstm_model.preprocess(sample_data, is_training=True)
    
    # Check output format
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape[1] == 6  # Number of features
    assert y_train.shape == (len(sample_data),)
    
    # Check if scaler is fitted
    assert hasattr(lstm_model.scaler, 'mean_')
    assert hasattr(lstm_model.scaler, 'scale_')

def test_create_dataloaders(lstm_model, sample_data):
    """Test dataloader creation."""
    # Preprocess data
    X, y = lstm_model.preprocess(sample_data)
    
    # Create dataloaders
    train_loader = lstm_model.create_dataloaders(X, y)
    
    # Check dataloader properties
    assert isinstance(train_loader.dataset, TimeSeriesDataset)
    assert train_loader.batch_size == lstm_model.batch_size
    
    # Check batch format
    batch_features, batch_targets = next(iter(train_loader))
    assert isinstance(batch_features, torch.Tensor)
    assert isinstance(batch_targets, torch.Tensor)
    assert batch_features.shape[0] <= lstm_model.batch_size
    assert batch_features.shape[1] == lstm_model.sequence_length

def test_model_training(lstm_model, sample_data):
    """Test model training."""
    # Split data into train and validation
    train_data = sample_data[sample_data['date'] < '2021-12-01']
    val_data = sample_data[sample_data['date'] >= '2021-12-01']
    
    # Train model with small number of epochs
    lstm_model.num_epochs = 2
    metrics = lstm_model.train(train_data, val_data)
    
    # Check if model is trained
    assert lstm_model.model is not None
    assert isinstance(lstm_model.model, LSTMModel)
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert 'train_loss' in metrics
    assert 'val_loss' in metrics
    assert all(isinstance(v, float) for v in metrics.values())

def test_model_prediction(lstm_model, sample_data):
    """Test model prediction."""
    # Train model
    train_data = sample_data[sample_data['date'] < '2021-12-01']
    lstm_model.num_epochs = 2
    lstm_model.train(train_data)
    
    # Make predictions
    predictions = lstm_model.predict(train_data, prediction_steps=7)
    
    # Check predictions format
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (7, 1)  # 7 days, 1 output feature
    assert not np.any(np.isnan(predictions))

def test_plot_training_curves(lstm_model, sample_data, tmp_path):
    """Test training curves plotting."""
    # Train model
    train_data = sample_data[sample_data['date'] < '2021-12-01']
    val_data = sample_data[sample_data['date'] >= '2021-12-01']
    
    # Set output directory
    os.makedirs(os.path.join(tmp_path, 'models/lstm'), exist_ok=True)
    
    # Train model and plot curves
    lstm_model.num_epochs = 2
    with pytest.warns(None) as record:
        lstm_model.train(train_data, val_data)
    
    # Check if plot is created
    assert os.path.exists(os.path.join('models/lstm/training_curves.png'))

def test_model_validation(lstm_model, sample_data):
    """Test model validation."""
    # Preprocess data
    X, y = lstm_model.preprocess(sample_data)
    val_loader = lstm_model.create_dataloaders(X, y)
    
    # Initialize model
    lstm_model.model = LSTMModel(
        input_size=X.shape[1],
        hidden_size=lstm_model.hidden_size,
        num_layers=lstm_model.num_layers,
        output_size=1
    ).to(lstm_model.device)
    
    # Validate model
    criterion = torch.nn.MSELoss()
    val_loss = lstm_model._validate(val_loader, criterion)
    
    # Check validation loss
    assert isinstance(val_loss, float)
    assert val_loss >= 0

def test_error_handling(lstm_model):
    """Test error handling."""
    # Test prediction without training
    with pytest.raises(ValueError):
        lstm_model.predict(pd.DataFrame(), 1)
    
    # Test with invalid data
    with pytest.raises(Exception):
        lstm_model.train(None)

def test_model_persistence(lstm_model, sample_data, tmp_path):
    """Test model saving and loading."""
    # Train model
    train_data = sample_data[sample_data['date'] < '2021-12-01']
    lstm_model.num_epochs = 2
    lstm_model.train(train_data)
    
    # Save model
    save_path = tmp_path / "lstm_model.pth"
    lstm_model.save(str(save_path))
    
    # Create new model instance
    new_model = LSTMTimeSeriesModel()
    
    # Load model
    new_model.load(str(save_path))
    
    # Check if model is loaded correctly
    assert new_model.model is not None
    assert new_model.hidden_size == lstm_model.hidden_size
    assert new_model.num_layers == lstm_model.num_layers
    
    # Compare predictions
    pred1 = lstm_model.predict(sample_data, 7)
    pred2 = new_model.predict(sample_data, 7)
    np.testing.assert_array_almost_equal(pred1, pred2)

def test_gpu_support(lstm_model):
    """Test GPU support."""
    # Check if GPU is available
    if torch.cuda.is_available():
        assert str(lstm_model.device) == 'cuda'
    else:
        assert str(lstm_model.device) == 'cpu' 