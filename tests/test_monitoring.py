"""
Tests for the monitoring module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from src.monitoring.monitoring import ModelMonitor

@pytest.fixture
def model_monitor(tmp_path):
    """Create ModelMonitor instance with temporary log directory."""
    log_dir = tmp_path / "logs/monitoring"
    return ModelMonitor(log_dir=str(log_dir))

@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    return np.random.rand(100)

@pytest.fixture
def sample_actuals():
    """Create sample actual values."""
    return np.random.rand(100)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create date range
    dates = pd.date_range(start='2021-01-01', end='2021-12-31', freq='D')
    
    # Create sample data
    data = pd.DataFrame({
        'feature1': np.random.rand(len(dates)),
        'feature2': np.random.rand(len(dates)),
        'target': np.random.rand(len(dates))
    })
    
    return data

def test_initialization(model_monitor):
    """Test ModelMonitor initialization."""
    assert model_monitor.config is not None
    assert os.path.exists(model_monitor.log_dir)

def test_monitor_predictions(model_monitor, sample_predictions, sample_actuals):
    """Test prediction monitoring."""
    # Monitor predictions without actuals
    metrics = model_monitor.monitor_predictions(sample_predictions)
    
    # Check basic metrics
    assert 'mean' in metrics
    assert 'std' in metrics
    assert 'min' in metrics
    assert 'max' in metrics
    
    # Monitor predictions with actuals
    metrics = model_monitor.monitor_predictions(sample_predictions, sample_actuals)
    
    # Check performance metrics
    assert 'mse' in metrics
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'mape' in metrics
    
    # Check if plots are created
    plots_dir = os.path.join(model_monitor.log_dir, 'plots')
    assert os.path.exists(plots_dir)
    assert any(f.startswith('predictions_') for f in os.listdir(plots_dir))

def test_monitor_data_drift(model_monitor, sample_data):
    """Test data drift monitoring."""
    # Create reference and current data
    reference_data = sample_data.iloc[:100]
    current_data = sample_data.iloc[100:200]
    
    # Monitor drift
    metrics = model_monitor.monitor_data_drift(reference_data, current_data)
    
    # Check metrics for each numeric column
    for col in sample_data.select_dtypes(include=[np.number]).columns:
        assert f'{col}_ks_stat' in metrics
        assert f'{col}_p_value' in metrics
        assert f'{col}_mean_diff' in metrics
        assert f'{col}_std_diff' in metrics
        
    # Check if plots are created
    plots_dir = os.path.join(model_monitor.log_dir, 'plots')
    assert os.path.exists(plots_dir)
    assert any(f.startswith('distribution_') for f in os.listdir(plots_dir))

def test_monitor_performance(model_monitor):
    """Test performance monitoring."""
    # Create sample metrics
    metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88
    }
    
    # Monitor performance
    model_monitor.monitor_performance('test_model', metrics)
    
    # Check if metrics are logged
    metrics_dir = os.path.join(model_monitor.log_dir, 'metrics')
    assert os.path.exists(metrics_dir)
    assert os.path.exists(os.path.join(metrics_dir, 'test_model_performance.jsonl'))

def test_monitor_system_resources(model_monitor):
    """Test system resource monitoring."""
    # Monitor resources
    metrics = model_monitor.monitor_system_resources()
    
    # Check metrics
    assert 'cpu_percent' in metrics
    assert 'memory_percent' in metrics
    assert 'memory_used_gb' in metrics
    assert 'memory_available_gb' in metrics
    assert 'disk_percent' in metrics
    assert 'disk_used_gb' in metrics
    assert 'disk_free_gb' in metrics
    
    # Check if metrics are logged
    metrics_dir = os.path.join(model_monitor.log_dir, 'metrics')
    assert os.path.exists(metrics_dir)
    assert os.path.exists(os.path.join(metrics_dir, 'system_metrics.jsonl'))

def test_log_metrics(model_monitor):
    """Test metrics logging."""
    # Create sample metrics
    metrics = {'metric1': 0.5, 'metric2': 0.8}
    timestamp = datetime.now()
    
    # Log metrics
    model_monitor._log_metrics('test_metrics', metrics, timestamp)
    
    # Check if metrics are logged
    metrics_file = os.path.join(
        model_monitor.log_dir,
        'metrics',
        'test_metrics.jsonl'
    )
    assert os.path.exists(metrics_file)
    
    # Check log content
    with open(metrics_file, 'r') as f:
        log_entry = json.loads(f.readline())
        assert log_entry['timestamp'] == timestamp.isoformat()
        assert log_entry['metrics'] == metrics

def test_plot_predictions(model_monitor, sample_predictions, sample_actuals):
    """Test prediction plotting."""
    timestamp = datetime.now()
    
    # Plot predictions without actuals
    model_monitor._plot_predictions(sample_predictions, None, timestamp)
    
    # Plot predictions with actuals
    model_monitor._plot_predictions(sample_predictions, sample_actuals, timestamp)
    
    # Check if plots are created
    plots_dir = os.path.join(model_monitor.log_dir, 'plots')
    plot_files = os.listdir(plots_dir)
    assert len(plot_files) == 2
    assert all(f.startswith('predictions_') for f in plot_files)

def test_plot_distributions(model_monitor, sample_data):
    """Test distribution plotting."""
    # Create reference and current data
    reference_data = sample_data.iloc[:100]
    current_data = sample_data.iloc[100:200]
    timestamp = datetime.now()
    
    # Plot distributions
    model_monitor._plot_distributions(reference_data, current_data, timestamp)
    
    # Check if plots are created
    plots_dir = os.path.join(model_monitor.log_dir, 'plots')
    assert os.path.exists(plots_dir)
    assert any(f.startswith('distribution_') for f in os.listdir(plots_dir))

def test_plot_metrics_history(model_monitor):
    """Test metrics history plotting."""
    # Create sample metrics history
    metrics = [
        {'accuracy': 0.8, 'loss': 0.2},
        {'accuracy': 0.85, 'loss': 0.15},
        {'accuracy': 0.9, 'loss': 0.1}
    ]
    
    metrics_dir = os.path.join(model_monitor.log_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Write metrics to file
    metrics_file = os.path.join(metrics_dir, 'test_model_performance.jsonl')
    with open(metrics_file, 'w') as f:
        for m in metrics:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'metrics': m
            }
            f.write(json.dumps(log_entry) + '\n')
            
    # Plot metrics history
    model_monitor._plot_metrics_history('test_model', datetime.now())
    
    # Check if plots are created
    plots_dir = os.path.join(model_monitor.log_dir, 'plots')
    assert os.path.exists(plots_dir)
    assert any(f.startswith('test_model_') for f in os.listdir(plots_dir))

def test_get_metrics_history(model_monitor):
    """Test getting metrics history."""
    # Create sample metrics history
    metrics = [
        {'metric1': 0.5, 'metric2': 0.8},
        {'metric1': 0.6, 'metric2': 0.7},
        {'metric1': 0.7, 'metric2': 0.6}
    ]
    
    metrics_dir = os.path.join(model_monitor.log_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Write metrics to file
    metrics_file = os.path.join(metrics_dir, 'test_metrics.jsonl')
    timestamps = []
    with open(metrics_file, 'w') as f:
        for m in metrics:
            timestamp = datetime.now()
            timestamps.append(timestamp)
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'metrics': m
            }
            f.write(json.dumps(log_entry) + '\n')
            
    # Get metrics history
    history = model_monitor.get_metrics_history(
        'test_metrics',
        start_time=timestamps[0],
        end_time=timestamps[-1]
    )
    
    # Check history
    assert isinstance(history, pd.DataFrame)
    assert len(history) == len(metrics)
    assert 'metric1' in history.columns
    assert 'metric2' in history.columns

def test_check_thresholds(model_monitor):
    """Test threshold checking."""
    # Create sample metrics and thresholds
    metrics = {
        'metric1': 0.8,
        'metric2': 0.6,
        'metric3': 0.4
    }
    
    thresholds = {
        'metric1': 0.7,
        'metric2': 0.8,
        'metric3': 0.3
    }
    
    # Check thresholds
    violations = model_monitor.check_thresholds(metrics, thresholds)
    
    # Check results
    assert violations['metric1'] is True  # 0.8 > 0.7
    assert violations['metric2'] is False  # 0.6 < 0.8
    assert violations['metric3'] is True  # 0.4 > 0.3

def test_error_handling(model_monitor):
    """Test error handling."""
    # Test with invalid metrics
    with pytest.raises(Exception):
        model_monitor.monitor_predictions(None)
        
    # Test with mismatched arrays
    with pytest.raises(Exception):
        model_monitor.monitor_predictions(
            np.random.rand(10),
            np.random.rand(20)
        )
        
    # Test with invalid metric type
    with pytest.raises(Exception):
        model_monitor.get_metrics_history('invalid_type')
        
    # Test with invalid thresholds
    with pytest.raises(Exception):
        model_monitor.check_thresholds(
            {'metric': 0.5},
            'invalid_thresholds'
        ) 