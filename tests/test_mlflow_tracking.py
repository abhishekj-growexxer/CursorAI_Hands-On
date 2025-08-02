"""
Tests for the MLflow tracking module.
"""

import pytest
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import os
from datetime import datetime
from src.monitoring.mlflow_tracking import MLflowTracker

@pytest.fixture
def mlflow_tracker(tmp_path):
    """Create MLflowTracker instance with temporary tracking URI."""
    tracking_uri = str(tmp_path / "mlruns")
    return MLflowTracker(tracking_uri=tracking_uri)

@pytest.fixture
def sample_metrics():
    """Create sample metrics."""
    return {
        'rmse': 0.5,
        'mae': 0.3,
        'mape': 10.0
    }

@pytest.fixture
def sample_params():
    """Create sample parameters."""
    return {
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 100
    }

@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    return np.random.rand(100)

def test_initialization(mlflow_tracker):
    """Test MLflowTracker initialization."""
    assert mlflow_tracker.experiment_name == "store_sales_forecasting"
    assert mlflow_tracker.tracking_uri is not None
    assert mlflow_tracker.config is not None

def test_start_run(mlflow_tracker):
    """Test starting MLflow run."""
    # Start run
    with mlflow_tracker.start_run() as run:
        assert run is not None
        assert run.info.run_id is not None
        
    # Start run with custom name
    run_name = "test_run"
    with mlflow_tracker.start_run(run_name=run_name) as run:
        assert run.info.run_name == run_name

def test_log_params(mlflow_tracker, sample_params):
    """Test logging parameters."""
    with mlflow_tracker.start_run():
        mlflow_tracker.log_params(sample_params)
        
        # Check logged parameters
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        for key, value in sample_params.items():
            assert run.data.params[key] == str(value)

def test_log_metrics(mlflow_tracker, sample_metrics):
    """Test logging metrics."""
    with mlflow_tracker.start_run():
        mlflow_tracker.log_metrics(sample_metrics)
        
        # Check logged metrics
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        for key, value in sample_metrics.items():
            assert run.data.metrics[key] == value

def test_log_artifact(mlflow_tracker, tmp_path):
    """Test logging artifact."""
    # Create test file
    test_file = tmp_path / "test.txt"
    with open(test_file, "w") as f:
        f.write("test content")
        
    with mlflow_tracker.start_run():
        mlflow_tracker.log_artifact(str(test_file))
        
        # Check logged artifact
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        artifacts = mlflow.artifacts.download_artifacts(run.info.run_id)
        assert os.path.exists(os.path.join(artifacts, "test.txt"))

def test_log_model(mlflow_tracker):
    """Test logging model."""
    # Create dummy model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    
    with mlflow_tracker.start_run():
        mlflow_tracker.log_model(model, "model")
        
        # Check logged model
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        model_uri = f"runs:/{run.info.run_id}/model"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        assert isinstance(loaded_model, LinearRegression)

def test_log_figure(mlflow_tracker):
    """Test logging figure."""
    # Create test figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    with mlflow_tracker.start_run():
        mlflow_tracker.log_figure(fig, "plot.png")
        
        # Check logged figure
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        artifacts = mlflow.artifacts.download_artifacts(run.info.run_id)
        assert os.path.exists(os.path.join(artifacts, "plot.png"))

def test_log_predictions(mlflow_tracker, sample_predictions):
    """Test logging predictions."""
    with mlflow_tracker.start_run():
        mlflow_tracker.log_predictions(sample_predictions)
        
        # Check logged predictions
        run = mlflow.get_run(mlflow.active_run().info.run_id)
        artifacts = mlflow.artifacts.download_artifacts(run.info.run_id)
        assert os.path.exists(os.path.join(artifacts, "predictions.csv"))

def test_create_experiment(mlflow_tracker):
    """Test experiment creation."""
    # Create experiment
    experiment_name = "test_experiment"
    experiment_id = mlflow_tracker.create_experiment_if_not_exists(experiment_name)
    
    # Check experiment
    experiment = mlflow.get_experiment(experiment_id)
    assert experiment.name == experiment_name
    
    # Try creating same experiment
    experiment_id2 = mlflow_tracker.create_experiment_if_not_exists(experiment_name)
    assert experiment_id == experiment_id2

def test_get_best_run(mlflow_tracker, sample_metrics):
    """Test getting best run."""
    # Create multiple runs with different metrics
    metrics_list = [
        {'val_rmse': 0.5},
        {'val_rmse': 0.3},
        {'val_rmse': 0.4}
    ]
    
    experiment_name = "test_experiment"
    mlflow_tracker.create_experiment_if_not_exists(experiment_name)
    
    for metrics in metrics_list:
        with mlflow_tracker.start_run():
            mlflow_tracker.log_metrics(metrics)
            
    # Get best run
    best_run = mlflow_tracker.get_best_run(
        experiment_name=experiment_name,
        metric_name="val_rmse",
        ascending=True
    )
    
    assert best_run is not None
    assert best_run.data.metrics['val_rmse'] == 0.3

def test_compare_runs(mlflow_tracker, sample_metrics):
    """Test comparing runs."""
    # Create multiple runs
    experiment_name = "test_experiment"
    mlflow_tracker.create_experiment_if_not_exists(experiment_name)
    
    for _ in range(3):
        with mlflow_tracker.start_run():
            mlflow_tracker.log_metrics(sample_metrics)
            
    # Compare runs
    comparison = mlflow_tracker.compare_runs(
        metric_names=list(sample_metrics.keys()),
        experiment_name=experiment_name
    )
    
    assert isinstance(comparison, pd.DataFrame)
    assert len(comparison) == 3
    for metric in sample_metrics:
        assert metric in comparison.columns

def test_plot_metric_comparison(mlflow_tracker, sample_metrics):
    """Test plotting metric comparison."""
    # Create multiple runs
    experiment_name = "test_experiment"
    mlflow_tracker.create_experiment_if_not_exists(experiment_name)
    
    for _ in range(3):
        with mlflow_tracker.start_run():
            mlflow_tracker.log_metrics(sample_metrics)
            
    # Create plot
    fig = mlflow_tracker.plot_metric_comparison(
        metric_names=list(sample_metrics.keys()),
        experiment_name=experiment_name
    )
    
    assert isinstance(fig, plt.Figure)

def test_error_handling(mlflow_tracker):
    """Test error handling."""
    # Test invalid experiment name
    with pytest.raises(Exception):
        mlflow_tracker.get_best_run("nonexistent_experiment")
        
    # Test logging metrics without active run
    with pytest.raises(Exception):
        mlflow_tracker.log_metrics({'test': 1.0})
        
    # Test invalid metric name
    with pytest.raises(Exception):
        mlflow_tracker.get_best_run(metric_name="nonexistent_metric")

def test_nested_runs(mlflow_tracker, sample_metrics):
    """Test nested runs."""
    with mlflow_tracker.start_run(run_name="parent"):
        mlflow_tracker.log_metrics({'parent_metric': 1.0})
        
        with mlflow_tracker.start_run(run_name="child", nested=True):
            mlflow_tracker.log_metrics({'child_metric': 2.0})
            
            # Check current run is child
            assert mlflow.active_run().info.run_name == "child"
            
        # Check current run is parent
        assert mlflow.active_run().info.run_name == "parent" 