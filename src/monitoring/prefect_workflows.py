"""
Prefect workflows for orchestrating the ML pipeline.
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from prefect import task, flow
from prefect.tasks import task_input_hash
from prefect.context import get_run_context

from src.data.data_processor import DataProcessor
from src.data.feature_engineering import FeatureEngineer
from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel
from src.monitoring.mlflow_tracking import MLflowTracker
from src.monitoring.monitoring import ModelMonitor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def load_and_process_data(
    data_path: str = "dataset/store-sales-time-series-forecasting"
) -> pd.DataFrame:
    """Load and process raw data."""
    logger.info("Starting data loading and processing")
    try:
        # Initialize processor
        processor = DataProcessor()
        
        # Load data
        logger.debug(f"Loading data from {data_path}")
        raw_data = processor.load_data(data_path)
        
        # Process data
        logger.debug("Processing data")
        processed_data = processor.preprocess_data(raw_data)
        
        logger.info("Data loading and processing completed successfully")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        raise

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features."""
    logger.info("Starting feature engineering")
    try:
        # Initialize engineer
        engineer = FeatureEngineer()
        
        # Create features
        logger.debug("Creating features")
        features = engineer.create_all_features(data)
        
        logger.info("Feature engineering completed successfully")
        return features
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

@task
def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: Optional[float] = None
) -> Dict[str, pd.DataFrame]:
    """Split data into train, validation, and test sets."""
    logger.info("Splitting data")
    try:
        # Initialize processor
        processor = DataProcessor()
        
        # Calculate split sizes
        if validation_size is None:
            validation_size = test_size
            
        first_split = 1 - test_size - validation_size
        
        # Sort by date
        data = data.sort_values('date')
        
        # Create splits
        train_idx = int(len(data) * first_split)
        val_idx = int(len(data) * (1 - test_size))
        
        splits = {
            'train': data.iloc[:train_idx],
            'validation': data.iloc[train_idx:val_idx],
            'test': data.iloc[val_idx:]
        }
        
        logger.debug(f"Split sizes - Train: {len(splits['train'])}, "
                    f"Val: {len(splits['validation'])}, "
                    f"Test: {len(splits['test'])}")
        
        return splits
        
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise

@task
def train_model(
    model_name: str,
    train_data: pd.DataFrame,
    validation_data: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    """Train a specific model."""
    logger.info(f"Training {model_name} model")
    try:
        # Initialize model
        model_classes = {
            'arima': ARIMAModel,
            'prophet': ProphetModel,
            'lstm': LSTMModel
        }
        
        if model_name not in model_classes:
            raise ValueError(f"Unknown model: {model_name}")
            
        model = model_classes[model_name]()
        
        # Train model
        logger.debug("Starting model training")
        metrics = model.train(train_data, validation_data)
        
        # Save model
        model_path = f"models/{model_name}/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.debug(f"Saving model to {model_path}")
        model.save(model_path)
        
        logger.info(f"{model_name} training completed successfully")
        logger.debug(f"Training metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

@task
def evaluate_model(
    model_name: str,
    test_data: pd.DataFrame
) -> Dict[str, float]:
    """Evaluate a trained model."""
    logger.info(f"Evaluating {model_name} model")
    try:
        # Get best model from MLflow
        mlflow_tracker = MLflowTracker()
        best_run = mlflow_tracker.get_best_run(
            metric_name="val_rmse",
            ascending=True
        )
        
        if best_run is None:
            raise ValueError(f"No trained model found for {model_name}")
            
        # Load model
        model_path = best_run.info.artifact_uri + "/model"
        model_classes = {
            'arima': ARIMAModel,
            'prophet': ProphetModel,
            'lstm': LSTMModel
        }
        model = model_classes[model_name]()
        model.load(model_path)
        
        # Evaluate
        logger.debug("Evaluating model on test data")
        metrics = model.evaluate(test_data)
        
        logger.info("Model evaluation completed successfully")
        logger.debug(f"Evaluation metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

@task
def compare_models(
    model_metrics: Dict[str, Dict[str, float]]
) -> str:
    """Compare models and select the best one."""
    logger.info("Comparing model performance")
    try:
        best_model = None
        best_rmse = float('inf')
        
        for model_name, metrics in model_metrics.items():
            rmse = metrics.get('rmse', float('inf'))
            logger.debug(f"{model_name} RMSE: {rmse}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name
                
        logger.info(f"Best model: {best_model} (RMSE: {best_rmse})")
        return best_model
        
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise

@flow
def train_and_evaluate_models(
    data_path: str = "dataset/store-sales-time-series-forecasting",
    models: Optional[List[str]] = None
) -> str:
    """Main training and evaluation flow."""
    logger.info("Starting training and evaluation flow")
    try:
        # Set default models if none specified
        if models is None:
            models = ['arima', 'prophet', 'lstm']
            
        # Process data
        data = load_and_process_data(data_path)
        features = engineer_features(data)
        splits = split_data(features)
        
        # Train and evaluate models
        model_metrics = {}
        for model_name in models:
            logger.info(f"Processing {model_name} model")
            
            # Train
            train_metrics = train_model(
                model_name,
                splits['train'],
                splits['validation']
            )
            
            # Evaluate
            test_metrics = evaluate_model(
                model_name,
                splits['test']
            )
            
            model_metrics[model_name] = {**train_metrics, **test_metrics}
            
        # Compare models
        best_model = compare_models(model_metrics)
        
        logger.info("Training and evaluation flow completed successfully")
        return best_model
        
    except Exception as e:
        logger.error(f"Error in training flow: {str(e)}")
        raise

@flow
def retrain_best_model(
    data: pd.DataFrame,
    schedule: str = "daily"
) -> None:
    """Automated retraining flow."""
    logger.info(f"Starting automated retraining ({schedule})")
    try:
        # Get best model
        mlflow_tracker = MLflowTracker()
        best_run = mlflow_tracker.get_best_run()
        
        if best_run is None:
            raise ValueError("No best model found")
            
        model_name = best_run.data.tags.get('model_name')
        logger.info(f"Best model to retrain: {model_name}")
        
        # Process new data
        features = engineer_features(data)
        splits = split_data(features)
        
        # Retrain model
        metrics = train_model(
            model_name,
            splits['train'],
            splits['validation']
        )
        
        # Monitor performance
        monitor = ModelMonitor()
        monitor.monitor_performance(model_name, metrics)
        
        logger.info("Automated retraining completed successfully")
        
    except Exception as e:
        logger.error(f"Error in retraining flow: {str(e)}")
        raise

@flow
def generate_predictions(
    prediction_days: int = 30,
    store_ids: Optional[List[int]] = None
) -> pd.DataFrame:
    """Generate predictions using best model."""
    logger.info(f"Generating predictions for {prediction_days} days")
    try:
        # Get best model
        mlflow_tracker = MLflowTracker()
        best_run = mlflow_tracker.get_best_run()
        
        if best_run is None:
            raise ValueError("No best model found")
            
        model_name = best_run.data.tags.get('model_name')
        logger.info(f"Using model: {model_name}")
        
        # Load model
        model_path = best_run.info.artifact_uri + "/model"
        model_classes = {
            'arima': ARIMAModel,
            'prophet': ProphetModel,
            'lstm': LSTMModel
        }
        model = model_classes[model_name]()
        model.load(model_path)
        
        # Load and process latest data
        data = load_and_process_data()
        features = engineer_features(data)
        
        # Filter stores if specified
        if store_ids is not None:
            features = features[features['store_nbr'].isin(store_ids)]
            logger.debug(f"Filtered to {len(store_ids)} stores")
            
        # Generate predictions
        logger.debug("Generating predictions")
        predictions = model.predict(features, prediction_days)
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'date': pd.date_range(
                start=features['date'].max() + timedelta(days=1),
                periods=prediction_days
            ),
            'predictions': predictions
        })
        
        # Monitor predictions
        monitor = ModelMonitor()
        monitor.monitor_predictions(predictions)
        
        logger.info("Predictions generated successfully")
        return pred_df
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise 