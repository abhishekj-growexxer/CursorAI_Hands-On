"""
Training pipeline for time series models.
"""

import os
import mlflow
import optuna
from typing import Dict, Optional, Type, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
from pathlib import Path

from src.models.base_model import BaseTimeSeriesModel
from src.monitoring.mlflow_tracking import MLflowTracker
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TrainingPipeline:
    """Training pipeline for time series models."""
    
    def __init__(
        self,
        config_path: str = "configs/training_configs/default.yaml",
        experiment_name: str = "store_sales_forecasting"
    ):
        """Initialize training pipeline."""
        logger.info(f"Initializing TrainingPipeline with config: {config_path}")
        self.config_path = config_path
        self.config = self._load_config()
        self.experiment_name = experiment_name
        self.mlflow_tracker = MLflowTracker(experiment_name=experiment_name)
        logger.debug("TrainingPipeline initialized successfully")
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.debug(f"Successfully loaded config from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
            
    def train_model(
        self,
        model_class: Type[BaseTimeSeriesModel],
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Train a single model."""
        logger.info(f"Training {model_class.__name__}")
        try:
            # Initialize model
            model = model_class(config_path=self.config_path)
            
            # Train model
            metrics = model.train(train_data, validation_data)
            
            # Log metrics
            self.mlflow_tracker.log_metrics(metrics)
            
            logger.info(f"Model training completed with metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def cross_validate(
        self,
        model_class: Type[BaseTimeSeriesModel],
        data: pd.DataFrame,
        n_folds: int = 5
    ) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        logger.info(f"Starting {n_folds}-fold cross-validation")
        try:
            # Sort data by date
            data = data.sort_values('date')
            
            # Calculate fold size
            fold_size = len(data) // n_folds
            metrics_list = []
            
            # Perform cross-validation
            for fold in range(n_folds):
                logger.debug(f"Processing fold {fold + 1}/{n_folds}")
                
                # Split data
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size
                
                train_data = pd.concat([
                    data.iloc[:val_start],
                    data.iloc[val_end:]
                ])
                val_data = data.iloc[val_start:val_end]
                
                # Train and evaluate
                fold_metrics = self.train_model(model_class, train_data, val_data)
                metrics_list.append(fold_metrics)
                
            # Calculate average metrics
            avg_metrics = {}
            for metric in metrics_list[0].keys():
                values = [m[metric] for m in metrics_list]
                avg_metrics[metric] = float(np.mean(values))
                avg_metrics[f"{metric}_std"] = float(np.std(values))
                
            logger.info(f"Cross-validation completed with metrics: {avg_metrics}")
            return avg_metrics
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            raise
            
    def hyperparameter_search(
        self,
        model_class: Type[BaseTimeSeriesModel],
        param_grid: Dict[str, List[Union[int, float, str]]],
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        n_trials: int = 10
    ) -> Dict[str, Union[float, Dict]]:
        """Perform hyperparameter optimization."""
        logger.info(f"Starting hyperparameter search for {model_class.__name__}")
        logger.debug(f"Parameter grid: {param_grid}")
        
        def objective(trial):
            """Optuna objective function."""
            # Create model config
            config = {}
            for param, values in param_grid.items():
                if isinstance(values[0], int):
                    config[param] = trial.suggest_int(param, min(values), max(values))
                elif isinstance(values[0], float):
                    config[param] = trial.suggest_float(param, min(values), max(values))
                else:
                    config[param] = trial.suggest_categorical(param, values)
                    
            logger.debug(f"Trial config: {config}")
            
            try:
                # Create and train model
                model = model_class(config_path=self.config_path)
                metrics = model.train(train_data, validation_data)
                
                # Use primary metric for optimization
                return metrics.get('rmse', float('inf'))
                
            except Exception as e:
                logger.error(f"Error in trial: {str(e)}")
                return float('inf')
                
        try:
            # Create study
            study = optuna.create_study(direction='minimize')
            
            # Run optimization
            logger.info(f"Running {n_trials} optimization trials")
            study.optimize(objective, n_trials=n_trials)
            
            # Get best results
            best_params = study.best_params
            best_value = study.best_value
            
            results = {
                'best_value': float(best_value),
                'best_params': best_params
            }
            
            logger.info(f"Hyperparameter search completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error during hyperparameter search: {str(e)}")
            raise
            
    def compare_models(
        self,
        models: List[Type[BaseTimeSeriesModel]],
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple models."""
        logger.info(f"Comparing models: {[m.__name__ for m in models]}")
        try:
            results = {}
            
            # Train and evaluate each model
            for model_class in models:
                model_name = model_class.__name__
                logger.info(f"Training {model_name}")
                
                metrics = self.train_model(model_class, train_data, validation_data)
                results[model_name] = metrics
                
            logger.info("Model comparison completed")
            return results
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
            
    def get_best_model(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'rmse'
    ) -> str:
        """Get the best performing model."""
        logger.info(f"Finding best model based on {metric}")
        try:
            # Get metric values for each model
            metric_values = {
                model: metrics.get(metric, float('inf'))
                for model, metrics in results.items()
            }
            
            # Find best model
            best_model = min(metric_values.items(), key=lambda x: x[1])[0]
            
            logger.info(f"Best model: {best_model} with {metric}={metric_values[best_model]}")
            return best_model
            
        except Exception as e:
            logger.error(f"Error finding best model: {str(e)}")
            raise 