"""
Base model class for time series forecasting models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
from datetime import datetime

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class BaseTimeSeriesModel(ABC):
    """Abstract base class for time series models."""
    
    def __init__(self, name: str, config_path: str):
        """Initialize base model."""
        logger.info(f"Initializing {name} model")
        self.name = name
        self.config = self._load_config(config_path)
        logger.debug(f"Loaded configuration: {self.config}")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Get model-specific config
            if self.name in config:
                model_config = config[self.name]
            else:
                logger.warning(f"No specific configuration found for {self.name}")
                model_config = {}
                
            logger.debug(f"Loaded {self.name} configuration from {config_path}")
            return model_config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
            
    @abstractmethod
    def preprocess(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data for model.
        
        Args:
            data: Input data
            is_training: Whether preprocessing is for training
            
        Returns:
            Tuple of features and optional targets
        """
        pass
        
    @abstractmethod
    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            train_data: Training data
            validation_data: Optional validation data
            
        Returns:
            Training metrics
        """
        pass
        
    @abstractmethod
    def predict(
        self,
        data: pd.DataFrame,
        prediction_steps: int
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            data: Input data
            prediction_steps: Number of steps to predict
            
        Returns:
            Model predictions
        """
        pass
        
    def evaluate(
        self,
        test_data: pd.DataFrame,
        metrics: List[str] = ['mse', 'mae', 'mape']
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            test_data: Test data
            metrics: Metrics to calculate
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {self.name} model")
        try:
            # Get predictions
            logger.debug("Generating predictions for evaluation")
            predictions = self.predict(test_data, 1)
            
            # Calculate metrics
            logger.debug(f"Calculating metrics: {metrics}")
            results = {}
            
            if 'mse' in metrics:
                mse = np.mean((test_data['sales'].values - predictions) ** 2)
                results['mse'] = float(mse)
                results['rmse'] = float(np.sqrt(mse))
                
            if 'mae' in metrics:
                mae = np.mean(np.abs(test_data['sales'].values - predictions))
                results['mae'] = float(mae)
                
            if 'mape' in metrics:
                mape = np.mean(np.abs(
                    (test_data['sales'].values - predictions) / test_data['sales'].values
                )) * 100
                results['mape'] = float(mape)
                
            logger.info("Evaluation completed successfully")
            logger.debug(f"Evaluation metrics: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
            
    @abstractmethod
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        logger.info(f"Saving {self.name} model to {path}")
        
    @abstractmethod
    def load(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        logger.info(f"Loading {self.name} model from {path}")
        
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            Whether data is valid
        """
        logger.debug("Validating input data")
        try:
            # Check required columns
            required_columns = ['date', 'store_nbr', 'sales']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Check data types
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                raise ValueError("'date' column must be datetime")
                
            if not pd.api.types.is_numeric_dtype(data['store_nbr']):
                raise ValueError("'store_nbr' column must be numeric")
                
            if not pd.api.types.is_numeric_dtype(data['sales']):
                raise ValueError("'sales' column must be numeric")
                
            # Check for missing values
            missing_values = data[required_columns].isnull().sum()
            if missing_values.any():
                logger.warning(f"Missing values found: {missing_values}")
                
            # Check for negative sales
            negative_sales = (data['sales'] < 0).sum()
            if negative_sales > 0:
                logger.warning(f"Found {negative_sales} negative sales values")
                
            logger.debug("Data validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False
            
    def _check_data_continuity(self, data: pd.DataFrame) -> bool:
        """
        Check time series continuity.
        
        Args:
            data: Input data to check
            
        Returns:
            Whether data is continuous
        """
        logger.debug("Checking time series continuity")
        try:
            # Sort data
            data = data.sort_values(['store_nbr', 'date'])
            
            # Check date continuity for each store
            for store in data['store_nbr'].unique():
                store_data = data[data['store_nbr'] == store]
                date_range = pd.date_range(
                    start=store_data['date'].min(),
                    end=store_data['date'].max(),
                    freq='D'
                )
                
                missing_dates = set(date_range) - set(store_data['date'])
                if missing_dates:
                    logger.warning(
                        f"Store {store} has {len(missing_dates)} missing dates"
                    )
                    return False
                    
            logger.debug("Time series continuity check passed")
            return True
            
        except Exception as e:
            logger.error(f"Continuity check failed: {str(e)}")
            return False
            
    def _log_training_start(
        self,
        train_size: int,
        val_size: Optional[int] = None
    ):
        """Log training start information."""
        logger.info(f"Starting {self.name} model training")
        logger.debug(f"Training data size: {train_size}")
        if val_size:
            logger.debug(f"Validation data size: {val_size}")
            
    def _log_training_progress(
        self,
        epoch: int,
        metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log training progress."""
        progress_msg = f"Epoch {epoch} - "
        progress_msg += " - ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        
        if val_metrics:
            progress_msg += " - "
            progress_msg += " - ".join(f"val_{k}: {v:.4f}" for k, v in val_metrics.items())
            
        logger.debug(progress_msg)
        
    def _log_training_complete(
        self,
        final_metrics: Dict[str, float],
        training_time: float
    ):
        """Log training completion information."""
        logger.info(f"{self.name} model training completed in {training_time:.2f}s")
        logger.debug(f"Final metrics: {final_metrics}")
        
    def _log_prediction_info(
        self,
        data_size: int,
        prediction_steps: int
    ):
        """Log prediction information."""
        logger.info(
            f"Generating {prediction_steps} step predictions "
            f"for {data_size} samples"
        ) 