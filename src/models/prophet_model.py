"""
Prophet model for time series forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import joblib
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.base_model import BaseTimeSeriesModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ProphetModel(BaseTimeSeriesModel):
    """Prophet model for time series forecasting."""
    
    def __init__(self, config_path: str = "configs/model_configs/default.yaml"):
        """Initialize Prophet model."""
        logger.info(f"Initializing ProphetModel with config: {config_path}")
        super().__init__("prophet", config_path)
        self.models = {}  # Store-specific models
        
        # Set model parameters from config
        self.seasonality_mode = self.config.get('seasonality_mode', 'multiplicative')
        self.yearly_seasonality = self.config.get('yearly_seasonality', 'auto')
        self.weekly_seasonality = self.config.get('weekly_seasonality', 'auto')
        self.daily_seasonality = self.config.get('daily_seasonality', False)
        self.changepoint_prior_scale = self.config.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = self.config.get('seasonality_prior_scale', 10.0)
        self.holidays_prior_scale = self.config.get('holidays_prior_scale', 10.0)
        self.changepoint_range = self.config.get('changepoint_range', 0.8)
        
        logger.debug("ProphetModel initialized successfully")
        
    def _create_prophet_df(
        self,
        data: pd.DataFrame,
        store: int
    ) -> pd.DataFrame:
        """Create DataFrame in Prophet format."""
        logger.debug(f"Creating Prophet DataFrame for store {store}")
        try:
            # Create Prophet DataFrame
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(data.index),
                'y': data['sales'],
                'onpromotion': data.get('onpromotion', 0),  # Default to 0 if missing
                'dcoilwtico': data.get('dcoilwtico', 0),  # Default to 0 if missing
                'transactions': data.get('transactions', 0),  # Default to 0 if missing
                'holiday': data.get('holiday_type_Holiday', 0).astype(float)  # Convert to float
            })
            
            # Fill missing values
            prophet_df['onpromotion'] = prophet_df['onpromotion'].fillna(0)
            prophet_df['dcoilwtico'] = prophet_df['dcoilwtico'].ffill().bfill()
            prophet_df['transactions'] = prophet_df['transactions'].fillna(0)
            prophet_df['holiday'] = prophet_df['holiday'].fillna(0)
            
            logger.debug(f"Created Prophet DataFrame with shape: {prophet_df.shape}")
            return prophet_df
            
        except Exception as e:
            logger.error(f"Error creating Prophet DataFrame: {str(e)}")
            raise
            
    def _create_model(self, store: int) -> Prophet:
        """Create Prophet model with configuration."""
        logger.debug(f"Creating Prophet model for store {store}")
        try:
            # Create model with configuration
            model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                changepoint_range=self.changepoint_range
            )
            
            # Add regressors
            model.add_regressor('onpromotion')
            model.add_regressor('dcoilwtico')
            model.add_regressor('transactions')
            
            logger.debug(f"Created Prophet model with config: {self.config}")
            return model
            
        except Exception as e:
            logger.error(f"Error creating Prophet model: {str(e)}")
            raise
            
    def preprocess(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> Dict[int, pd.DataFrame]:
        """Preprocess data for Prophet model."""
        logger.info("Preprocessing data for Prophet")
        try:
            store_data = {}
            
            # Process each store
            for store in data['store_nbr'].unique():
                logger.debug(f"Processing store {store}")
                
                # Get store data
                store_df = data[data['store_nbr'] == store].copy()
                store_df.set_index('date', inplace=True)
                store_df.sort_index(inplace=True)
                
                # Create Prophet DataFrame
                prophet_df = self._create_prophet_df(store_df, store)
                store_data[store] = prophet_df
                
            logger.info(f"Preprocessed data for {len(store_data)} stores")
            return store_data
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
            
    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Train Prophet models for each store."""
        logger.info("Starting Prophet model training")
        metrics = {}
        
        try:
            # Preprocess data
            store_data = self.preprocess(train_data)
            
            # Train store-specific models
            for store, data in store_data.items():
                logger.info(f"Training model for store {store}")
                
                # Create and train model
                model = self._create_model(store)
                model.fit(data)
                self.models[store] = model
                
                # Calculate in-sample metrics
                predictions = model.predict(data)['yhat']
                mse = np.mean((data['y'] - predictions) ** 2)
                mae = np.mean(np.abs(data['y'] - predictions))
                mape = np.mean(np.abs((data['y'] - predictions) / data['y'])) * 100
                
                # Store metrics
                metrics[f'store_{store}_mse'] = float(mse)
                metrics[f'store_{store}_mae'] = float(mae)
                metrics[f'store_{store}_mape'] = float(mape)
                
                logger.debug(f"Calculated metrics: {metrics}")
                
            # Calculate average metrics
            avg_metrics = {}
            for metric in ['mse', 'mae', 'mape']:
                values = [v for k, v in metrics.items() if k.endswith(metric)]
                avg_metrics[f'avg_{metric}'] = float(np.mean(values))
                
            metrics.update(avg_metrics)
            
            logger.info("Prophet training completed successfully")
            logger.debug(f"Training metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
            
    def predict(
        self,
        data: pd.DataFrame,
        prediction_steps: int
    ) -> np.ndarray:
        """Generate predictions using Prophet models."""
        logger.info(f"Generating predictions for {prediction_steps} steps")
        try:
            if not self.models:
                raise ValueError("Models not trained")
                
            # Preprocess data
            store_data = self.preprocess(data, is_training=False)
            predictions = []
            
            # Generate predictions for each store
            for store, prophet_df in store_data.items():
                logger.debug(f"Predicting for store {store}")
                
                if store not in self.models:
                    raise ValueError(f"No model found for store {store}")
                    
                # Create future DataFrame
                model = self.models[store]
                future = model.make_future_dataframe(
                    periods=prediction_steps,
                    freq='D',
                    include_history=False
                )
                
                # Add regressors to future DataFrame
                future['onpromotion'] = prophet_df['onpromotion'].iloc[-1]  # Use last known value
                future['dcoilwtico'] = prophet_df['dcoilwtico'].iloc[-1]  # Use last known value
                future['transactions'] = prophet_df['transactions'].iloc[-1]  # Use last known value
                
                # Generate predictions
                forecast = model.predict(future)
                store_preds = forecast['yhat'].values[-prediction_steps:]
                predictions.append(store_preds)
                
            # Stack predictions into shape (prediction_steps, n_stores)
            predictions = np.stack(predictions, axis=1)
            logger.debug(f"Generated predictions with shape: {predictions.shape}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
            
    def save(self, path: str):
        """Save Prophet models to disk."""
        logger.info(f"Saving models to {path}")
        try:
            if not self.models:
                raise ValueError("No models to save")
                
            # Create directory if it doesn't exist
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save models and configuration
            save_dict = {
                'models': self.models,
                'config': self.config
            }
            
            joblib.dump(save_dict, path)
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
            
    def load(self, path: str):
        """Load Prophet models from disk."""
        logger.info(f"Loading models from {path}")
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")
                
            # Load models and configuration
            save_dict = joblib.load(path)
            self.models = save_dict['models']
            self.config = save_dict['config']
            
            # Restore model parameters
            self.seasonality_mode = self.config.get('seasonality_mode', 'multiplicative')
            self.yearly_seasonality = self.config.get('yearly_seasonality', 'auto')
            self.weekly_seasonality = self.config.get('weekly_seasonality', 'auto')
            self.daily_seasonality = self.config.get('daily_seasonality', False)
            self.changepoint_prior_scale = self.config.get('changepoint_prior_scale', 0.05)
            self.seasonality_prior_scale = self.config.get('seasonality_prior_scale', 10.0)
            self.holidays_prior_scale = self.config.get('holidays_prior_scale', 10.0)
            self.changepoint_range = self.config.get('changepoint_range', 0.8)
            
            logger.info(f"Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
            
    def get_components(self, store_id: int) -> Dict[str, pd.Series]:
        """Get forecast components for a specific store."""
        logger.info(f"Getting components for store {store_id}")
        try:
            if store_id not in self.models:
                raise ValueError(f"No model found for store {store_id}")
                
            model = self.models[store_id]
            forecast = model.predict()
            
            components = {
                'trend': forecast['trend'],
                'yearly': forecast['yearly'],
                'weekly': forecast['weekly'],
                'holidays': forecast.get('holidays', pd.Series(0, index=forecast.index))
            }
            
            logger.debug("Retrieved forecast components")
            return components
            
        except Exception as e:
            logger.error(f"Error getting components: {str(e)}")
            raise
            
    def plot_components(self, store_id: int, output_dir: str):
        """Plot forecast components for a specific store."""
        logger.info(f"Plotting components for store {store_id}")
        try:
            if store_id not in self.models:
                raise ValueError(f"No model found for store {store_id}")
                
            model = self.models[store_id]
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Plot components
            fig = model.plot_components(model.predict())
            plt.title(f"Forecast Components - Store {store_id}")
            
            # Save plot
            plot_path = output_path / f"components_store_{store_id}.png"
            plt.savefig(plot_path)
            plt.close()
            
            # Plot cross-validation metrics
            cv_results = cross_validation(
                model,
                initial='30 days',  # Shorter initial period
                period='7 days',   # Shorter period
                horizon='7 days'   # Shorter horizon
            )
            metrics = performance_metrics(cv_results)
            
            plt.figure(figsize=(10, 6))
            plt.plot(cv_results['cutoff'], cv_results['rmse'], label='RMSE')
            plt.title(f"Cross-Validation Metrics - Store {store_id}")
            plt.xlabel('Cutoff Date')
            plt.ylabel('RMSE')
            plt.legend()
            
            # Save plot
            plot_path = output_path / f"cv_metrics_store_{store_id}.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.debug(f"Saved component plots to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error plotting components: {str(e)}")
            raise
            
    def get_forecast_uncertainty(
        self,
        store_id: int,
        prediction_steps: int
    ) -> Dict[str, np.ndarray]:
        """Get forecast uncertainty intervals."""
        logger.info(f"Getting forecast uncertainty for store {store_id}")
        try:
            if store_id not in self.models:
                raise ValueError(f"No model found for store {store_id}")
                
            model = self.models[store_id]
            
            # Make future DataFrame
            future = model.make_future_dataframe(
                periods=prediction_steps,
                freq='D'
            )
            
            # Add regressors to future DataFrame
            future['onpromotion'] = 0  # Default values for regressors
            future['dcoilwtico'] = 0
            future['transactions'] = 0
            
            # Generate forecast with uncertainty
            forecast = model.predict(future)
            uncertainty = {
                'yhat_lower': forecast['yhat_lower'].values[-prediction_steps:],
                'yhat_upper': forecast['yhat_upper'].values[-prediction_steps:]
            }
            
            logger.debug("Retrieved forecast uncertainty")
            return uncertainty
            
        except Exception as e:
            logger.error(f"Error getting forecast uncertainty: {str(e)}")
            raise 