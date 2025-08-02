"""
MLflow tracking module for experiment tracking and model registry.
"""

import os
from typing import Dict, List, Optional, Union
import mlflow
from mlflow.entities import Run
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MLflowTracker:
    """MLflow experiment tracking and model registry."""
    
    def __init__(
        self,
        experiment_name: str = "store_sales_forecasting",
        tracking_uri: Optional[str] = None
    ):
        """Initialize MLflow tracker."""
        logger.info(f"Initializing MLflowTracker with experiment: {experiment_name}")
        
        # Set tracking URI
        tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        logger.debug(f"Set tracking URI: {tracking_uri}")
        
        # Set experiment
        self.experiment_name = experiment_name
        self.experiment = self.create_experiment_if_not_exists(experiment_name)
        logger.debug(f"Using experiment ID: {self.experiment.experiment_id}")
        
    def create_experiment_if_not_exists(self, experiment_name: str) -> mlflow.entities.Experiment:
        """Create MLflow experiment if it doesn't exist."""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.info(f"Creating new experiment: {experiment_name}")
                experiment_id = mlflow.create_experiment(experiment_name)
                experiment = mlflow.get_experiment(experiment_id)
            else:
                logger.debug(f"Using existing experiment: {experiment_name}")
            return experiment
        except Exception as e:
            logger.error(f"Error creating/getting experiment: {str(e)}")
            raise
            
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False
    ) -> mlflow.ActiveRun:
        """Start an MLflow run."""
        logger.info(f"Starting MLflow run: {run_name or 'unnamed'}")
        try:
            return mlflow.start_run(
                experiment_id=self.experiment.experiment_id,
                run_name=run_name,
                nested=nested
            )
        except Exception as e:
            logger.error(f"Error starting run: {str(e)}")
            raise
            
    def log_params(self, params: Dict[str, any]):
        """Log parameters to MLflow."""
        logger.debug(f"Logging parameters: {params}")
        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.error(f"Error logging parameters: {str(e)}")
            raise
            
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics to MLflow."""
        logger.debug(f"Logging metrics: {metrics}")
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            raise
            
    def log_artifact(self, local_path: str):
        """Log an artifact to MLflow."""
        logger.debug(f"Logging artifact: {local_path}")
        try:
            mlflow.log_artifact(local_path)
        except Exception as e:
            logger.error(f"Error logging artifact: {str(e)}")
            raise
            
    def log_model(
        self,
        model: object,
        artifact_path: str,
        conda_env: Optional[Dict[str, any]] = None
    ):
        """Log a model to MLflow."""
        logger.info(f"Logging model to {artifact_path}")
        try:
            mlflow.pyfunc.log_model(
                artifact_path,
                python_model=model,
                conda_env=conda_env
            )
        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            raise
            
    def log_figure(self, figure: plt.Figure, artifact_path: str):
        """Log a matplotlib figure to MLflow."""
        logger.debug(f"Logging figure to {artifact_path}")
        try:
            mlflow.log_figure(figure, artifact_path)
        except Exception as e:
            logger.error(f"Error logging figure: {str(e)}")
            raise
            
    def log_predictions(
        self,
        predictions: np.ndarray,
        targets: Optional[np.ndarray] = None,
        artifact_path: str = "predictions.csv"
    ):
        """Log predictions to MLflow."""
        logger.debug(f"Logging predictions to {artifact_path}")
        try:
            # Create DataFrame
            df = pd.DataFrame({'predictions': predictions})
            if targets is not None:
                df['targets'] = targets
                
            # Save to CSV
            temp_path = f"/tmp/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{artifact_path}"
            df.to_csv(temp_path, index=False)
            
            # Log to MLflow
            self.log_artifact(temp_path)
            
            # Clean up
            os.remove(temp_path)
            
        except Exception as e:
            logger.error(f"Error logging predictions: {str(e)}")
            raise
            
    def get_best_run(
        self,
        metric_name: str = "test_rmse",
        ascending: bool = True
    ) -> Optional[Run]:
        """Get the best run based on a metric."""
        logger.info(f"Finding best run by {metric_name} ({'ascending' if ascending else 'descending'})")
        try:
            # Get all runs
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string=f"metrics.{metric_name} IS NOT NULL"
            )
            
            if runs.empty:
                logger.warning("No runs found with specified metric")
                return None
                
            # Sort by metric
            runs = runs.sort_values(
                f"metrics.{metric_name}",
                ascending=ascending
            )
            
            # Get best run
            best_run_id = runs.iloc[0]['run_id']
            best_run = mlflow.get_run(best_run_id)
            
            logger.info(f"Found best run: {best_run_id}")
            logger.debug(f"Best run metric: {metric_name}={best_run.data.metrics.get(metric_name)}")
            return best_run
            
        except Exception as e:
            logger.error(f"Error getting best run: {str(e)}")
            raise
            
    def compare_runs(
        self,
        metric_names: List[str],
        experiment_name: Optional[str] = None,
        max_runs: int = 5
    ) -> pd.DataFrame:
        """Compare metrics across multiple runs."""
        logger.info(f"Comparing runs for metrics: {metric_names}")
        try:
            # Get experiment
            if experiment_name is None:
                experiment_name = self.experiment_name
                
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment not found: {experiment_name}")
                
            # Get runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_runs
            )
            
            if runs.empty:
                logger.warning("No runs found")
                return pd.DataFrame()
                
            # Extract metrics
            metrics_df = runs[['run_id', 'start_time'] + [f"metrics.{m}" for m in metric_names]]
            metrics_df = metrics_df.rename(
                columns={f"metrics.{m}": m for m in metric_names}
            )
            
            logger.info(f"Compared {len(metrics_df)} runs")
            return metrics_df
            
        except Exception as e:
            logger.error(f"Error comparing runs: {str(e)}")
            raise
            
    def plot_metric_comparison(
        self,
        metric_names: List[str],
        experiment_name: Optional[str] = None,
        max_runs: int = 5
    ) -> plt.Figure:
        """Plot metric comparison across runs."""
        logger.info(f"Plotting metric comparison for: {metric_names}")
        try:
            # Get comparison data
            metrics_df = self.compare_runs(
                metric_names,
                experiment_name,
                max_runs
            )
            
            if metrics_df.empty:
                logger.warning("No data to plot")
                return None
                
            # Create figure
            fig, axes = plt.subplots(
                len(metric_names),
                1,
                figsize=(10, 4 * len(metric_names))
            )
            if len(metric_names) == 1:
                axes = [axes]
                
            # Plot each metric
            for ax, metric in zip(axes, metric_names):
                ax.bar(range(len(metrics_df)), metrics_df[metric])
                ax.set_title(f"{metric} Comparison")
                ax.set_xlabel("Run")
                ax.set_ylabel(metric)
                
            plt.tight_layout()
            
            logger.debug("Created metric comparison plot")
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting metric comparison: {str(e)}")
            raise 