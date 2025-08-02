"""
Monitoring system for model and system health tracking.
"""

import os
import json
import psutil
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class MonitoringThresholds:
    """Thresholds for monitoring alerts."""
    rmse_threshold: float = 1.0
    mape_threshold: float = 10.0
    drift_p_value: float = 0.05
    cpu_threshold: float = 90.0
    memory_threshold: float = 90.0
    disk_threshold: float = 90.0

class ModelMonitor:
    """Monitor model performance and system health."""
    
    def __init__(
        self,
        config_path: str = "configs/monitoring_thresholds.json",
        log_dir: str = "logs/monitoring"
    ):
        """Initialize monitoring system."""
        logger.info("Initializing ModelMonitor")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load thresholds
        self.thresholds = self._load_thresholds(config_path)
        logger.debug(f"Loaded thresholds: {asdict(self.thresholds)}")
        
        # Create log files
        self._initialize_log_files()
        logger.debug("Initialized log files")
        
    def _load_thresholds(self, config_path: str) -> MonitoringThresholds:
        """Load monitoring thresholds from config."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.debug(f"Loaded thresholds from {config_path}")
                return MonitoringThresholds(**config)
            return MonitoringThresholds()
        except Exception as e:
            logger.error(f"Error loading thresholds: {str(e)}")
            return MonitoringThresholds()
            
    def _initialize_log_files(self):
        """Initialize log files for different metrics."""
        log_files = [
            'prediction_metrics.jsonl',
            'drift_metrics.jsonl',
            'performance_metrics.jsonl',
            'system_metrics.jsonl'
        ]
        
        for file in log_files:
            log_file = self.log_dir / file
            if not log_file.exists():
                log_file.touch()
                logger.debug(f"Created log file: {file}")
                
    def _log_metrics(
        self,
        metric_type: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        """Log metrics to JSONL file."""
        try:
            timestamp = timestamp or datetime.now()
            log_entry = {
                'timestamp': timestamp.isoformat(),
                **metrics
            }
            
            log_file = self.log_dir / f"{metric_type}.jsonl"
            with open(log_file, 'a') as f:
                json.dump(log_entry, f)
                f.write('\n')
                
            logger.debug(f"Logged {metric_type} metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            
    def monitor_predictions(
        self,
        predictions: np.ndarray,
        actuals: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Monitor prediction statistics and performance."""
        logger.info("Monitoring predictions")
        try:
            # Calculate prediction statistics
            stats = {
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions))
            }
            
            # Calculate performance metrics if actuals available
            if actuals is not None:
                mse = np.mean((predictions - actuals) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - actuals))
                mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
                
                stats.update({
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'mape': float(mape)
                })
                
                # Check thresholds
                if rmse > self.thresholds.rmse_threshold:
                    logger.warning(f"RMSE ({rmse:.2f}) exceeds threshold ({self.thresholds.rmse_threshold})")
                if mape > self.thresholds.mape_threshold:
                    logger.warning(f"MAPE ({mape:.2f}%) exceeds threshold ({self.thresholds.mape_threshold}%)")
                    
            # Log metrics
            self._log_metrics('prediction_metrics', stats, timestamp)
            
            # Plot predictions
            if actuals is not None:
                self._plot_predictions(predictions, actuals, timestamp)
                
            logger.info("Prediction monitoring completed")
            return stats
            
        except Exception as e:
            logger.error(f"Error monitoring predictions: {str(e)}")
            raise
            
    def monitor_data_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Monitor data drift using statistical tests."""
        logger.info("Monitoring data drift")
        try:
            drift_metrics = {}
            
            # Check each numerical column
            for col in reference_data.select_dtypes(include=[np.number]).columns:
                # Perform Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(
                    reference_data[col],
                    current_data[col]
                )
                
                drift_metrics[f"{col}_ks_stat"] = float(ks_stat)
                drift_metrics[f"{col}_p_value"] = float(p_value)
                
                # Check for significant drift
                if p_value < self.thresholds.drift_p_value:
                    logger.warning(f"Significant drift detected in {col} (p-value: {p_value:.4f})")
                    
            # Log metrics
            self._log_metrics('drift_metrics', drift_metrics, timestamp)
            
            # Plot distributions
            self._plot_distributions(reference_data, current_data, timestamp)
            
            logger.info("Data drift monitoring completed")
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Error monitoring data drift: {str(e)}")
            raise
            
    def monitor_performance(
        self,
        model_name: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Monitor model performance metrics."""
        logger.info(f"Monitoring {model_name} performance")
        try:
            # Add model name to metrics
            metrics = {f"{model_name}_{k}": v for k, v in metrics.items()}
            
            # Check performance thresholds
            if 'rmse' in metrics and metrics['rmse'] > self.thresholds.rmse_threshold:
                logger.warning(f"Model RMSE ({metrics['rmse']:.2f}) exceeds threshold")
            if 'mape' in metrics and metrics['mape'] > self.thresholds.mape_threshold:
                logger.warning(f"Model MAPE ({metrics['mape']:.2f}%) exceeds threshold")
                
            # Log metrics
            self._log_metrics('performance_metrics', metrics, timestamp)
            
            # Plot metrics history
            self._plot_metrics_history(model_name, timestamp)
            
            logger.info("Performance monitoring completed")
            
        except Exception as e:
            logger.error(f"Error monitoring performance: {str(e)}")
            raise
            
    def monitor_system_resources(
        self,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Monitor system resource utilization."""
        logger.info("Monitoring system resources")
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used = memory.used / (1024 ** 3)  # GB
            memory_available = memory.available / (1024 ** 3)  # GB
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_used = disk.used / (1024 ** 3)  # GB
            disk_free = disk.free / (1024 ** 3)  # GB
            
            metrics = {
                'cpu_percent': float(cpu_percent),
                'memory_percent': float(memory_percent),
                'memory_used_gb': float(memory_used),
                'memory_available_gb': float(memory_available),
                'disk_percent': float(disk_percent),
                'disk_used_gb': float(disk_used),
                'disk_free_gb': float(disk_free)
            }
            
            # Check resource thresholds
            if cpu_percent > self.thresholds.cpu_threshold:
                logger.warning(f"CPU usage ({cpu_percent}%) exceeds threshold")
            if memory_percent > self.thresholds.memory_threshold:
                logger.warning(f"Memory usage ({memory_percent}%) exceeds threshold")
            if disk_percent > self.thresholds.disk_threshold:
                logger.warning(f"Disk usage ({disk_percent}%) exceeds threshold")
                
            # Log metrics
            self._log_metrics('system_metrics', metrics, timestamp)
            
            logger.info("System resource monitoring completed")
            return metrics
            
        except Exception as e:
            logger.error(f"Error monitoring system resources: {str(e)}")
            raise
            
    def get_metrics_history(
        self,
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Retrieve metrics history from log files."""
        logger.info(f"Retrieving {metric_type} metrics history")
        try:
            log_file = self.log_dir / f"{metric_type}.jsonl"
            if not log_file.exists():
                logger.warning(f"No history found for {metric_type}")
                return pd.DataFrame()
                
            # Read log file
            metrics = []
            with open(log_file, 'r') as f:
                for line in f:
                    metrics.append(json.loads(line.strip()))
                    
            # Convert to DataFrame
            df = pd.DataFrame(metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by time range
            if start_time:
                df = df[df['timestamp'] >= start_time]
            if end_time:
                df = df[df['timestamp'] <= end_time]
                
            logger.debug(f"Retrieved {len(df)} metrics records")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving metrics history: {str(e)}")
            raise
            
    def _plot_predictions(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        timestamp: datetime
    ) -> None:
        """Plot predictions vs actuals."""
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(actuals, label='Actual')
            plt.plot(predictions, label='Predicted')
            plt.title('Predictions vs Actuals')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            
            # Save plot
            plot_file = self.log_dir / f"predictions_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file)
            plt.close()
            
            logger.debug(f"Saved predictions plot: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")
            
    def _plot_distributions(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        timestamp: datetime
    ) -> None:
        """Plot feature distributions."""
        try:
            numerical_cols = reference_data.select_dtypes(include=[np.number]).columns
            n_cols = len(numerical_cols)
            
            plt.figure(figsize=(15, 5 * ((n_cols + 1) // 2)))
            for i, col in enumerate(numerical_cols, 1):
                plt.subplot((n_cols + 1) // 2, 2, i)
                plt.hist(reference_data[col], alpha=0.5, label='Reference')
                plt.hist(current_data[col], alpha=0.5, label='Current')
                plt.title(f'{col} Distribution')
                plt.legend()
                
            plt.tight_layout()
            
            # Save plot
            plot_file = self.log_dir / f"distributions_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file)
            plt.close()
            
            logger.debug(f"Saved distributions plot: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error plotting distributions: {str(e)}")
            
    def _plot_metrics_history(
        self,
        model_name: str,
        timestamp: datetime
    ) -> None:
        """Plot metrics history."""
        try:
            # Get metrics history
            history = self.get_metrics_history('performance_metrics')
            if history.empty:
                return
                
            # Plot metrics
            metrics = ['rmse', 'mae', 'mape']
            plt.figure(figsize=(15, 5))
            
            for metric in metrics:
                col = f"{model_name}_{metric}"
                if col in history.columns:
                    plt.plot(history['timestamp'], history[col], label=metric.upper())
                    
            plt.title(f'{model_name} Performance History')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.xticks(rotation=45)
            
            # Save plot
            plot_file = self.log_dir / f"performance_{model_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file)
            plt.close()
            
            logger.debug(f"Saved performance history plot: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error plotting metrics history: {str(e)}")
            
    def check_thresholds(
        self,
        metrics: Dict[str, float],
        thresholds: Dict[str, float]
    ) -> Dict[str, bool]:
        """Check if metrics exceed thresholds."""
        logger.debug("Checking metric thresholds")
        try:
            alerts = {}
            for metric, value in metrics.items():
                if metric in thresholds:
                    exceeded = value > thresholds[metric]
                    alerts[metric] = exceeded
                    if exceeded:
                        logger.warning(f"{metric} ({value:.2f}) exceeds threshold ({thresholds[metric]})")
            return alerts
        except Exception as e:
            logger.error(f"Error checking thresholds: {str(e)}")
            raise 