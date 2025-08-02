"""
Central dashboard control center.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
import os
import json
from src.monitoring.prefect_workflows import (
    train_and_evaluate_models,
    retrain_best_model,
    generate_predictions
)
from src.monitoring.mlflow_tracking import MLflowTracker
from src.monitoring.monitoring import ModelMonitor

# Page config
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize components
mlflow_tracker = MLflowTracker()
model_monitor = ModelMonitor()

def main():
    """Main dashboard function."""
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Training", "Predictions", "Monitoring", "Settings"]
    )
    
    # Page content
    if page == "Overview":
        show_overview_page()
    elif page == "Training":
        show_training_page()
    elif page == "Predictions":
        show_predictions_page()
    elif page == "Monitoring":
        show_monitoring_page()
    else:
        show_settings_page()

def show_overview_page():
    """Show overview page."""
    st.title("Sales Forecasting Dashboard")
    
    # System status
    st.header("System Status")
    col1, col2, col3 = st.columns(3)
    
    # Get system metrics
    system_metrics = model_monitor.monitor_system_resources()
    
    with col1:
        st.metric(
            "CPU Usage",
            f"{system_metrics['cpu_percent']}%"
        )
    with col2:
        st.metric(
            "Memory Usage",
            f"{system_metrics['memory_percent']}%"
        )
    with col3:
        st.metric(
            "Disk Usage",
            f"{system_metrics['disk_percent']}%"
        )
        
    # Model performance summary
    st.header("Model Performance Summary")
    
    # Get best model metrics
    best_run = mlflow_tracker.get_best_run()
    if best_run:
        metrics = best_run.data.metrics
        model_name = best_run.data.tags.get('model_name', 'Unknown')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Best Model", model_name)
        with col2:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
        with col3:
            st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
            
    # Recent predictions
    st.header("Recent Predictions")
    predictions_history = model_monitor.get_metrics_history(
        'prediction_metrics',
        start_time=datetime.now() - timedelta(days=7)
    )
    
    if not predictions_history.empty:
        fig = px.line(
            predictions_history,
            x='timestamp',
            y=['mean', 'min', 'max'],
            title='Prediction Trends'
        )
        st.plotly_chart(fig)

def show_training_page():
    """Show training page."""
    st.title("Model Training")
    
    # Training controls
    st.header("Training Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        models = st.multiselect(
            "Select Models to Train",
            ["arima", "prophet", "lstm"],
            default=["arima", "prophet", "lstm"]
        )
        
    with col2:
        if st.button("Start Training"):
            with st.spinner("Training models..."):
                best_model = train_and_evaluate_models(models=models)
                st.success(f"Training completed! Best model: {best_model}")
                
    # Training history
    st.header("Training History")
    
    # Get training metrics for each model
    for model in ["arima", "prophet", "lstm"]:
        metrics_history = model_monitor.get_metrics_history(
            f'{model}_performance'
        )
        
        if not metrics_history.empty:
            fig = px.line(
                metrics_history,
                x='timestamp',
                y=['rmse', 'mae', 'mape'],
                title=f'{model.upper()} Performance History'
            )
            st.plotly_chart(fig)
            
    # Automated retraining
    st.header("Automated Retraining")
    
    col1, col2 = st.columns(2)
    
    with col1:
        schedule = st.selectbox(
            "Retraining Schedule",
            ["Daily", "Weekly", "Monthly"]
        )
        
    with col2:
        if st.button("Set Up Automated Retraining"):
            # TODO: Implement automated retraining setup
            st.info("Automated retraining will be set up")

def show_predictions_page():
    """Show predictions page."""
    st.title("Sales Predictions")
    
    # Prediction controls
    st.header("Generate Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        days = st.number_input(
            "Number of Days to Predict",
            min_value=1,
            max_value=90,
            value=30
        )
        
    with col2:
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                predictions = generate_predictions(prediction_days=days)
                
                # Plot predictions
                fig = px.line(
                    predictions,
                    x='date',
                    y='predictions',
                    title='Sales Predictions'
                )
                st.plotly_chart(fig)
                
                # Show predictions table
                st.dataframe(predictions)
                
    # Prediction history
    st.header("Prediction History")
    
    predictions_history = model_monitor.get_metrics_history(
        'prediction_metrics'
    )
    
    if not predictions_history.empty:
        # Create subplot with metrics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Prediction Mean",
                "Prediction Std",
                "Prediction Range",
                "Prediction Error"
            )
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=predictions_history['timestamp'],
                y=predictions_history['mean'],
                name="Mean"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=predictions_history['timestamp'],
                y=predictions_history['std'],
                name="Std"
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=predictions_history['timestamp'],
                y=predictions_history['max'],
                name="Max",
                line=dict(dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=predictions_history['timestamp'],
                y=predictions_history['min'],
                name="Min",
                line=dict(dash='dash')
            ),
            row=2, col=1
        )
        
        if 'mape' in predictions_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=predictions_history['timestamp'],
                    y=predictions_history['mape'],
                    name="MAPE"
                ),
                row=2, col=2
            )
            
        fig.update_layout(height=800)
        st.plotly_chart(fig)

def show_monitoring_page():
    """Show monitoring page."""
    st.title("Model Monitoring")
    
    # Data drift monitoring
    st.header("Data Drift Monitoring")
    
    drift_metrics = model_monitor.get_metrics_history(
        'drift_metrics'
    )
    
    if not drift_metrics.empty:
        # Plot drift metrics
        cols = [col for col in drift_metrics.columns if col != 'timestamp']
        for col in cols:
            if col.endswith('_ks_stat'):
                feature = col.replace('_ks_stat', '')
                fig = go.Figure()
                
                # Add KS statistic
                fig.add_trace(
                    go.Scatter(
                        x=drift_metrics['timestamp'],
                        y=drift_metrics[col],
                        name="KS Statistic"
                    )
                )
                
                # Add p-value
                fig.add_trace(
                    go.Scatter(
                        x=drift_metrics['timestamp'],
                        y=drift_metrics[f'{feature}_p_value'],
                        name="P-Value"
                    )
                )
                
                fig.update_layout(title=f"Drift Metrics - {feature}")
                st.plotly_chart(fig)
                
    # Performance monitoring
    st.header("Performance Monitoring")
    
    # Get best model
    best_run = mlflow_tracker.get_best_run()
    if best_run:
        model_name = best_run.data.tags.get('model_name', 'Unknown')
        performance_metrics = model_monitor.get_metrics_history(
            f'{model_name}_performance'
        )
        
        if not performance_metrics.empty:
            # Plot performance metrics
            fig = px.line(
                performance_metrics,
                x='timestamp',
                y=['rmse', 'mae', 'mape'],
                title=f'{model_name} Performance Metrics'
            )
            st.plotly_chart(fig)
            
    # System monitoring
    st.header("System Monitoring")
    
    system_metrics = model_monitor.get_metrics_history(
        'system_metrics'
    )
    
    if not system_metrics.empty:
        # Create subplot with system metrics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "CPU Usage",
                "Memory Usage",
                "Memory Details",
                "Disk Usage"
            )
        )
        
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=system_metrics['timestamp'],
                y=system_metrics['cpu_percent'],
                name="CPU %"
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=system_metrics['timestamp'],
                y=system_metrics['memory_percent'],
                name="Memory %"
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=system_metrics['timestamp'],
                y=system_metrics['memory_used_gb'],
                name="Used Memory (GB)"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=system_metrics['timestamp'],
                y=system_metrics['memory_available_gb'],
                name="Available Memory (GB)"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=system_metrics['timestamp'],
                y=system_metrics['disk_percent'],
                name="Disk %"
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800)
        st.plotly_chart(fig)

def show_settings_page():
    """Show settings page."""
    st.title("Settings")
    
    # MLflow settings
    st.header("MLflow Settings")
    
    mlflow_uri = st.text_input(
        "MLflow Tracking URI",
        value=os.getenv('MLFLOW_TRACKING_URI', 'mlruns')
    )
    
    if st.button("Update MLflow Settings"):
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
        st.success("MLflow settings updated!")
        
    # Monitoring settings
    st.header("Monitoring Settings")
    
    # Threshold settings
    st.subheader("Alert Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rmse_threshold = st.number_input(
            "RMSE Threshold",
            value=1.0,
            step=0.1
        )
        
        mape_threshold = st.number_input(
            "MAPE Threshold",
            value=10.0,
            step=0.1
        )
        
    with col2:
        drift_threshold = st.number_input(
            "Drift P-value Threshold",
            value=0.05,
            step=0.01
        )
        
        resource_threshold = st.number_input(
            "Resource Usage Threshold (%)",
            value=90.0,
            step=1.0
        )
        
    if st.button("Update Thresholds"):
        # Save thresholds to config
        thresholds = {
            'rmse': rmse_threshold,
            'mape': mape_threshold,
            'drift_p_value': drift_threshold,
            'resource_usage': resource_threshold
        }
        
        with open('configs/monitoring_thresholds.json', 'w') as f:
            json.dump(thresholds, f)
            
        st.success("Thresholds updated!")

if __name__ == "__main__":
    main() 