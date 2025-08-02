# Getting Started

This guide will help you set up and run the Store Sales Time Series Forecasting System. Follow these steps carefully to ensure a proper installation and configuration.

## Prerequisites

### System Requirements

1. **Hardware**:
   - CPU: 4+ cores recommended
   - RAM: 8GB minimum, 16GB recommended
   - Storage: 20GB minimum
   - GPU: Optional, recommended for LSTM training

2. **Operating System**:
   - Linux (Ubuntu 20.04+)
   - macOS (10.15+)
   - Windows 10/11 with WSL2

3. **Software**:
   ```bash
   # Check Python version (3.8+ required)
   $ python --version
   
   # Check pip version
   $ pip --version
   
   # Check git version
   $ git --version
   ```

### Required Packages

The following key packages will be installed automatically:

1. **Data Processing**:
   - pandas
   - numpy
   - scikit-learn

2. **Models**:
   - statsmodels
   - prophet
   - torch
   - pmdarima

3. **MLOps**:
   - mlflow
   - prefect
   - streamlit

4. **Monitoring**:
   - psutil
   - plotly
   - watchdog

## Installation

1. **Clone the Repository**:
   ```bash
   $ git clone https://github.com/your-org/store-sales-forecasting.git
   $ cd store-sales-forecasting
   ```

2. **Run Setup Script**:
   ```bash
   $ chmod +x setup.sh
   $ ./setup.sh
   ```
   
   The setup script will:
   - Check Python version
   - Create virtual environment
   - Install dependencies
   - Create necessary directories
   - Generate default configurations
   - Initialize git repository

3. **Verify Installation**:
   ```bash
   $ source venv/bin/activate
   $ python -c "import pandas, numpy, torch, prophet"
   ```

## Initial Setup

### 1. Environment Configuration

Create a `.env` file in the project root:

```bash
# MLflow settings
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=store_sales_forecasting

# Prefect settings
PREFECT_SERVER_API_HOST=0.0.0.0
PREFECT_SERVER_API_PORT=4200

# Model paths
MODEL_SAVE_PATH=./models
DATA_PATH=./data

# Monitoring settings
LOG_LEVEL=INFO
MONITORING_INTERVAL=60
```

### 2. Data Preparation

1. **Download Dataset**:
   ```bash
   $ mkdir -p data/raw
   $ cd data/raw
   $ kaggle competitions download -c store-sales-time-series-forecasting
   $ unzip store-sales-time-series-forecasting.zip
   ```

2. **Verify Data Files**:
   ```bash
   $ ls data/raw
   train.csv
   test.csv
   stores.csv
   oil.csv
   holidays_events.csv
   transactions.csv
   ```

### 3. Configuration Files

1. **Model Configuration** (`configs/model_configs/default.yaml`):
   ```yaml
   arima:
     max_p: 5
     max_d: 2
     max_q: 5
     seasonal: true
     
   prophet:
     seasonality_mode: multiplicative
     changepoint_prior_scale: 0.05
     
   lstm:
     hidden_size: 64
     num_layers: 2
     dropout: 0.2
   ```

2. **Training Configuration** (`configs/training_configs/default.yaml`):
   ```yaml
   data:
     sequence_length: 30
     prediction_horizon: 15
     test_size: 0.2
     validation_size: 0.2
     
   training:
     batch_size: 32
     epochs: 100
     learning_rate: 0.001
     early_stopping_patience: 10
   ```

## Quick Start Guide

### 1. Start Services

Run the dashboard and required services:

```bash
$ ./run_dashboard.sh
```

This will start:
- MLflow server (port 5000)
- Prefect server (port 4200)
- Streamlit dashboard (port 8501)

### 2. Access Dashboard

Open your browser and navigate to:
- Dashboard: http://localhost:8501
- MLflow UI: http://localhost:5000
- Prefect UI: http://localhost:4200

### 3. Train Models

1. Go to the "Training" page in the dashboard
2. Select models to train
3. Click "Start Training"
4. Monitor progress in MLflow UI

### 4. Generate Predictions

1. Go to the "Predictions" page
2. Set prediction parameters
3. Click "Generate Predictions"
4. View results in the dashboard

### 5. Monitor System

1. Go to the "Monitoring" page
2. View:
   - Data drift metrics
   - Model performance
   - System resources
   - Alerts

## Development Setup

For development work:

1. **Install Development Dependencies**:
   ```bash
   $ pip install -r requirements-dev.txt
   ```

2. **Set Up Pre-commit Hooks**:
   ```bash
   $ pre-commit install
   ```

3. **Run Tests**:
   ```bash
   $ pytest tests/
   ```

4. **Format Code**:
   ```bash
   $ black src/ tests/
   $ flake8 src/ tests/
   ```

## Troubleshooting

### Common Issues

1. **Port Conflicts**:
   ```bash
   $ sudo lsof -i :8501  # Check if port is in use
   $ sudo kill -9 PID    # Kill process if needed
   ```

2. **Memory Issues**:
   - Reduce batch size in training config
   - Clear MLflow artifacts
   - Remove old logs

3. **GPU Issues**:
   ```bash
   $ nvidia-smi  # Check GPU status
   $ export CUDA_VISIBLE_DEVICES=0  # Set specific GPU
   ```

### Getting Help

1. Check the [Troubleshooting](troubleshooting.md) guide
2. Review logs in `logs/` directory
3. Check MLflow and Prefect UIs for errors
4. Contact support team

## Next Steps

After setup:

1. Read [Data Pipeline](data_pipeline.md) documentation
2. Explore [Model Architecture](model_architecture.md)
3. Review [MLOps Integration](mlops_integration.md)
4. Check [Monitoring System](monitoring_system.md)

## Security Notes

1. **Data Security**:
   - Keep raw data in secure location
   - Use appropriate permissions
   - Don't commit sensitive data

2. **API Security**:
   - Use environment variables for secrets
   - Enable authentication in production
   - Regular security updates

3. **Access Control**:
   - Set up user authentication
   - Configure role-based access
   - Monitor access logs 