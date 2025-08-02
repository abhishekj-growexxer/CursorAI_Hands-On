# Quick Start Guide

This guide will help you get the Store Sales Time Series Forecasting System up and running quickly.

## Prerequisites

Ensure you have:
- Python 3.8 or higher
- pip (latest version)
- git
- 8GB+ RAM
- 20GB+ free disk space

## Step 1: Clone and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-org/store-sales-forecasting.git
   cd store-sales-forecasting
   ```

2. **Run Setup Script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Activate Environment**:
   ```bash
   source venv/bin/activate
   ```

## Step 2: Add Data

1. **Download Dataset**:
   ```bash
   cd data/raw
   kaggle competitions download -c store-sales-time-series-forecasting
   unzip store-sales-time-series-forecasting.zip
   cd ../..
   ```

2. **Verify Data Files**:
   ```bash
   ls data/raw
   ```
   
   You should see:
   - train.csv
   - test.csv
   - stores.csv
   - oil.csv
   - holidays_events.csv
   - transactions.csv

## Step 3: Configure System

1. **Check Model Configuration** (`configs/model_configs/default.yaml`):
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

2. **Check Training Configuration** (`configs/training_configs/default.yaml`):
   ```yaml
   data:
     sequence_length: 30
     prediction_horizon: 15
     test_size: 0.2
   
   training:
     batch_size: 32
     epochs: 100
     learning_rate: 0.001
   ```

## Step 4: Verify Setup

Run the test suite to verify everything is working:
```bash
python -m pytest tests/
```

## Step 5: Start the System

1. **Start All Services**:
   ```bash
   ./run_dashboard.sh
   ```

2. **Access the System**:
   - Dashboard: http://localhost:8501
   - MLflow UI: http://localhost:5000
   - Prefect UI: http://localhost:4200

## Step 6: Train Models

1. Open the dashboard at http://localhost:8501
2. Go to the "Training" page
3. Select models to train (ARIMA, Prophet, LSTM)
4. Click "Start Training"
5. Monitor progress in MLflow UI

## Step 7: Generate Predictions

1. Go to the "Predictions" page
2. Set prediction parameters:
   - Number of days to predict
   - Confidence interval
3. Click "Generate Predictions"
4. View results in the dashboard

## Step 8: Monitor System

1. Go to the "Monitoring" page to view:
   - Model performance
   - Data drift
   - System resources
   - Alerts

## Step 9: Stop Services

When you're done:
```bash
./stop_services.sh
```

## Common Operations

### Process New Data
```python
from src.data.run_data_pipeline import process_data

process_data(
    input_path="data/raw/new_data.csv",
    output_path="data/processed/new_data_processed.csv"
)
```

### Train Specific Model
```python
from src.training.train_pipeline import train_model

train_model(
    model_name="prophet",
    data_path="data/processed/train_data.csv"
)
```

### Generate Predictions
```python
from src.models.predict import generate_predictions

predictions = generate_predictions(
    days=30,
    store_ids=[1, 2, 3]
)
```

## Troubleshooting

### Port Conflicts
If services fail to start due to port conflicts:
```bash
# Check ports
sudo lsof -i :8501  # Streamlit
sudo lsof -i :5000  # MLflow
sudo lsof -i :4200  # Prefect

# Kill process if needed
sudo kill -9 PID
```

### Memory Issues
If you encounter memory errors:
1. Reduce batch size in training config
2. Clear MLflow artifacts:
   ```bash
   rm -rf mlruns/*
   ```
3. Remove old logs:
   ```bash
   rm -rf logs/*
   ```

### Data Issues
If data validation fails:
1. Check data format:
   ```bash
   python src/data/validate_data.py data/raw/your_file.csv
   ```
2. Review error logs:
   ```bash
   cat logs/data_validation.log
   ```

## Next Steps

After getting started:

1. **Customize Models**:
   - Adjust model parameters in `configs/model_configs/default.yaml`
   - Add new features in `src/data/feature_engineering.py`
   - Modify training parameters in `configs/training_configs/default.yaml`

2. **Set Up Monitoring**:
   - Configure alerts in `configs/monitoring_thresholds.json`
   - Set up email notifications
   - Add custom metrics

3. **Automate Retraining**:
   - Configure retraining schedule
   - Set up data quality checks
   - Define retraining triggers

4. **Explore Advanced Features**:
   - Cross-validation
   - Hyperparameter optimization
   - Ensemble methods
   - Custom loss functions

## Getting Help

If you need help:

1. Check the full [documentation](technical_documentation.md)
2. Review [troubleshooting guide](troubleshooting.md)
3. Check [FAQs](faqs.md)
4. Contact support team

## Quick Reference

### Key Files
- `setup.sh`: Initial setup
- `run_dashboard.sh`: Start services
- `stop_services.sh`: Stop services
- `configs/model_configs/default.yaml`: Model configuration
- `configs/training_configs/default.yaml`: Training configuration
- `.env`: Environment variables

### Key URLs
- Dashboard: http://localhost:8501
- MLflow UI: http://localhost:5000
- Prefect UI: http://localhost:4200

### Common Commands
```bash
# Setup
./setup.sh

# Start system
./run_dashboard.sh

# Run tests
python -m pytest tests/

# Stop system
./stop_services.sh
```

### Directory Structure
```
project_root/
├── configs/          # Configuration files
├── data/             # Data files
├── logs/             # Log files
├── models/           # Saved models
├── src/              # Source code
└── tests/            # Test files
``` 