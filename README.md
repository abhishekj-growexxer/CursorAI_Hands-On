# Time Series Sales Forecasting Project

This project implements a production-ready time series forecasting system for daily product sales using multiple models (ARIMA, Prophet, and LSTM) with automated retraining capabilities.

## Project Structure

```
├── configs/            # Configuration files for models and training
├── data/              
│   ├── raw/           # Original, immutable data
│   └── processed/     # Cleaned and preprocessed data
├── logs/              # Application and training logs
├── models/            # Saved model artifacts
├── notebooks/         # Jupyter notebooks for exploration
├── src/
│   ├── data/         # Data loading and preprocessing
│   ├── models/       # Model implementations
│   ├── training/     # Training pipelines
│   └── utils/        # Helper functions
├── requirements.txt   # Project dependencies
└── setup.sh          # Environment setup script
```

## Setup

1. Clone the repository
2. Make the setup script executable:
   ```bash
   chmod +x setup.sh
   ```
3. Run the setup script:
   ```bash
   ./setup.sh
   ```
4. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

## Components

1. **Data Pipeline**
   - Load and preprocess retail sales data
   - Feature engineering for time series
   - Train/test splitting with time-based validation

2. **Models**
   - ARIMA: Statistical time series model
   - Prophet: Facebook's forecasting tool
   - LSTM: Deep learning approach using PyTorch

3. **MLflow Integration**
   - Experiment tracking
   - Model versioning
   - Performance metrics logging

4. **Prefect Workflows**
   - Automated model retraining
   - Data pipeline orchestration
   - Monitoring and alerting

## Usage

[To be added as components are implemented]

## Development

- Code formatting: `black src/`
- Linting: `flake8 src/`
- Run tests: `pytest tests/`

## Monitoring

- MLflow UI: Track experiments and model performance
- Prefect UI: Monitor workflow executions
- Application logs: Check `logs/` directory 