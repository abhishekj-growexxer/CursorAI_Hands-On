# Time Series Sales Forecasting Project

A production-ready time series forecasting system for daily product sales using multiple models (ARIMA, Prophet, and LSTM) with automated retraining capabilities.

## Features

- Multiple forecasting models:
  - ARIMA: Statistical time series forecasting
  - Prophet: Facebook's forecasting tool
  - LSTM: Deep learning approach using PyTorch
- Automated model retraining and selection
- MLflow integration for experiment tracking
- Prefect workflows for task orchestration
- Interactive dashboard for visualizations
- Comprehensive monitoring and logging

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd time-series-forecasting
   ```

2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Add your data files to `data/raw/`

5. Start the training pipeline:
   ```bash
   python src/training/train.py
   ```

## Project Structure

```
├── configs/            # Configuration files
│   ├── model_configs/  # Model-specific parameters
│   └── training_configs/ # Training parameters
├── data/              
│   ├── raw/           # Original data
│   └── processed/     # Cleaned data
├── logs/              # Application logs
│   ├── training/      # Training logs
│   ├── api/           # API logs
│   └── monitoring/    # Monitoring logs
├── models/            # Saved models
│   ├── arima/         # ARIMA models
│   ├── prophet/       # Prophet models
│   └── lstm/          # LSTM models
├── notebooks/         # Jupyter notebooks
├── src/
│   ├── data/         # Data processing
│   ├── models/       # Model implementations
│   ├── training/     # Training pipelines
│   ├── monitoring/   # Monitoring tools
│   └── utils/        # Helper functions
└── tests/            # Unit tests
```

## Configuration

1. Model parameters: `configs/model_configs/default.yaml`
2. Training settings: `configs/training_configs/default.yaml`
3. Environment variables: `.env`

## Development

1. Code Style:
   ```bash
   # Format code
   black src/
   
   # Run linter
   flake8 src/
   ```

2. Testing:
   ```bash
   # Run all tests
   pytest tests/
   
   # Run specific test file
   pytest tests/test_models.py
   ```

3. MLflow UI:
   ```bash
   mlflow ui
   ```

4. Prefect Dashboard:
   ```bash
   prefect orion start
   ```

## Monitoring

1. MLflow UI: Track experiments and model versions
   - Access at: http://localhost:5000

2. Prefect UI: Monitor workflow executions
   - Access at: http://localhost:4200

3. Application Logs:
   - Training logs: `logs/training/`
   - API logs: `logs/api/`
   - Monitoring logs: `logs/monitoring/`

## Model Details

### ARIMA
- Statistical time series model
- Handles trend and seasonality
- Configuration in `configs/model_configs/default.yaml`

### Prophet
- Facebook's forecasting tool
- Handles holidays and special events
- Configuration in `configs/model_configs/default.yaml`

### LSTM
- Deep learning approach
- Captures complex patterns
- Configuration in `configs/model_configs/default.yaml`

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here] 