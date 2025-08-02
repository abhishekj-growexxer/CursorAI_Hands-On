# Time Series Sales Forecasting Project

A production-ready time series forecasting system for daily product sales using multiple models (ARIMA, Prophet, and LSTM) with automated retraining capabilities.

## Project Documentation

This project includes various documentation to help you understand the system, data pipeline, and model architecture. Below is a list of available documents:

### System Overview & Architecture
- [System Overview](docs/system_overview.md)
- [System Design](docs/system_design.md)
- [Model Architecture](docs/model_architecture.md)
- [Project Flow](docs/project_flow.md)
- [Architecture Diagrams](docs/architecture_diagrams.md)

### Getting Started
- [Getting Started](docs/getting_started.md)
- [Quick Start](docs/quick_start.md)

### Technical & Data Documentation
- [Technical Documentation](docs/technical_documentation.md)
- [Data Pipeline](docs/data_pipeline.md)
- [Data Analysis Report](docs/data_analysis_report.md)

### Miscellaneous
- [Diagram Index](docs/diagram_index.md)


---

## Dataset Information: Store Sales - Time Series Forecasting

This project utilizes the **Store Sales - Time Series Forecasting** dataset, which is available through a Kaggle competition.

- **Competition Link**: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Competition Status**: **Ongoing** (as of now, but check Kaggle for updates)
- **Objective**: The goal is to forecast the sales for a store based on historical data and various features such as promotions, holiday seasons, and store details.
  
  
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
├── configs
│ ├── model_configs
│ └── training_configs
├── data
│ ├── processed
│ └── raw
├── dataset
│ ├── dataset_info.txt
│ └── store-sales-time-series-forecasting
├── docs
│ ├── architecture_diagrams.md
│ ├── data_analysis_report.md
│ ├── data_pipeline.md
│ ├── getting_started.md
│ ├── model_architecture.md
│ ├── project_flow.md
│ ├── quick_start.md
│ ├── system_design.md
│ ├── system_overview.md
│ └── technical_documentation.md
├── generate_diagrams.sh
├── logs
│ ├── api
│ ├── data
│ ├── models
│ ├── monitoring
│ └── training
├── mlruns
│ └── models
├── models
│ ├── arima
│ ├── lstm
│ └── prophet
├── notebooks
│ └── 01_data_exploration.ipynb
├── README.md
├── requirements.txt
├── run_dashboard.sh
├── setup.sh
├── src
│ ├── analysis
│ ├── dashboard
│ ├── data
│ ├── models
│ ├── monitoring
│ ├── training
│ └── utils
├── stop_services.sh
└── tests
├── test_arima_model.py
├── test_data_processor.py
├── test_feature_engineering.py
├── test_lstm_model.py
├── test_mlflow_tracking.py
├── test_monitoring.py
├── test_prefect_workflows.py
├── test_prophet_model.py
└── test_training_pipeline.py
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
