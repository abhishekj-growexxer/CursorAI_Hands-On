#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up Time Series Forecasting project environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed}
mkdir -p models/{arima,prophet,lstm}
mkdir -p notebooks
mkdir -p src/{data,models,training,utils,monitoring}
mkdir -p logs/{training,api,monitoring}
mkdir -p configs/{model_configs,training_configs}

# Create __init__.py files
find src -type d -exec touch {}/__init__.py \;

# Create default configuration files
echo "📝 Creating default configuration files..."
cat > configs/model_configs/default.yaml << EOL
model_params:
  arima:
    p: 1
    d: 1
    q: 1
  prophet:
    seasonality_mode: 'multiplicative'
    yearly_seasonality: 'auto'
    weekly_seasonality: 'auto'
    daily_seasonality: 'auto'
  lstm:
    hidden_size: 64
    num_layers: 2
    dropout: 0.2
    learning_rate: 0.001
EOL

cat > configs/training_configs/default.yaml << EOL
training:
  batch_size: 32
  epochs: 100
  validation_split: 0.2
  early_stopping_patience: 10
  
data:
  sequence_length: 30
  prediction_horizon: 7
  test_size: 0.2
EOL

# Create .env file with default settings
echo "🔒 Creating .env file with default settings..."
cat > .env << EOL
# MLflow settings
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=store_sales_forecasting

# Prefect settings
PREFECT_LOGGING_LEVEL=INFO
PREFECT_HOME=~/.prefect

# Model settings
MODEL_REGISTRY_PATH=./models
DATA_PATH=./data
EOL

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "🔄 Initializing git repository..."
    git init
    # Create .gitignore
    cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.env

# Data
data/processed/*
data/raw/*
!data/processed/.gitkeep
!data/raw/.gitkeep

# Models
models/*
!models/.gitkeep

# Logs
logs/*
!logs/.gitkeep

# MLflow
mlruns/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints
EOL
fi

echo "✨ Setup complete! Activate your environment with: source venv/bin/activate"
echo "📝 Next steps:"
echo "1. Add your data files to data/raw/"
echo "2. Configure your model parameters in configs/model_configs/"
echo "3. Run 'python -m pytest tests/' to verify the setup" 
