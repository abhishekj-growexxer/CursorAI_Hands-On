#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting Sales Forecasting Dashboard..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start MLflow server in background
echo "📊 Starting MLflow server..."
mlflow server \
    --backend-store-uri ./mlruns \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 &

# Wait for MLflow server to start
sleep 5

# Start Prefect server in background
echo "🔄 Starting Prefect server..."
prefect server start &

# Wait for Prefect server to start
sleep 5

# Start Streamlit dashboard
echo "📈 Starting Streamlit dashboard..."
streamlit run src/dashboard/app.py

# Note: The dashboard will run in the foreground.
# When you stop the dashboard (Ctrl+C), the background servers will continue running.
# To stop them, you'll need to find and kill their processes:
# ps aux | grep mlflow
# ps aux | grep prefect
# kill [PID] 