#!/bin/bash

# Exit on error
set -e

echo "🛑 Stopping all services..."

# Stop MLflow server
echo "📊 Stopping MLflow server..."
pkill -f "mlflow server" || true

# Stop Prefect server
echo "🔄 Stopping Prefect server..."
pkill -f "prefect server" || true

# Stop Streamlit dashboard
echo "📈 Stopping Streamlit dashboard..."
pkill -f "streamlit run" || true

echo "✅ All services stopped successfully!" 