#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Install requirements if needed
pip install -r requirements.txt

# Run Streamlit dashboard
streamlit run src/monitoring/data_monitor.py 