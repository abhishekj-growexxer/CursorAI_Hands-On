"""
Centralized logging configuration for the entire application.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create subdirectories for different components
for subdir in ["data", "models", "training", "monitoring", "api"]:
    (log_dir / subdir).mkdir(exist_ok=True)

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Set up a logger with standardized configuration.
    
    Args:
        name: Logger name (usually __name__ of the module)
        log_file: Optional specific log file path
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # File handler
    if log_file is None:
        # Default log file based on module name
        module_name = name.split('.')[-1]
        component = name.split('.')[1] if len(name.split('.')) > 1 else "general"
        log_file = log_dir / component / f"{module_name}.log"
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Example usage:
# logger = setup_logger(__name__) 