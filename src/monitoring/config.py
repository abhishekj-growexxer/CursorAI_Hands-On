"""Configuration settings for the data monitoring dashboard"""

# Data quality thresholds
QUALITY_THRESHOLDS = {
    'completeness_threshold': 95.0,  # Minimum acceptable data completeness (%)
    'missing_values_threshold': 5.0,  # Maximum acceptable missing values (%)
    'outlier_threshold': 3.0,  # Standard deviations for outlier detection
    'duplicates_threshold': 0.1,  # Maximum acceptable duplicate records (%)
}

# Business metrics thresholds
BUSINESS_THRESHOLDS = {
    'min_daily_sales': 1000,  # Minimum expected daily sales
    'max_daily_sales': 1000000,  # Maximum expected daily sales
    'min_transactions': 10,  # Minimum daily transactions per store
    'promotion_effect_min': 1.1,  # Minimum expected lift from promotions
}

# Time-based checks
TIME_CHECKS = {
    'max_days_gap': 1,  # Maximum acceptable gap in daily data
    'lookback_days': 30,  # Days to look back for trend analysis
    'seasonal_cycle': 7,  # Days in seasonal cycle (weekly)
}

# Alert settings
ALERT_SETTINGS = {
    'email_alerts': True,
    'slack_alerts': False,
    'alert_frequency': 'daily',
    'alert_levels': ['warning', 'critical'],
}

# Visualization settings
VIZ_SETTINGS = {
    'default_height': 400,
    'default_width': 600,
    'color_scheme': 'viridis',
    'trend_window': 7,  # Rolling average window
}

# Data refresh settings
REFRESH_SETTINGS = {
    'auto_refresh': True,
    'refresh_interval': 3600,  # Seconds
    'cache_timeout': 1800,  # Seconds
}

# Storage settings
STORAGE_SETTINGS = {
    'metrics_history': True,
    'history_retention': 90,  # Days
    'compression': True,
} 