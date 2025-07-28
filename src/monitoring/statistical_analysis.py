import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acf
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class StatisticalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.stats_results = {}
        
    def analyze_distributions(self, column):
        """Analyze distribution of a numeric column"""
        data = self.data[column].dropna()
        
        # Basic statistics
        basic_stats = {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'skew': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25)
        }
        
        # Normality tests
        _, shapiro_p = stats.shapiro(data.sample(min(len(data), 5000)))
        
        # Create distribution plot
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Distribution', 'Q-Q Plot'))
        
        # Histogram with KDE
        hist_data = px.histogram(data, nbins=50, marginal='box')
        fig.add_trace(hist_data.data[0], row=1, col=1)
        
        # Q-Q plot
        qq = stats.probplot(data)
        fig.add_trace(
            go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Q-Q Plot'),
            row=2, col=1
        )
        
        return basic_stats, shapiro_p, fig

    def decompose_time_series(self, date_col, value_col, period=7):
        """Perform time series decomposition"""
        # Prepare time series data
        ts_data = self.data.set_index(date_col)[value_col]
        ts_data = ts_data.resample('D').mean().fillna(method='ffill')
        
        # Perform decomposition
        decomposition = seasonal_decompose(ts_data, period=period)
        
        # Create plot
        fig = make_subplots(rows=4, cols=1, subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
        
        components = [ts_data, decomposition.trend, decomposition.seasonal, decomposition.resid]
        for i, comp in enumerate(components, 1):
            fig.add_trace(go.Scatter(x=comp.index, y=comp.values), row=i, col=1)
            
        fig.update_layout(height=800, showlegend=False)
        
        return decomposition, fig

    def correlation_analysis(self, numeric_cols=None):
        """Perform correlation analysis"""
        if numeric_cols is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
        corr_matrix = self.data[numeric_cols].corr()
        
        # Create heatmap
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       color_continuous_scale='RdBu_r')
        
        return corr_matrix, fig

    def analyze_seasonality(self, date_col, value_col):
        """Analyze seasonal patterns"""
        df = self.data.copy()
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['hour'] = df[date_col].dt.hour if df[date_col].dt.hour.nunique() > 1 else None
        
        seasonal_patterns = {
            'monthly': df.groupby('month')[value_col].mean(),
            'daily': df.groupby('day_of_week')[value_col].mean(),
            'yearly_trend': df.groupby('year')[value_col].mean()
        }
        
        if df['hour'] is not None:
            seasonal_patterns['hourly'] = df.groupby('hour')[value_col].mean()
        
        # Create seasonal plots
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Monthly Pattern', 'Daily Pattern', 
                                                           'Yearly Trend', 'ACF Plot'))
        
        # Monthly pattern
        fig.add_trace(go.Scatter(x=seasonal_patterns['monthly'].index, 
                               y=seasonal_patterns['monthly'].values,
                               mode='lines+markers'), row=1, col=1)
        
        # Daily pattern
        fig.add_trace(go.Scatter(x=seasonal_patterns['daily'].index,
                               y=seasonal_patterns['daily'].values,
                               mode='lines+markers'), row=1, col=2)
        
        # Yearly trend
        fig.add_trace(go.Scatter(x=seasonal_patterns['yearly_trend'].index,
                               y=seasonal_patterns['yearly_trend'].values,
                               mode='lines+markers'), row=2, col=1)
        
        # ACF plot
        acf_values = acf(df[value_col].dropna(), nlags=40)
        fig.add_trace(go.Bar(x=list(range(len(acf_values))), y=acf_values), row=2, col=2)
        
        return seasonal_patterns, fig

    def outlier_analysis(self, column, method='zscore', threshold=3):
        """Perform outlier analysis"""
        data = self.data[column].dropna()
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > threshold]
        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))]
        
        # Create box plot with outliers
        fig = go.Figure()
        fig.add_trace(go.Box(y=data, name=column, boxpoints='outliers'))
        
        outlier_stats = {
            'total_outliers': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'outlier_mean': outliers.mean(),
            'outlier_std': outliers.std()
        }
        
        return outlier_stats, fig

    def feature_importance(self, target_col, categorical_cols=None):
        """Calculate feature importance using various methods"""
        df = self.data.copy()
        
        # Handle categorical variables
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols)
        
        # Prepare numeric features
        feature_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = feature_cols[feature_cols != target_col]
        
        # Calculate correlations with target
        correlations = df[feature_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        
        # Create importance plot
        fig = px.bar(x=correlations.index, y=correlations.values,
                    title=f'Feature Importance (Correlation with {target_col})')
        
        return correlations, fig 