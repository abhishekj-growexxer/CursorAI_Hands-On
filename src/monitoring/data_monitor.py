import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from statistical_analysis import StatisticalAnalyzer

class DataMonitor:
    def __init__(self):
        self.data = {}
        self.metrics = {}
        self.stats_analyzer = None
        
    def load_data(self, train_path, stores_path, oil_path, holidays_path, transactions_path):
        """Load all datasets"""
        self.data['train'] = pd.read_csv(train_path, parse_dates=['date'])
        self.data['stores'] = pd.read_csv(stores_path)
        self.data['oil'] = pd.read_csv(oil_path, parse_dates=['date'])
        self.data['holidays'] = pd.read_csv(holidays_path, parse_dates=['date'])
        self.data['transactions'] = pd.read_csv(transactions_path, parse_dates=['date'])
        self._calculate_metrics()
        self.stats_analyzer = StatisticalAnalyzer(self.data['train'])

    def _calculate_metrics(self):
        """Calculate key business metrics"""
        # Sales Metrics
        self.metrics['total_sales'] = self.data['train']['sales'].sum()
        self.metrics['avg_daily_sales'] = self.data['train'].groupby('date')['sales'].sum().mean()
        self.metrics['sales_trend'] = self.data['train'].groupby('date')['sales'].sum().rolling(7).mean()
        
        # Store Metrics
        self.metrics['total_stores'] = len(self.data['stores'])
        self.metrics['stores_by_type'] = self.data['stores']['type'].value_counts()
        
        # Product Metrics
        self.metrics['total_families'] = self.data['train']['family'].nunique()
        self.metrics['top_families'] = self.data['train'].groupby('family')['sales'].sum().sort_values(ascending=False)
        
        # Promotion Metrics
        self.metrics['promo_items'] = self.data['train']['onpromotion'].sum()
        self.metrics['promo_effect'] = self.data['train'].groupby('onpromotion')['sales'].mean()

def render_dashboard():
    st.set_page_config(page_title="Sales Data Monitor", layout="wide")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
        ["Business Metrics", "Statistical Analysis", "Time Series Analysis", "Feature Analysis"])
    
    # Initialize monitor
    monitor = DataMonitor()
    
    # Data Loading Section
    with st.expander("Data Loading and Status"):
        col1, col2, col3 = st.columns(3)
        with col1:
            train_path = st.text_input("Training Data Path", "dataset/store-sales-time-series-forecasting/train.csv")
        with col2:
            stores_path = st.text_input("Stores Data Path", "dataset/store-sales-time-series-forecasting/stores.csv")
        with col3:
            oil_path = st.text_input("Oil Data Path", "dataset/store-sales-time-series-forecasting/oil.csv")
        
        col4, col5 = st.columns(2)
        with col4:
            holidays_path = st.text_input("Holidays Data Path", "dataset/store-sales-time-series-forecasting/holidays_events.csv")
        with col5:
            transactions_path = st.text_input("Transactions Data Path", "dataset/store-sales-time-series-forecasting/transactions.csv")
        
        if st.button("Load Data"):
            with st.spinner("Loading data..."):
                monitor.load_data(train_path, stores_path, oil_path, holidays_path, transactions_path)
                st.success("Data loaded successfully!")

    if page == "Business Metrics":
        render_business_metrics(monitor)
    elif page == "Statistical Analysis":
        render_statistical_analysis(monitor)
    elif page == "Time Series Analysis":
        render_time_series_analysis(monitor)
    elif page == "Feature Analysis":
        render_feature_analysis(monitor)

def render_business_metrics(monitor):
    st.header("Key Business Metrics")
    if hasattr(monitor, 'metrics') and monitor.metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sales", f"${monitor.metrics['total_sales']:,.2f}")
        with col2:
            st.metric("Average Daily Sales", f"${monitor.metrics['avg_daily_sales']:,.2f}")
        with col3:
            st.metric("Total Stores", monitor.metrics['total_stores'])
        with col4:
            st.metric("Product Families", monitor.metrics['total_families'])

        # Sales Trends
        st.subheader("Sales Trends")
        sales_trend = monitor.data['train'].groupby('date')['sales'].sum().reset_index()
        fig = px.line(sales_trend, x='date', y='sales', title='Daily Sales Trend')
        st.plotly_chart(fig, use_container_width=True)

        # Store Analysis
        st.subheader("Store Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(monitor.metrics['stores_by_type'], title='Stores by Type')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            store_sales = monitor.data['train'].groupby('store_nbr')['sales'].sum().sort_values(ascending=False)
            fig = px.bar(store_sales.head(10), title='Top 10 Stores by Sales')
            st.plotly_chart(fig, use_container_width=True)

        # Product Analysis
        st.subheader("Product Family Analysis")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(monitor.metrics['top_families'].head(10), title='Top 10 Product Families')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            promo_impact = monitor.metrics['promo_effect']
            fig = px.bar(promo_impact, title='Sales by Promotion Status')
            st.plotly_chart(fig, use_container_width=True)

        # Data Quality Metrics
        st.header("Data Quality Metrics")
        col1, col2 = st.columns(2)
        with col1:
            missing_data = monitor.data['train'].isnull().sum()
            fig = px.bar(missing_data[missing_data > 0], title='Missing Values by Column')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Calculate daily data completeness
            expected_daily_records = monitor.metrics['total_stores'] * monitor.metrics['total_families']
            daily_completeness = monitor.data['train'].groupby('date').size() / expected_daily_records * 100
            fig = px.line(daily_completeness, title='Daily Data Completeness (%)')
            st.plotly_chart(fig, use_container_width=True)

def render_statistical_analysis(monitor):
    if not monitor.stats_analyzer:
        st.warning("Please load data first!")
        return
        
    st.header("Statistical Analysis")
    
    # Select column for analysis
    numeric_cols = monitor.data['train'].select_dtypes(include=[np.number]).columns
    selected_col = st.selectbox("Select Column for Analysis", numeric_cols)
    
    # Distribution Analysis
    st.subheader("Distribution Analysis")
    basic_stats, shapiro_p, dist_fig = monitor.stats_analyzer.analyze_distributions(selected_col)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Basic Statistics:")
        st.write(pd.Series(basic_stats))
    with col2:
        st.write("Normality Test:")
        st.write(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
        if shapiro_p < 0.05:
            st.write("Distribution is likely not normal")
        else:
            st.write("Distribution appears to be normal")
    
    st.plotly_chart(dist_fig, use_container_width=True)
    
    # Outlier Analysis
    st.subheader("Outlier Analysis")
    method = st.radio("Outlier Detection Method", ["zscore", "iqr"])
    outlier_stats, outlier_fig = monitor.stats_analyzer.outlier_analysis(selected_col, method=method)
    
    st.write("Outlier Statistics:")
    st.write(pd.Series(outlier_stats))
    st.plotly_chart(outlier_fig, use_container_width=True)

def render_time_series_analysis(monitor):
    if not monitor.stats_analyzer:
        st.warning("Please load data first!")
        return
        
    st.header("Time Series Analysis")
    
    # Time Series Decomposition
    st.subheader("Time Series Decomposition")
    period = st.slider("Select Seasonality Period (days)", 1, 30, 7)
    decomp, decomp_fig = monitor.stats_analyzer.decompose_time_series('date', 'sales', period=period)
    st.plotly_chart(decomp_fig, use_container_width=True)
    
    # Seasonality Analysis
    st.subheader("Seasonality Analysis")
    patterns, season_fig = monitor.stats_analyzer.analyze_seasonality('date', 'sales')
    st.plotly_chart(season_fig, use_container_width=True)
    
    # Additional Time Series Metrics
    st.subheader("Time Series Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Seasonal Strength:")
        seasonal_strength = np.var(decomp.seasonal) / np.var(decomp.seasonal + decomp.resid)
        st.metric("Seasonal Strength", f"{seasonal_strength:.2%}")
    with col2:
        st.write("Trend Strength:")
        trend_strength = 1 - np.var(decomp.resid) / np.var(decomp.trend + decomp.resid)
        st.metric("Trend Strength", f"{trend_strength:.2%}")

def render_feature_analysis(monitor):
    if not monitor.stats_analyzer:
        st.warning("Please load data first!")
        return
        
    st.header("Feature Analysis")
    
    # Correlation Analysis
    st.subheader("Correlation Analysis")
    corr_matrix, corr_fig = monitor.stats_analyzer.correlation_analysis()
    st.plotly_chart(corr_fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("Feature Importance")
    target_col = st.selectbox("Select Target Variable", monitor.data['train'].select_dtypes(include=[np.number]).columns)
    categorical_cols = monitor.data['train'].select_dtypes(include=['object']).columns.tolist()
    importance, imp_fig = monitor.stats_analyzer.feature_importance(target_col, categorical_cols)
    st.plotly_chart(imp_fig, use_container_width=True)

if __name__ == "__main__":
    render_dashboard() 