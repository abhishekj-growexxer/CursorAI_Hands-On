import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class DataAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = {}
        self.analysis = {}
        
    def load_data(self):
        """Load all datasets"""
        # Load training data
        self.data['train'] = pd.read_csv(
            os.path.join(self.data_dir, 'train.csv'),
            parse_dates=['date']
        )
        
        # Load store data
        self.data['stores'] = pd.read_csv(
            os.path.join(self.data_dir, 'stores.csv')
        )
        
        # Load oil price data
        self.data['oil'] = pd.read_csv(
            os.path.join(self.data_dir, 'oil.csv'),
            parse_dates=['date']
        )
        
        # Load holidays data
        self.data['holidays'] = pd.read_csv(
            os.path.join(self.data_dir, 'holidays_events.csv'),
            parse_dates=['date']
        )
        
        # Load transactions data
        self.data['transactions'] = pd.read_csv(
            os.path.join(self.data_dir, 'transactions.csv'),
            parse_dates=['date']
        )

    def analyze_training_data(self):
        """Analyze training dataset"""
        train = self.data['train']
        
        self.analysis['train'] = {
            'structure': {
                'records': len(train),
                'unique_stores': train['store_nbr'].nunique(),
                'product_families': train['family'].nunique()
            },
            'statistics': {
                'total_sales': train['sales'].sum(),
                'avg_daily_sales': train.groupby('date')['sales'].sum().mean(),
                'sales_range': (train['sales'].min(), train['sales'].max()),
                'promotion_freq': (train['onpromotion'] > 0).mean() * 100
            },
            'quality': {
                'missing_values': train.isnull().sum().to_dict(),
                'duplicates': train.duplicated().sum(),
                'zero_sales': (train['sales'] == 0).sum(),
                'negative_sales': (train['sales'] < 0).sum()
            }
        }

    def analyze_store_data(self):
        """Analyze store dataset"""
        stores = self.data['stores']
        
        self.analysis['stores'] = {
            'structure': {
                'total_stores': len(stores),
                'cities': stores['city'].nunique(),
                'states': stores['state'].nunique(),
                'types': stores['type'].nunique()
            },
            'distribution': {
                'store_types': stores['type'].value_counts().to_dict(),
                'clusters': stores['cluster'].value_counts().to_dict()
            },
            'geographic': {
                'cities_per_state': stores.groupby('state')['city'].nunique().describe().to_dict(),
                'stores_per_city': stores.groupby('city')['store_nbr'].count().describe().to_dict()
            }
        }

    def analyze_oil_data(self):
        """Analyze oil price dataset"""
        oil = self.data['oil']
        
        self.analysis['oil'] = {
            'structure': {
                'records': len(oil),
                'date_range': (oil['date'].min(), oil['date'].max())
            },
            'statistics': {
                'price_range': (oil['dcoilwtico'].min(), oil['dcoilwtico'].max()),
                'avg_price': oil['dcoilwtico'].mean(),
                'volatility': oil['dcoilwtico'].std()
            },
            'quality': {
                'missing_values': oil['dcoilwtico'].isnull().sum(),
                'gaps': self._analyze_time_gaps(oil['date'])
            }
        }

    def analyze_holidays(self):
        """Analyze holidays dataset"""
        holidays = self.data['holidays']
        
        self.analysis['holidays'] = {
            'structure': {
                'records': len(holidays),
                'date_range': (holidays['date'].min(), holidays['date'].max())
            },
            'distribution': {
                'types': holidays['type'].value_counts().to_dict(),
                'locale': holidays['locale'].value_counts().to_dict(),
                'transferred': holidays['transferred'].sum()
            },
            'temporal': {
                'monthly_density': holidays.groupby(holidays['date'].dt.month).size().to_dict(),
                'top_types': holidays['type'].value_counts().head().to_dict()
            }
        }

    def analyze_transactions(self):
        """Analyze transactions dataset"""
        trans = self.data['transactions']
        
        self.analysis['transactions'] = {
            'structure': {
                'records': len(trans),
                'date_range': (trans['date'].min(), trans['date'].max())
            },
            'statistics': {
                'total_transactions': trans['transactions'].sum(),
                'avg_daily': trans.groupby('date')['transactions'].sum().mean(),
                'per_store': trans.groupby('store_nbr')['transactions'].describe().to_dict()
            },
            'patterns': {
                'weekly': trans.groupby(trans['date'].dt.dayofweek)['transactions'].mean().to_dict(),
                'monthly': trans.groupby(trans['date'].dt.month)['transactions'].mean().to_dict()
            }
        }

    def cross_dataset_analysis(self):
        """Perform analysis across datasets"""
        # Merge relevant datasets
        sales_oil = pd.merge(
            self.data['train'].groupby('date')['sales'].sum().reset_index(),
            self.data['oil'],
            on='date',
            how='left'
        )
        
        # Sales vs Oil analysis
        self.analysis['cross'] = {
            'oil_impact': {
                'correlation': sales_oil['sales'].corr(sales_oil['dcoilwtico']),
                'lag_correlations': self._analyze_lags(sales_oil, 'dcoilwtico', 'sales', max_lags=7)
            }
        }
        
        # Holiday impact
        holiday_dates = self.data['holidays']['date'].unique()
        sales_holidays = self.data['train'].copy()
        sales_holidays['is_holiday'] = sales_holidays['date'].isin(holiday_dates)
        
        self.analysis['cross']['holiday_impact'] = {
            'avg_lift': (
                sales_holidays[sales_holidays['is_holiday']]['sales'].mean() /
                sales_holidays[~sales_holidays['is_holiday']]['sales'].mean() - 1
            ) * 100
        }
        
        # Promotional impact
        self.analysis['cross']['promo_impact'] = {
            'avg_lift': (
                self.data['train'][self.data['train']['onpromotion'] > 0]['sales'].mean() /
                self.data['train'][self.data['train']['onpromotion'] == 0]['sales'].mean() - 1
            ) * 100
        }

    def _analyze_time_gaps(self, date_series):
        """Analyze gaps in time series data"""
        date_series = pd.Series(date_series).sort_values()
        gaps = (date_series - date_series.shift()).dropna()
        return (gaps != pd.Timedelta('1 day')).sum()

    def _analyze_lags(self, df, col1, col2, max_lags):
        """Analyze lagged correlations"""
        correlations = {}
        for lag in range(max_lags + 1):
            correlations[lag] = df[col1].shift(lag).corr(df[col2])
        return correlations

    def generate_report(self):
        """Generate the analysis report"""
        # Load data
        self.load_data()
        
        # Perform analyses
        self.analyze_training_data()
        self.analyze_store_data()
        self.analyze_oil_data()
        self.analyze_holidays()
        self.analyze_transactions()
        self.cross_dataset_analysis()
        
        # Generate insights
        insights = self._generate_insights()
        
        # Update report template
        report_template = open('docs/data_analysis_report.md', 'r').read()
        report = self._fill_template(report_template, self.analysis, insights)
        
        # Save updated report
        with open('docs/data_analysis_report_filled.md', 'w') as f:
            f.write(report)

    def _generate_insights(self):
        """Generate key insights from the analysis"""
        insights = {
            'business': [
                f"Average sales lift during promotions: {self.analysis['cross']['promo_impact']['avg_lift']:.1f}%",
                f"Holiday impact on sales: {self.analysis['cross']['holiday_impact']['avg_lift']:.1f}% increase",
                f"Oil price correlation with sales: {self.analysis['cross']['oil_impact']['correlation']:.3f}"
            ],
            'quality': [
                f"Missing oil prices: {self.analysis['oil']['quality']['missing_values']} days",
                f"Time series gaps: {self.analysis['oil']['quality']['gaps']} discontinuities",
                f"Zero sales entries: {self.analysis['train']['quality']['zero_sales']} records"
            ],
            'modeling': [
                "Strong weekly seasonality in transactions",
                "Significant holiday effects require special handling",
                "Oil price effects show lagged correlation"
            ]
        }
        return insights

    def _fill_template(self, template, analysis, insights):
        """Fill the report template with analysis results"""
        # Time coverage
        template = template.replace(
            "[Will be filled after analysis]",
            f"{self.data['train']['date'].min().strftime('%Y-%m-%d')} to {self.data['train']['date'].max().strftime('%Y-%m-%d')}"
        )
        
        # Training data
        template = template.replace("[Count]", f"{analysis['train']['structure']['records']:,}")
        template = template.replace("[Amount]", f"{analysis['train']['statistics']['total_sales']:,.2f}")
        template = template.replace("$[Amount]", f"${analysis['train']['statistics']['avg_daily_sales']:,.2f}")
        template = template.replace("$[Min]", f"${analysis['train']['statistics']['sales_range'][0]:,.2f}")
        template = template.replace("$[Max]", f"${analysis['train']['statistics']['sales_range'][1]:,.2f}")
        template = template.replace("[%]", f"{analysis['train']['statistics']['promotion_freq']:.1f}%")
        
        # Store data
        template = template.replace("[Breakdown]", str(analysis['stores']['distribution']['store_types']))
        template = template.replace("[Count]", str(analysis['stores']['structure']['total_stores']))
        
        # Oil data
        template = template.replace("$[Min]", f"${analysis['oil']['statistics']['price_range'][0]:.2f}")
        template = template.replace("$[Max]", f"${analysis['oil']['statistics']['price_range'][1]:.2f}")
        template = template.replace("$[Avg]", f"${analysis['oil']['statistics']['avg_price']:.2f}")
        template = template.replace("[Std]", f"{analysis['oil']['statistics']['volatility']:.2f}")
        
        # Holiday data
        template = template.replace("[Count/Days]", str(analysis['oil']['quality']['missing_values']))
        template = template.replace("[Gaps analysis]", str(analysis['oil']['quality']['gaps']))
        
        # Transaction data
        template = template.replace("[Count]", str(analysis['transactions']['structure']['records']))
        template = template.replace("[Min/Max/Avg]", 
            f"Min: {analysis['transactions']['statistics']['per_store']['min']:.0f}, "
            f"Max: {analysis['transactions']['statistics']['per_store']['max']:.0f}, "
            f"Avg: {analysis['transactions']['statistics']['per_store']['mean']:.0f}"
        )
        
        # Cross-dataset analysis
        template = template.replace("[Value]", f"{analysis['cross']['oil_impact']['correlation']:.3f}")
        template = template.replace("[%]", f"{analysis['cross']['holiday_impact']['avg_lift']:.1f}%")
        template = template.replace("[Analysis]", str(analysis['cross']['oil_impact']['lag_correlations']))
        
        # Insights
        for i, insight in enumerate(insights['business'], 1):
            template = template.replace(f"[Insight {i}]", insight)
        
        for i, issue in enumerate(insights['quality'], 1):
            template = template.replace(f"[Issue {i}]", issue)
        
        for i, consideration in enumerate(insights['modeling'], 1):
            template = template.replace(f"[Consideration {i}]", consideration)
        
        # Recommendations
        recommendations = {
            'preprocessing': [
                "Handle missing oil prices through interpolation",
                "Create consistent holiday flags across the dataset",
                "Normalize sales data by store size/type"
            ],
            'features': [
                "Engineer lag features for oil prices",
                "Create holiday-specific features by type",
                "Add rolling statistics for sales and transactions"
            ],
            'strategy': [
                "Use hierarchical modeling for store-level forecasts",
                "Implement separate models for holiday periods",
                "Consider ensemble of different model types"
            ]
        }
        
        for i, rec in enumerate(recommendations['preprocessing'], 1):
            template = template.replace(f"[Recommendation {i}]", rec)
        
        for i, sug in enumerate(recommendations['features'], 1):
            template = template.replace(f"[Suggestion {i}]", sug)
        
        for i, strat in enumerate(recommendations['strategy'], 1):
            template = template.replace(f"[Strategy {i}]", strat)
        
        return template

if __name__ == "__main__":
    analyzer = DataAnalyzer("dataset/store-sales-time-series-forecasting")
    analyzer.generate_report() 