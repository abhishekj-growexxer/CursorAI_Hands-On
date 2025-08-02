"""
Script to demonstrate the feature engineering process.
"""

import os
import logging
import pandas as pd
from data_processor import DataProcessor
from feature_engineering import FeatureEngineer
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(importance_df: pd.DataFrame, output_dir: str):
    """Plot feature importance scores."""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def main():
    """Run the feature engineering process."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize processors
        logger.info("Initializing data processor and feature engineer...")
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        datasets = data_processor.load_data()
        processed = data_processor.preprocess_data(datasets)
        
        # Get the final merged dataset
        data = processed['final']
        logger.info(f"Initial dataset shape: {data.shape}")
        
        # Create all features
        logger.info("Creating engineered features...")
        data_with_features = feature_engineer.create_all_features(data)
        logger.info(f"Dataset shape after feature engineering: {data_with_features.shape}")
        
        # Calculate feature importance
        logger.info("Calculating feature importance...")
        importance_correlation = feature_engineer.get_feature_importance(
            data_with_features,
            'sales',
            method='correlation'
        )
        
        importance_mutual_info = feature_engineer.get_feature_importance(
            data_with_features,
            'sales',
            method='mutual_info'
        )
        
        # Create output directory
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed data and feature importance
        logger.info("Saving results...")
        data_with_features.to_csv(
            os.path.join(output_dir, "data_with_features.csv"),
            index=False
        )
        
        importance_correlation.to_csv(
            os.path.join(output_dir, "feature_importance_correlation.csv"),
            index=False
        )
        
        importance_mutual_info.to_csv(
            os.path.join(output_dir, "feature_importance_mutual_info.csv"),
            index=False
        )
        
        # Plot feature importance
        logger.info("Creating feature importance plots...")
        plot_feature_importance(
            importance_correlation,
            output_dir
        )
        
        # Log feature statistics
        logger.info("\nFeature Statistics:")
        logger.info(f"Total number of features: {len(data_with_features.columns)}")
        logger.info("\nTop 10 most important features (correlation):")
        for _, row in importance_correlation.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
            
        logger.info("\nTop 10 most important features (mutual information):")
        for _, row in importance_mutual_info.head(10).iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")
            
        logger.info("Feature engineering process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in feature engineering process: {e}")
        raise

if __name__ == "__main__":
    main() 