"""
Script to demonstrate the data processing pipeline.
"""

import os
import logging
from data_processor import DataProcessor

def main():
    """Run the data processing pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize data processor
        logger.info("Initializing data processor...")
        processor = DataProcessor()
        
        # Load data
        logger.info("Loading datasets...")
        datasets = processor.load_data()
        
        # Preprocess data
        logger.info("Preprocessing datasets...")
        processed = processor.preprocess_data(datasets)
        
        # Get the final merged dataset
        final_data = processed['final']
        logger.info(f"Final dataset shape: {final_data.shape}")
        
        # Split data into train and test sets
        logger.info("Splitting data into train and test sets...")
        train_data, test_data = processor.split_data(final_data)
        
        # Scale numerical features
        logger.info("Scaling numerical features...")
        numerical_features = ['sales', 'transactions', 'dcoilwtico']
        train_scaled, test_scaled = processor.scale_features(
            train_data,
            test_data,
            numerical_features
        )
        
        # Create sequences for time series prediction
        logger.info("Creating sequences for time series prediction...")
        X_train, y_train = processor.create_sequences(train_scaled)
        X_test, y_test = processor.create_sequences(test_scaled)
        
        # Log processing results
        logger.info("Data processing completed successfully")
        logger.info(f"Training sequences shape: {X_train.shape}")
        logger.info(f"Training targets shape: {y_train.shape}")
        logger.info(f"Testing sequences shape: {X_test.shape}")
        logger.info(f"Testing targets shape: {y_test.shape}")
        
        # Save processed data
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        train_scaled.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        test_scaled.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        logger.info(f"Saved processed data to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {e}")
        raise

if __name__ == "__main__":
    main() 