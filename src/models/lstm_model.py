"""
LSTM model for time series forecasting.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.base_model import BaseTimeSeriesModel
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TimeSeriesDataset(Dataset):
    """Dataset class for time series data."""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        sequence_length: int = 30
    ):
        """Initialize dataset."""
        self.features = torch.FloatTensor(features)
        self.sequence_length = sequence_length
        
        if targets is not None:
            self.targets = torch.FloatTensor(targets).reshape(-1, 1)
        else:
            self.targets = None
            
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.features) - self.sequence_length + 1
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get item by index."""
        x = self.features[idx:idx + self.sequence_length]
        
        if self.targets is not None:
            y = self.targets[idx + self.sequence_length - 1]
            return x, y
        return x, None

class LSTMModel(nn.Module):
    """LSTM neural network model."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2
    ):
        """Initialize LSTM model."""
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class LSTMTimeSeriesModel(BaseTimeSeriesModel):
    """LSTM model for time series forecasting."""
    
    def __init__(self, config_path: str = "configs/model_configs/default.yaml"):
        """Initialize LSTM model."""
        super().__init__("lstm", config_path)
        
        # Set default parameters if not in config
        self.sequence_length = self.config.get('sequence_length', 30)
        self.hidden_size = self.config.get('hidden_size', 64)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.num_epochs = self.config.get('num_epochs', 100)
        
        logger.debug(f"Model parameters: {self.config}")
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize lists for training curves
        self.train_losses = []
        self.val_losses = []
        
    def preprocess(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess data for LSTM model."""
        logger.info("Preprocessing data for LSTM")
        try:
            # Extract features and targets
            features = data[['sales']].values
            
            if is_training:
                return features[:-1], features[1:]
            return features, None
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
            
    def create_dataloaders(
        self,
        features: np.ndarray,
        targets: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None
    ) -> DataLoader:
        """Create data loaders for training."""
        logger.debug("Creating data loaders")
        try:
            batch_size = batch_size or self.batch_size
            
            # Create dataset
            dataset = TimeSeriesDataset(
                features=features,
                targets=targets,
                sequence_length=self.sequence_length
            )
            
            # Create data loader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True if targets is not None else False
            )
            
            return dataloader
            
        except Exception as e:
            logger.error(f"Error creating data loaders: {str(e)}")
            raise
            
    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Train the LSTM model."""
        logger.info("Starting LSTM model training")
        try:
            # Preprocess data
            X_train, y_train = self.preprocess(train_data)
            train_loader = self.create_dataloaders(X_train, y_train)
            
            if validation_data is not None:
                X_val, y_val = self.preprocess(validation_data)
                val_loader = self.create_dataloaders(X_val, y_val)
            else:
                val_loader = None
                
            # Initialize model
            input_size = 1  # Single feature (sales)
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            
            # Initialize optimizer and loss function
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
            criterion = nn.MSELoss()
            
            # Training loop
            best_val_loss = float('inf')
            self.train_losses = []
            self.val_losses = []
            
            for epoch in range(self.num_epochs):
                # Training
                self.model.train()
                epoch_loss = 0.0
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    
                avg_train_loss = epoch_loss / len(train_loader)
                self.train_losses.append(avg_train_loss)
                
                # Validation
                if val_loader:
                    val_loss = self._validate(val_loader, criterion)
                    self.val_losses.append(val_loss)
                    
                    # Log progress
                    logger.debug(
                        f"Epoch {epoch + 1}/{self.num_epochs}, "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}"
                    )
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save('models/lstm/best_model.pth')
                else:
                    logger.debug(
                        f"Epoch {epoch + 1}/{self.num_epochs}, "
                        f"Train Loss: {avg_train_loss:.4f}"
                    )
                    
            # Load best model
            if val_loader:
                self.load('models/lstm/best_model.pth')
                
            # Calculate final metrics
            final_metrics = {
                'train_loss': self.train_losses[-1],
                'val_loss': self.val_losses[-1] if self.val_losses else None,
                'best_val_loss': best_val_loss if val_loader else None
            }
            
            logger.info(f"Training completed with metrics: {final_metrics}")
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
        return val_loss / len(val_loader)
        
    def predict(
        self,
        data: pd.DataFrame,
        prediction_steps: int
    ) -> np.ndarray:
        """Generate predictions."""
        logger.info(f"Generating predictions for {prediction_steps} steps")
        try:
            if self.model is None:
                raise ValueError("Model not trained")
                
            # Preprocess data
            features, _ = self.preprocess(data, is_training=False)
            dataloader = self.create_dataloaders(features)
            
            # Generate predictions
            self.model.eval()
            predictions = []
            
            with torch.no_grad():
                for batch_x, _ in dataloader:
                    batch_x = batch_x.to(self.device)
                    output = self.model(batch_x)
                    predictions.extend(output.cpu().numpy())
                    
            predictions = np.array(predictions)
            predictions = predictions.reshape(prediction_steps, -1)
            logger.debug(f"Generated predictions with shape: {predictions.shape}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
            
    def save(self, path: str):
        """Save model to disk."""
        logger.info(f"Saving model to {path}")
        try:
            if self.model is None:
                raise ValueError("No model to save")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state and configuration
            save_dict = {
                'model_state': self.model.state_dict(),
                'config': self.config,
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
            
            torch.save(save_dict, path)
            logger.debug("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    def load(self, path: str):
        """Load model from disk."""
        logger.info(f"Loading model from {path}")
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
                
            # Load model state and configuration
            save_dict = torch.load(path, map_location=self.device)
            
            # Recreate model with saved configuration
            self.config = save_dict['config']
            self.sequence_length = save_dict['sequence_length']
            self.hidden_size = save_dict['hidden_size']
            self.num_layers = save_dict['num_layers']
            self.dropout = save_dict['dropout']
            self.train_losses = save_dict.get('train_losses', [])
            self.val_losses = save_dict.get('val_losses', [])
            
            # Initialize model
            self.model = LSTMModel(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            
            # Load state
            self.model.load_state_dict(save_dict['model_state'])
            logger.debug("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training and validation loss curves."""
        logger.info("Plotting training curves")
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, label='Training Loss')
            if self.val_losses:
                plt.plot(self.val_losses, label='Validation Loss')
                
            plt.title('Training Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            if save_path:
                plt.savefig(save_path)
                logger.debug(f"Saved training curves to {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting training curves: {str(e)}")
            raise 