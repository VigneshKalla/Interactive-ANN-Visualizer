"""
Training module with snapshot capture for visualization.

This module handles:
- Training loop with loss tracking
- Model snapshots at key epochs (0%, 25%, 75%, 100%)
- Support for classification and regression
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from model import calculate_regularization_loss

class ANNTrainer:
    def __init__(self, model, problem_type="classification"):
      
        self.model = model
        self.problem_type = problem_type
        
        # Storage for snapshots and metrics
        self.snapshots = {}  # {epoch: model_state_dict}
        self.train_losses = []
        self.test_losses = []
    
    def prepare_data(self, X, y, test_size=0.2):

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train); X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train); y_test = torch.FloatTensor(y_test)
        
        # For classification, ensure labels are long integers
        if self.problem_type == "classification":
            y_train = y_train.long()
            y_test = y_test.long()
        
        return X_train, X_test, y_train, y_test
    
    def _get_loss_function(self):
      
        if self.problem_type == "classification":
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()
    
    def _calculate_loss(self, outputs, targets, criterion, reg_type=None, reg_strength=0.0):
    
        # Main loss (data fitting)
        data_loss = criterion(outputs, targets)
        
        # Regularization loss (weight penalty)
        reg_loss = calculate_regularization_loss(self.model, reg_type, reg_strength)
        
        # Total loss
        total_loss = data_loss + reg_loss
        
        return total_loss
    
    def train(self, X, y, epochs=100, learning_rate=0.01, reg_type=None, reg_strength=0.0):

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Setup loss and optimizer
        criterion = self._get_loss_function()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Calculate snapshot epochs
        snapshot_epochs = [
            0,  # Initial state
            max(1, epochs // 4),  # 25%
            max(1, 3 * epochs // 4),  # 75%
            epochs - 1] # Final state
         
        # Reset storage
        self.snapshots = {}
        self.train_losses = []
        self.test_losses = []
        
        # Training loop
        for epoch in range(epochs):
            # Training mode
            self.model.train()
            
            # Forward pass
            train_outputs = self.model(X_train)
            train_loss = self._calculate_loss(train_outputs, y_train, criterion, reg_type, reg_strength)
            
            # Backward pass
            optimizer.zero_grad()  # Clear gradients from previous step
            train_loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            
            # Evaluation mode (no gradient tracking)
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test)
                test_loss = self._calculate_loss(test_outputs, y_test, criterion, reg_type, reg_strength)
            
            # Store losses
            self.train_losses.append(train_loss.item())
            self.test_losses.append(test_loss.item())
            
            # Capture snapshots at key epochs
            if epoch in snapshot_epochs:
                # Deep copy the model state
                self.snapshots[epoch] = {
                    "state_dict": {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "test_loss": test_loss.item()}
        
        return {
            "snapshots": self.snapshots,
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "snapshot_epochs": snapshot_epochs}
    
    def predict(self, X):
        self.model.eval()
        
        # Convert to tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)

        with torch.no_grad():
            outputs = self.model(X)
            
            if self.problem_type == "classification":
                # Get class with highest probability
                _, predictions = torch.max(outputs, 1)
                return predictions.numpy()
            else:
                # Return raw outputs for regression
                return outputs.numpy()