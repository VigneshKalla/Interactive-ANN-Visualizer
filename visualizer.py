"""
Visualization module for ANN decision boundaries and metrics.

This module creates:
- Decision boundary plots showing how the model classifies different regions
- Loss curves showing training progress
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y, title="Decision Boundary", resolution=0.02):
    # Set model to evaluation mode
    model.eval()
    
    # Define bounds with padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Create a mesh grid covering the entire space
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution))
    
    # Flatten grid to pass through model
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid_points)
    
    # Get predictions for all grid points
    with torch.no_grad():
        outputs = model(grid_tensor)
        
        # For classification, get the predicted class
        if outputs.shape[1] > 1:  # Multi-class
            _, predictions = torch.max(outputs, 1)
            Z = predictions.numpy()
        else:  # Binary (single output)
            Z = (outputs > 0).squeeze().numpy().astype(int)
    
    # Reshape predictions to match grid
    Z = Z.reshape(xx.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot decision boundary (colored regions)
    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background, levels=np.arange(Z.max() + 2) - 0.5)
    # Plot actual data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="black", s=50, alpha=0.8)
    
    # Formatting
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for classes
    plt.colorbar(scatter, ax=ax, label="Class"); plt.tight_layout()
    
    return fig


def plot_snapshot_grid(snapshots, snapshot_epochs, model_class, model_config, X, y):

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    # Titles for each subplot
    titles = [
        f"Initial State (Epoch {snapshot_epochs[0]})", f"25% Training (Epoch {snapshot_epochs[1]})",
        f"75% Training (Epoch {snapshot_epochs[2]})", f"Final State (Epoch {snapshot_epochs[3]})"]
    
    # Define bounds for consistent visualization across all subplots
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    resolution = 0.02
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid_points)
    
    # Plot each snapshot
    for idx, (epoch, ax, title) in enumerate(zip(snapshot_epochs, axes, titles)):
        # Load snapshot into a new model instance
        model = model_class(**model_config)
        model.load_state_dict(snapshots[epoch]["state_dict"])
        model.eval()
        
        # Get predictions for grid
        with torch.no_grad():
            outputs = model(grid_tensor)
            
            if outputs.shape[1] > 1:
                _, predictions = torch.max(outputs, 1)
                Z = predictions.numpy()
            else:
                Z = (outputs > 0).squeeze().numpy().astype(int)
        
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background, levels=np.arange(Z.max() + 2) - 0.5)
        
        # Plot data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="black", s=40, alpha=0.8)
        
        # Formatting
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Feature 1", fontsize=10)
        ax.set_ylabel("Feature 2", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        # Add loss info to subplot
        train_loss = snapshots[epoch]["train_loss"]
        test_loss = snapshots[epoch]["test_loss"]
        ax.text(0.02, 0.98, f"Train Loss: {train_loss:.4f}\nTest Loss: {test_loss:.4f}",
               transform=ax.transAxes, fontsize=9,
               verticalalignment="top",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    
    return fig


def plot_loss_curves(train_losses, test_losses):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot both losses
    ax.plot(epochs, train_losses, label="Training Loss", color="blue", linewidth=2, alpha=0.7)
    ax.plot(epochs, test_losses, label="Test Loss", color="red", linewidth=2, alpha=0.7)
    
    # Find minimum test loss
    min_test_loss = min(test_losses)
    min_epoch = test_losses.index(min_test_loss) + 1
    ax.axvline(x=min_epoch, color="green", linestyle="--", alpha=0.5, label=f"Best Test Loss (Epoch {min_epoch})")
    
    # Formatting
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Test Loss Over Time", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add text annotation for final losses
    final_train = train_losses[-1]
    final_test = test_losses[-1]
    ax.text(0.98, 0.98, 
           f"Final Train Loss: {final_train:.4f}\nFinal Test Loss: {final_test:.4f}",
           transform=ax.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="right",
           bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
    plt.tight_layout()
    
    return fig

def plot_dataset_preview(X, y, title="Dataset Preview"):

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="black", s=40, alpha=0.7)
    
    ax.set_xlabel("Feature 1", fontsize=10)
    ax.set_ylabel("Feature 2", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label="Class"); plt.tight_layout()
    
    return fig