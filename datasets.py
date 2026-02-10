"""
Dataset generation utilities for ANN visualizer.

This module provides:
- Predefined toy datasets (moons, circles, classification, XOR)
- Custom multi-class datasets with configurable class balance
"""

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification

# Generate XOR dataset - classic non-linearly separable problem.
def generate_xor_dataset(n_samples=200, noise=0.1):
  
    # Generate points in four quadrants
    half = n_samples // 2; quarter = half // 2
    
    # Quadrant 1: (positive, positive) -> label 0
    X1 = np.random.randn(quarter, 2) * 0.3 + np.array([1, 1])
    y1 = np.zeros(quarter)
    
    # Quadrant 2: (negative, positive) -> label 1
    X2 = np.random.randn(quarter, 2) * 0.3 + np.array([-1, 1])
    y2 = np.ones(quarter)
    
    # Quadrant 3: (negative, negative) -> label 0
    X3 = np.random.randn(quarter, 2) * 0.3 + np.array([-1, -1])
    y3 = np.zeros(quarter)
    
    # Quadrant 4: (positive, negative) -> label 1
    X4 = np.random.randn(quarter, 2) * 0.3 + np.array([1, -1])
    y4 = np.ones(quarter)
    
    # Combine all quadrants
    X = np.vstack([X1, X2, X3, X4]); y = np.hstack([y1, y2, y3, y4])
    
    # Add noise to make it more realistic
    X += np.random.randn(*X.shape) * noise
    
    return X, y.astype(int)

#  Get a predefined dataset
def get_predefined_dataset(dataset_name, n_samples=200, noise=0.1):
    
    if dataset_name == "Moons":
        # Two interleaving half circles
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        
    elif dataset_name == "Circles":
        # One circle inside another
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        
    elif dataset_name == "Classification":
        # General classification with configurable complexity
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,  # 2D for easy visualization
            n_redundant=0,  # No redundant features
            n_informative=2,  # Both features are useful
            n_clusters_per_class=1,
            flip_y=noise,  # Use noise parameter to flip labels
            random_state=42)
        
    elif dataset_name == "XOR":
        X, y = generate_xor_dataset(n_samples=n_samples, noise=noise)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return X, y

# Create a custom dataset with user-defined parameters.
def create_custom_dataset(n_samples=200, n_classes=2, noise=0.1, class_weights=None):

    # Handle class weights
    if class_weights is None:
        # Balanced classes
        class_weights = [1.0 / n_classes] * n_classes
    else:
        # Normalize weights to sum to 1
        total = sum(class_weights)
        class_weights = [w / total for w in class_weights]
    
    # Calculate samples per class
    samples_per_class = [int(n_samples * w) for w in class_weights]
    # Fix rounding errors
    samples_per_class[-1] += n_samples - sum(samples_per_class)
    # Generate class clusters
    X_list = []; y_list = []
    
    # Generate clusters for each class
    # Arrange centers in a circle for visual separation
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    radius = 3.0
    
    for class_idx in range(n_classes):
        n = samples_per_class[class_idx]
        
        # Center for this class
        center_x = radius * np.cos(angles[class_idx])
        center_y = radius * np.sin(angles[class_idx])
        center = np.array([center_x, center_y])
        
        # Generate points around this center
        X_class = np.random.randn(n, 2) * (1.0 + noise) + center
        y_class = np.full(n, class_idx)
        
        X_list.append(X_class); y_list.append(y_class)
    
    # Combine all classes 
    X = np.vstack(X_list)
    y = np.hstack(y_list).astype(int)
    
    # Shuffle to mix classes
    indices = np.random.permutation(len(X))
    X = X[indices]; y = y[indices]
    
    return X, y