# PyTorch ANN model with configurable architecture.

import torch
import torch.nn as nn

#  Flexible Artificial Neural Network with configurable architecture.
class FlexibleANN(nn.Module):
 
    def __init__(self, input_size, hidden_layers, neurons_per_layer, output_size, activation="ReLu"):
        
        super(FlexibleANN, self).__init__()
        
        # Store configuration for later inspection
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.output_size = output_size
        self.activation_name = activation
        
        # Build the network layer by layer
        self.layers = nn.ModuleList()
        
        # First hidden layer: input -> first hidden
        if hidden_layers > 0:
            self.layers.append(nn.Linear(input_size, neurons_per_layer))
            
            # Additional hidden layers: hidden -> hidden
            for _ in range(hidden_layers - 1):
                self.layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            
            # Output layer: last hidden -> output
            self.layers.append(nn.Linear(neurons_per_layer, output_size))
        else:
            # No hidden layers: direct input -> output (linear model)
            self.layers.append(nn.Linear(input_size, output_size))
        
        # Set activation function
        self.activation = self._get_activation(activation)
        
    def _get_activation(self, activation_name):
        activations = {
            "ReLu": nn.ReLU(),
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "Linear": nn.Identity()}  # No activation
        
        if activation_name not in activations:
            raise ValueError(f"Unknown activation: {activation_name}")
        
        return activations[activation_name]
    
    def forward(self, x):
        # Pass through all hidden layers with activation
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        # Final layer WITHOUT activation
        # (Activation will be handled by loss function for stability)
        x = self.layers[-1](x)
        
        return x
    
    def get_architecture_summary(self):
        summary = f"Input Layer: {self.input_size} features\n"
        
        if self.hidden_layers > 0:
            summary += f"Hidden Layers: {self.hidden_layers} layers x {self.neurons_per_layer} neurons\n"
            summary += f"Activation: {self.activation_name}\n"
        else:
            summary += "Hidden Layers: None (Linear model)\n"
        
        summary += f"Output Layer: {self.output_size} neuron(s)"
        
        return summary
    
    def count_parameters(self):

        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def calculate_regularization_loss(model, reg_type, reg_strength):
    if reg_type is None or reg_strength == 0:
        return 0.0
    
    reg_loss = 0.0
    
    for param in model.parameters():
        if reg_type.lower() == "l1":
            # L1: Sum of absolute values
            reg_loss += torch.sum(torch.abs(param))
        elif reg_type.lower() == "l2":
            # L2: Sum of squared values
            reg_loss += torch.sum(param ** 2)
    
    return reg_strength * reg_loss