import torch.nn as nn
from typing import List, Optional

def get_probe(
    model_type: str,
    in_features: int,
    n_outputs: int,
    hidden_units: Optional[List[int]] = None,
    weight_decay: float = 0,
    output_activation: str = "softmax"
):
    """
    Factory function to create and return a model based on the specified type.

    Args:
        model_type: Type of model to create ("classifier" or "regressor").
        in_features: Dimensions of the input data (non-batched).
        n_outputs: Number of classes for classification/outputs for regression.
        hidden_units: List of integers specifying the number of units in each
                      hidden layer.
        weight_decay: L2 regularization factor.
        output_activation: Activation function for the output layer.

    Returns:
        nn.Module: The constructed model.

    Raises:
        ValueError: If an unsupported model type is specified.
    """
    if model_type == "classifier":
        return Classifier(
            in_features, n_outputs, hidden_units, weight_decay, output_activation
        )
    elif model_type == "regressor":
        return Regressor(
            in_features, n_outputs, hidden_units, weight_decay, output_activation
        )
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")

class Classifier(nn.Module):
    """Customizable NN classifier."""
    def __init__(
        self, in_features, n_classes, hidden_units, weight_decay, output_activation
    ):
        super(Classifier, self).__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.hidden_units = hidden_units
        self.weight_decay = weight_decay
        self.output_activation = output_activation

        self.layers = nn.ModuleList()
        self.build_layers()

    def build_layers(self):
        """Construct the layers of the neural network."""
        layer_sizes = [self.in_features] + (self.hidden_units or []) + [self.n_classes]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())

        if self.output_activation == "softmax":
            self.layers.append(nn.Softmax(dim=1))
        elif self.output_activation == "sigmoid":
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        """Define the forward pass of the network."""
        for layer in self.layers:
            x = layer(x)
        return x


class Regressor(nn.Module):
    """Customizable NN regressor."""

    def __init__(
        self, in_features, n_outputs, hidden_units, weight_decay, output_activation
    ):
        super(Regressor, self).__init__()
        self.in_features = in_features
        self.n_outputs = n_outputs
        self.hidden_units = hidden_units
        self.weight_decay = weight_decay
        self.output_activation = output_activation

        self.layers = nn.ModuleList()
        self.build_layers()

    def build_layers(self):
        """Construct the layers of the neural network."""
        layer_sizes = [self.in_features] + (self.hidden_units or []) + [self.n_outputs]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())

        if self.output_activation == "relu":
            self.layers.append(nn.ReLU())
        elif self.output_activation == "sigmoid":
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        """Define the forward pass of the network."""
        for layer in self.layers:
            x = layer(x)
        return x
