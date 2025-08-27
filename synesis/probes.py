from typing import Any

import torch
import torch.nn as nn
import torch.nn.init as init


def get_probe(
    model_type: str, in_features: int, n_outputs: int, use_temporal_pooling: bool = False, **kwargs: Any
) -> nn.Module:
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
        return Classifier(in_features, n_outputs, use_temporal_pooling=use_temporal_pooling, **kwargs)
    elif model_type == "regressor":
        return Regressor(in_features, n_outputs, use_temporal_pooling=use_temporal_pooling, **kwargs)
    elif model_type == "transcriber":
        return TranscriberProbe(in_features, n_outputs, **kwargs)
    else:
        raise ValueError(f"Model type '{model_type}' not supported.")


class Classifier(nn.Module):
    """Customizable NN classifier."""

    def __init__(self, in_features, n_outputs, use_temporal_pooling=False, **kwargs):
        super(Classifier, self).__init__()

        self.in_features = in_features
        self.n_outputs = n_outputs
        self.hidden_units = kwargs.get("hidden_units", [])
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.output_activation = kwargs.get("output_activation", None)
        self.use_temporal_pooling = use_temporal_pooling

        if self.use_temporal_pooling:
            self.temporal_pooling = nn.AdaptiveAvgPool1d(1)

        self.layers = nn.ModuleList()
        self.build_layers()

    def temporal_pool(self, x):
        """Apply temporal pooling to the input tensor."""
        x = self.temporal_pooling(x)
        return x.squeeze(-1)

    def build_layers(self):
        """Construct the layers of the neural network."""
        layer_sizes = [self.in_features] + (self.hidden_units or []) + [self.n_outputs]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())

        if self.output_activation == "softmax":
            self.layers.append(nn.Softmax(dim=1))
        elif self.output_activation == "sigmoid":
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        """Define the forward pass of the network."""
        # input for now is batch, channel (1), length
        x = x.squeeze(1)
        
        if self.use_temporal_pooling:
            x = self.temporal_pool(x)

        for layer in self.layers:
            x = layer(x)
        return x

class TranscriberProbe(Classifier):
    def __init__(self, in_features, n_outputs, **kwargs):
        kwargs.pop("output_activation", None) # Sigmoid is hardcoded; not necessary...
        self.dropout_rate = kwargs.get("dropout_rate", 0)
        self.dropout_flag = kwargs.get("dropout_flag", False)
        super().__init__(in_features, n_outputs, **kwargs)
        self.init_weights()

    def init_weights(self):
        """Function to apply weight initialisation. Needs to be called once after the model is instantiated.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()

    def build_layers(self):
        """
            Construct the layers of the TranscriberProbe!
        """
        layer_sizes = [self.in_features] + (self.hidden_units or []) + [self.n_outputs]

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())
                if self.dropout_flag:
                    self.layers.append(nn.Dropout(self.dropout_rate))

    def forward(self, x):
        """Define the forward pass of the network."""
        # input for now is batch, channel (1), length
        x = x.squeeze(1)
        for i, layer in enumerate(self.layers):
            x = layer(x)

        # return the probabilities and feature as well
        return torch.sigmoid(x)

class Regressor(nn.Module):
    """Customizable NN regressor with optional parameter embedding."""

    def __init__(
        self,
        in_features,
        n_outputs,
        emb_param=False,
        emb_param_dim=32,
        use_batch_norm=False,
        use_temporal_pooling=False,
        **kwargs,
    ):
        super(Regressor, self).__init__()
        self.in_features = in_features
        self.n_outputs = n_outputs
        self.emb_param = emb_param
        self.emb_param_dim = emb_param_dim
        self.hidden_units = kwargs.get("hidden_units", [])
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.output_activation = kwargs.get("output_activation", None)
        self.use_batch_norm = use_batch_norm
        self.use_temporal_pooling = use_temporal_pooling

        if self.use_temporal_pooling:
            self.temporal_pooling = nn.AdaptiveAvgPool1d(1)

        # Add parameter embedding layer if enabled
        if self.emb_param:
            param_input_dim = kwargs.get("param_input_dim", 1)
            self.param_encoder = nn.Sequential(
                nn.Linear(param_input_dim, self.emb_param_dim), nn.ReLU()
            )
            # Adjust input features to account for embedded parameter
            self.adjusted_in_features = self.in_features + self.emb_param_dim
        else:
            self.adjusted_in_features = self.in_features

        self.layers = nn.ModuleList()
        self.build_layers()

        if self.use_batch_norm:
            self.input_batch_norm = nn.BatchNorm1d(self.in_features)

    def temporal_pool(self, x):
        """Apply temporal pooling to the input tensor."""
        x = self.temporal_pooling(x)
        return x.squeeze(-1)

    def build_layers(self):
        """Construct the layers of the neural network."""
        # Use adjusted input features to account for parameter embedding
        layer_sizes = (
            [self.adjusted_in_features] + (self.hidden_units or []) + [self.n_outputs]
        )

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())

        if self.output_activation == "relu":
            self.layers.append(nn.ReLU())
        elif self.output_activation == "sigmoid":
            self.layers.append(nn.Sigmoid())

    def forward(self, x, param=None):
        """Define the forward pass of the network.

        Args:
            x: Input tensor of shape (batch, channel (1), length)
            param: Optional parameter tensor of shape (batch, 1) for embedding
        """
        # input will be batch, channel (1), length
        x = x.squeeze(1)

        if self.use_temporal_pooling:
            x = self.temporal_pool(x)

        if self.emb_param:
            if param is None:
                raise ValueError("No transform parameter provided")
            # Embed the parameter
            param_embedding = self.param_encoder(param)
            # Concatenate with input
            x = torch.cat([x, param_embedding], dim=1)

        for layer in self.layers:
            x = layer(x)

        return x
