#!/usr/bin/env python3
"""
Simple FFNN model for SSM state transformation
Input: doc1 SSM state
Output: predicted SSM state difference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """
    Simple MLP to transform SSM states
    """

    def __init__(self, input_dim=64*5120*16, hidden_dims=[128], output_dim=64*5120*16):
        super(SimpleMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            # Better weight initialization
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        output_linear = nn.Linear(prev_dim, output_dim)
        # Normal initialization for output layer
        nn.init.xavier_uniform_(output_linear.weight)
        nn.init.zeros_(output_linear.bias)
        layers.append(output_linear)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [batch_size, num_layers, d_inner, d_state] -> flattened to [batch_size, num_layers * d_inner * d_state]
        Returns:
            Predicted SSM state difference: [batch_size, num_layers, d_inner, d_state]
        """
        # Flatten input
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)  # [batch_size, num_layers * d_inner * d_state]

        # Pass through network
        output_flat = self.network(x_flat)

        # Reshape output
        output = output_flat.view(batch_size, 64, 5120, 16)  # [batch_size, num_layers, d_inner, d_state]

        return output


def create_model():
    """Create and return the model"""
    return SimpleMLP()


if __name__ == "__main__":
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleMLP().to(device)
    model.eval()

    batch_size = 2
    # create input directly on the target device
    input_tensor = torch.randn(batch_size, 64, 5120, 16, device=device)

    try:
        with torch.no_grad():
            output = model(input_tensor)
    except RuntimeError as e:
        # likely OOM on GPU for this very large FC model; retry on CPU
        print("RuntimeError during forward (possible OOM):", e)
        print("Retrying on CPU...")
        model = model.to("cpu")
        input_tensor = input_tensor.to("cpu")
        with torch.no_grad():
            output = model(input_tensor)
        device = torch.device("cpu")

    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device: {input_tensor.device}, shape: {input_tensor.shape}")
    print(f"Output device: {output.device}, shape: {output.shape}")
    print(f"Shapes match: {input_tensor.shape == output.shape}")
    print(model)