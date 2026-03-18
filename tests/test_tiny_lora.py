import unittest

import torch
from torch import nn

from latent_space.tiny_lora import TinyLoRALinear


class TestTinyLoRALinear(unittest.TestCase):
    def setUp(self) -> None:
        """Set up standard parameters used across multiple tests."""
        self.batch_size = 4
        self.seq_len = 10
        self.in_features = 32
        self.out_features = 64
        self.r = 2
        self.u = 1

        # Create a base linear layer to wrap
        self.base_layer = nn.Linear(self.in_features, self.out_features)
        self.dummy_input = torch.randn(self.batch_size, self.seq_len, self.in_features)

    def test_initialization_and_shapes(self) -> None:
        """Test if the layer initializes with the correct buffer and parameter shapes."""
        tiny_layer = TinyLoRALinear(self.base_layer, r=self.r, u=self.u)

        # Check buffer shapes
        assert tiny_layer.U.shape == (self.out_features, self.r)
        assert tiny_layer.Sigma.shape == (self.r, self.r)
        assert tiny_layer.Vh.shape == (self.r, self.in_features)
        assert tiny_layer.P.shape == (self.u, self.r, self.r)

        # Check parameter shape
        assert tiny_layer.v.shape == (self.u,)

        # Check if base weights are properly frozen
        assert not tiny_layer.weight.requires_grad
        assert not tiny_layer.bias.requires_grad

    def test_forward_pass_shape(self) -> None:
        """Test if the forward pass executes and returns the expected shape."""
        tiny_layer = TinyLoRALinear(self.base_layer, r=self.r, u=self.u)

        output = tiny_layer(self.dummy_input)
        expected_shape = (self.batch_size, self.seq_len, self.out_features)

        assert output.shape == expected_shape

    def test_gradient_flow(self) -> None:
        """Test that gradients only flow to the trainable vector `v`."""
        tiny_layer = TinyLoRALinear(self.base_layer, r=self.r, u=self.u)

        # Forward pass
        output = tiny_layer(self.dummy_input)

        # Dummy loss (sum of all elements)
        loss = output.sum()
        loss.backward()

        # Check that `v` gets a gradient
        assert tiny_layer.v.grad is not None

        # Check that frozen weights DO NOT get gradients
        assert tiny_layer.weight.grad is None
        assert tiny_layer.bias.grad is None

    def test_weight_tying(self) -> None:
        """Test if weight tying across multiple modules works correctly."""
        # Create a shared parameter `v`
        shared_v = nn.Parameter(torch.zeros(self.u))

        # Wrap two different base layers with the same shared `v`
        layer1 = TinyLoRALinear(nn.Linear(self.in_features, self.out_features), shared_v=shared_v)
        layer2 = TinyLoRALinear(nn.Linear(self.in_features, self.out_features), shared_v=shared_v)

        # Verify they point to the exact same memory address
        assert layer1.v is layer2.v
        assert layer1.v is shared_v

        # Forward pass through both
        out1 = layer1(self.dummy_input)
        out2 = layer2(self.dummy_input)

        # Backward pass on both
        loss = out1.sum() + out2.sum()
        loss.backward()

        # The shared parameter should have accumulated gradients from BOTH layers
        assert shared_v.grad is not None
        assert not torch.allclose(shared_v.grad, torch.zeros_like(shared_v.grad))


if __name__ == "__main__":
    unittest.main()
