import torch
import torch.nn as nn
import unittest

from latent_space.tiny_lora import TinyLoRALinear

class TestTinyLoRALinear(unittest.TestCase):
    def setUp(self):
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

    def test_initialization_and_shapes(self):
        """Test if the layer initializes with the correct buffer and parameter shapes."""
        tiny_layer = TinyLoRALinear(self.base_layer, r=self.r, u=self.u)
        
        # Check buffer shapes
        self.assertEqual(tiny_layer.U.shape, (self.out_features, self.r))
        self.assertEqual(tiny_layer.Sigma.shape, (self.r, self.r))
        self.assertEqual(tiny_layer.Vh.shape, (self.r, self.in_features))
        self.assertEqual(tiny_layer.P.shape, (self.u, self.r, self.r))
        
        # Check parameter shape
        self.assertEqual(tiny_layer.v.shape, (self.u,))
        
        # Check if base weights are properly frozen
        self.assertFalse(tiny_layer.weight.requires_grad)
        self.assertFalse(tiny_layer.bias.requires_grad)

    def test_forward_pass_shape(self):
        """Test if the forward pass executes and returns the expected shape."""
        tiny_layer = TinyLoRALinear(self.base_layer, r=self.r, u=self.u)
        
        output = tiny_layer(self.dummy_input)
        expected_shape = (self.batch_size, self.seq_len, self.out_features)
        
        self.assertEqual(output.shape, expected_shape)

    def test_gradient_flow(self):
        """Test that gradients only flow to the trainable vector `v`."""
        tiny_layer = TinyLoRALinear(self.base_layer, r=self.r, u=self.u)
        
        # Forward pass
        output = tiny_layer(self.dummy_input)
        
        # Dummy loss (sum of all elements)
        loss = output.sum()
        loss.backward()
        
        # Check that `v` gets a gradient
        self.assertIsNotNone(tiny_layer.v.grad)
        
        # Check that frozen weights DO NOT get gradients
        self.assertIsNone(tiny_layer.weight.grad)
        self.assertIsNone(tiny_layer.bias.grad)

    def test_weight_tying(self):
        """Test if weight tying across multiple modules works correctly."""
        # Create a shared parameter `v`
        shared_v = nn.Parameter(torch.zeros(self.u))
        
        # Wrap two different base layers with the same shared `v`
        layer1 = TinyLoRALinear(nn.Linear(self.in_features, self.out_features), shared_v=shared_v)
        layer2 = TinyLoRALinear(nn.Linear(self.in_features, self.out_features), shared_v=shared_v)
        
        # Verify they point to the exact same memory address
        self.assertIs(layer1.v, layer2.v)
        self.assertIs(layer1.v, shared_v)
        
        # Forward pass through both
        out1 = layer1(self.dummy_input)
        out2 = layer2(self.dummy_input)
        
        # Backward pass on both
        loss = out1.sum() + out2.sum()
        loss.backward()
        
        # The shared parameter should have accumulated gradients from BOTH layers
        self.assertIsNotNone(shared_v.grad)
        self.assertFalse(torch.allclose(shared_v.grad, torch.zeros_like(shared_v.grad)))

if __name__ == '__main__':
    unittest.main()