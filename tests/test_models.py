"""
Unit tests for model architectures.

This module tests the model creation and forward pass functionality.
"""

import unittest
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    BasicCNN,
    create_mobilenetv2,
    create_efficientnet_b0,
    get_model
)


class TestBasicCNN(unittest.TestCase):
    """Test cases for BasicCNN model."""
    
    def setUp(self):
        """Set up test environment."""
        self.batch_size = 4
        self.num_classes = 2
        self.input_size = (3, 224, 224)
        self.model = BasicCNN(num_classes=self.num_classes)
    
    def test_model_initialization(self):
        """Test model initialization and parameter count."""
        self.assertIsInstance(self.model, BasicCNN)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 0)
        print(f"BasicCNN parameters: {total_params:,}")
    
    def test_forward_pass(self):
        """Test forward pass of the model."""
        # Create random input
        x = torch.randn(self.batch_size, *self.input_size)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_output_range(self):
        """Test that output values are in a reasonable range."""
        x = torch.randn(self.batch_size, *self.input_size)
        output = self.model(x)
        
        # Logits can be any value, but we check they're finite
        self.assertTrue(torch.isfinite(output).all())
    
    def test_dropout_layer(self):
        """Test that dropout layer exists and is configured correctly."""
        # Check if dropout layer is present
        has_dropout = False
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                has_dropout = True
                self.assertEqual(module.p, 0.5)  # Default dropout rate
                break
        self.assertTrue(has_dropout, "Dropout layer not found in model")


class TestMobileNetV2(unittest.TestCase):
    """Test cases for MobileNetV2 model."""
    
    def setUp(self):
        """Set up test environment."""
        self.batch_size = 4
        self.num_classes = 2
        self.input_size = (3, 224, 224)
    
    def test_model_initialization(self):
        """Test MobileNetV2 initialization."""
        model = create_mobilenetv2(num_classes=self.num_classes, pretrained=False)
        self.assertIsInstance(model, torch.nn.Module)
        
        # Check classifier head
        self.assertEqual(model.classifier[1].out_features, self.num_classes)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        print(f"MobileNetV2 parameters: {total_params:,}")
    
    def test_forward_pass(self):
        """Test forward pass of MobileNetV2."""
        model = create_mobilenetv2(num_classes=self.num_classes, pretrained=False)
        x = torch.randn(self.batch_size, *self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


class TestEfficientNetB0(unittest.TestCase):
    """Test cases for EfficientNet-B0 model."""
    
    def setUp(self):
        """Set up test environment."""
        self.batch_size = 4
        self.num_classes = 2
        self.input_size = (3, 224, 224)
    
    def test_model_initialization(self):
        """Test EfficientNet-B0 initialization."""
        model = create_efficientnet_b0(num_classes=self.num_classes, pretrained=False)
        self.assertIsInstance(model, torch.nn.Module)
        
        # Check classifier head
        self.assertEqual(model.classifier[1].out_features, self.num_classes)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
        print(f"EfficientNet-B0 parameters: {total_params:,}")
    
    def test_forward_pass(self):
        """Test forward pass of EfficientNet-B0."""
        model = create_efficientnet_b0(num_classes=self.num_classes, pretrained=False)
        x = torch.randn(self.batch_size, *self.input_size)
        output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


class TestModelFactory(unittest.TestCase):
    """Test cases for model factory function."""
    
    def test_get_model_basic_cnn(self):
        """Test getting BasicCNN from factory."""
        model = get_model('baseline_cnn', num_classes=2)
        self.assertIsInstance(model, BasicCNN)
    
    def test_get_model_mobilenetv2(self):
        """Test getting MobileNetV2 from factory."""
        model = get_model('mobilenetv2', num_classes=2, pretrained=False)
        self.assertIsInstance(model, torch.nn.Module)
    
    def test_get_model_efficientnet_b0(self):
        """Test getting EfficientNet-B0 from factory."""
        model = get_model('efficientnet_b0', num_classes=2, pretrained=False)
        self.assertIsInstance(model, torch.nn.Module)
    
    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with self.assertRaises(ValueError):
            get_model('invalid_model', num_classes=2)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()