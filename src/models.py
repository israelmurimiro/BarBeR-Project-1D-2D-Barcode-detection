"""
Model architectures for BarBeR-Project.

This module contains the BasicCNN model and implementations
of MobileNetV2 and EfficientNet-B0 for transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import yaml
from pathlib import Path


class BasicCNN(nn.Module):
    """
    Basic CNN for barcode classification (1D vs 2D).
    
    Architecture:
    - 3 convolutional layers with ReLU and MaxPooling
    - AdaptiveAvgPool2d to fixed size
    - 2 fully connected layers with dropout
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        dropout_rate (float): Dropout rate (default: 0.5)
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(BasicCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to ensure consistent size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # Convolutional layers with pooling
        x = self.pool(F.relu(self.conv1(x)))  # 224x224 -> 112x112
        x = self.pool(F.relu(self.conv2(x)))  # 112x112 -> 56x56
        x = self.pool(F.relu(self.conv3(x)))  # 56x56 -> 28x28
        
        # Adaptive pooling to fixed size (7x7)
        x = self.adaptive_pool(x)  # 28x28 -> 7x7
        
        # Flatten
        x = x.view(-1, 128 * 7 * 7)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_mobilenetv2(num_classes=2, pretrained=True):
    """
    Create MobileNetV2 model for transfer learning.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: MobileNetV2 model
    """
    model = models.mobilenet_v2(weights='DEFAULT' if pretrained else None)
    
    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model


def create_efficientnet_b0(num_classes=2, pretrained=True):
    """
    Create EfficientNet-B0 model for transfer learning.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: EfficientNet-B0 model
    """
    model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
    
    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model


def get_model(model_name, num_classes=2, pretrained=True, dropout_rate=0.5):
    """
    Factory function to get a model by name.
    
    Args:
        model_name (str): Name of the model ('baseline_cnn', 'mobilenetv2', 'efficientnet_b0')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights (for transfer learning models)
        dropout_rate (float): Dropout rate (for BasicCNN)
        
    Returns:
        nn.Module: The requested model
    """
    model_name = model_name.lower()
    
    if model_name == 'baseline_cnn' or model_name == 'basiccnn':
        return BasicCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name == 'mobilenetv2':
        return create_mobilenetv2(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        return create_efficientnet_b0(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from: 'baseline_cnn', 'mobilenetv2', 'efficientnet_b0'")


def get_model_from_config(config_path):
    """
    Get model using configuration file.
    
    Args:
        config_path (str or Path): Path to config.yaml file
        
    Returns:
        nn.Module: The configured model
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['models']['classification']
    model_name = model_config['name']
    num_classes = model_config['num_classes']
    dropout = model_config.get('dropout', 0.5)
    
    # For transfer learning models, use pretrained=True
    if model_name.lower() in ['mobilenetv2', 'efficientnet_b0']:
        return get_model(model_name, num_classes=num_classes, pretrained=True)
    else:
        return get_model(model_name, num_classes=num_classes, dropout_rate=dropout)


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test BasicCNN
    model = BasicCNN(num_classes=2)
    print(f"BasicCNN: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test MobileNetV2
    model = create_mobilenetv2(num_classes=2, pretrained=False)
    print(f"MobileNetV2: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test EfficientNet-B0
    model = create_efficientnet_b0(num_classes=2, pretrained=False)
    print(f"EfficientNet-B0: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("All models created successfully!")