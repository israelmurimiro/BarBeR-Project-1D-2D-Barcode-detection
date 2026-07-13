"""
Utility functions for BarBeR-Project.

This module contains helper functions for logging, plotting,
path management, and other common utilities.
"""

import os
import json
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device():
    """
    Get the available device (CPU or CUDA).
    
    Returns:
        torch.device: Device to use
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def create_directories(paths):
    """
    Create directories if they don't exist.
    
    Args:
        paths (list): List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Created {len(paths)} directories")


def load_config(config_path):
    """
    Load YAML configuration file.
    
    Args:
        config_path (str or Path): Path to config.yaml file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from {config_path}")
    return config


def save_config(config, save_path):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        save_path (str or Path): Path to save the config
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {save_path}")


def save_results(results, save_path):
    """
    Save results to JSON file.
    
    Args:
        results (dict): Results dictionary
        save_path (str or Path): Path to save the results
    """
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_path}")


def load_results(load_path):
    """
    Load results from JSON file.
    
    Args:
        load_path (str or Path): Path to load the results
        
    Returns:
        dict: Results dictionary
    """
    with open(load_path, 'r') as f:
        results = json.load(f)
    print(f"Results loaded from {load_path}")
    return results


def get_timestamp():
    """
    Get current timestamp as string.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_experiment_dir(base_dir, experiment_name):
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir (str or Path): Base directory for experiments
        experiment_name (str): Name of the experiment
        
    Returns:
        Path: Created directory path
    """
    timestamp = get_timestamp()
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


def plot_metrics(history, save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        history (dict): Training history with losses and accuracies
        save_path (str or Path, optional): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.get('train_losses', []), marker='o', label='Train Loss', color='blue')
    axes[0].plot(history.get('val_losses', []), marker='s', label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history.get('train_accs', []), marker='o', label='Train Accuracy', color='blue')
    axes[1].plot(history.get('val_accs', []), marker='s', label='Val Accuracy', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_size=(3, 224, 224)):
    """
    Print model summary including parameter count.
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input size for the model
    """
    print("Model Summary")
    print("=" * 40)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Input size: {input_size}")
    print("=" * 40)


def setup_logging(log_dir):
    """
    Setup logging configuration.
    
    Args:
        log_dir (str or Path): Directory for log files
        
    Returns:
        str: Log file path
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"log_{get_timestamp()}.txt"
    print(f"Logging to {log_file}")
    return str(log_file)


def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path: Project root path
    """
    return Path(__file__).parent.parent


if __name__ == "__main__":
    # Test utilities
    print("Utility functions loaded successfully!")
    print(f"Project root: {get_project_root()}")
    print(f"Timestamp: {get_timestamp()}")