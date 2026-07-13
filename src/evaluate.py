"""
Evaluation module for BarBeR-Project.

This module contains functions for evaluating classification models,
including confusion matrix generation and classification reports.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate model on test set and return predictions and labels.
    
    Args:
        model (nn.Module): The model to evaluate
        test_loader (DataLoader): Test DataLoader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        
    Returns:
        tuple: (test_loss, test_accuracy, all_preds, all_labels)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    
    return test_loss, test_accuracy, all_preds, all_labels


def generate_confusion_matrix(all_labels, all_preds, class_names=None, save_path=None):
    """
    Generate and display confusion matrix.
    
    Args:
        all_labels (list): True labels
        all_preds (list): Predicted labels
        class_names (list, optional): Names of classes
        save_path (str or Path, optional): Path to save the plot
        
    Returns:
        numpy.ndarray: Confusion matrix
    """
    if class_names is None:
        class_names = ['1D', '2D']
    
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return cm


def generate_classification_report(all_labels, all_preds, class_names=None):
    """
    Generate and print classification report.
    
    Args:
        all_labels (list): True labels
        all_preds (list): Predicted labels
        class_names (list, optional): Names of classes
        
    Returns:
        str: Classification report as string
    """
    if class_names is None:
        class_names = ['1D', '2D']
    
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print("=" * 60)
    print(report)
    
    return report


def get_predictions(model, dataloader, device):
    """
    Get predictions and labels from a model.
    
    Args:
        model (nn.Module): The model to evaluate
        dataloader (DataLoader): DataLoader
        device (torch.device): Device to use
        
    Returns:
        tuple: (predictions, labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels


def evaluate_model_from_config(config_path, model, test_loader):
    """
    Evaluate model using configuration file.
    
    Args:
        config_path (str or Path): Path to config.yaml file
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test DataLoader
        
    Returns:
        tuple: (test_loss, test_accuracy, history)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config['training']
    device = torch.device(training_config['device'])
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Evaluate
    test_loss, test_accuracy, all_preds, all_labels = evaluate_model(
        model, test_loader, criterion, device
    )
    
    # Generate confusion matrix
    project_root = Path.cwd().parent
    save_dir = project_root / "outputs" / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    generate_confusion_matrix(
        all_labels, all_preds, 
        class_names=['1D', '2D'],
        save_path=save_dir / "confusion_matrix.png"
    )
    
    generate_classification_report(all_labels, all_preds, class_names=['1D', '2D'])
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    return test_loss, test_accuracy, all_preds, all_labels


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("Usage: from src.evaluate import evaluate_model, generate_confusion_matrix")