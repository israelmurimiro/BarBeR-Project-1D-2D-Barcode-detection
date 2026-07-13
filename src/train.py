"""
Training module for BarBeR-Project.

This module contains functions for training and evaluating
the classification models on the BarBeR dataset.
"""

import os
import time
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train
        dataloader (DataLoader): Training DataLoader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to use
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model (nn.Module): The model to evaluate
        dataloader (DataLoader): DataLoader for evaluation
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, save_dir=None):
    """
    Train the model for multiple epochs.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training DataLoader
        val_loader (DataLoader): Validation DataLoader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of epochs
        device (torch.device): Device to use
        save_dir (str or Path, optional): Directory to save model checkpoints
        
    Returns:
        dict: Training history (losses and accuracies)
    """
    # Initialize history
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': []
    }
    
    # Create save directory if provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting training...")
    print("=" * 60)
    
    start_time = time.time()
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Save history
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if save_dir and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_model.pth")
            print(f"  ✅ Best model saved (Val Acc: {val_acc:.2f}%)")
    
    end_time = time.time()
    print("=" * 60)
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")
    
    # Save final model
    if save_dir:
        torch.save(model.state_dict(), save_dir / "final_model.pth")
        print(f"Final model saved to {save_dir / 'final_model.pth'}")
    
    return history


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss and accuracy curves.
    
    Args:
        history (dict): Training history from train_model
        save_path (str or Path, optional): Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_losses'], marker='o', label='Train Loss', color='blue')
    axes[0].plot(history['val_losses'], marker='s', label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_accs'], marker='o', label='Train Accuracy', color='blue')
    axes[1].plot(history['val_accs'], marker='s', label='Val Accuracy', color='orange')
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


def train_from_config(config_path, model, train_loader, val_loader):
    """
    Train a model using configuration file.
    
    Args:
        config_path (str or Path): Path to config.yaml file
        model (nn.Module): Model to train
        train_loader (DataLoader): Training DataLoader
        val_loader (DataLoader): Validation DataLoader
        
    Returns:
        dict: Training history
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config['training']
    device = torch.device(training_config['device'])
    
    # Move model to device
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.0005)
    )
    
    # Save directory
    project_root = Path.cwd().parent
    save_dir = project_root / "experiments" / "baseline_cnn" / "checkpoints"
    
    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=training_config['num_epochs'],
        device=device,
        save_dir=save_dir
    )
    
    return history


if __name__ == "__main__":
    # Test training module
    print("Training module loaded successfully!")
    print("Usage: from src.train import train_model, evaluate, plot_training_history")