"""
Data loading module for BarBeR-Project.

This module contains the BarcodeDataset class and utility functions
for loading and preprocessing data for both classification and detection tasks.
"""

import os
import json
import random
from pathlib import Path
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import yaml
import numpy as np

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BarcodeDataset(Dataset):
    """
    Custom Dataset for barcode images with handling for corrupted files.
    
    Args:
        image_dir (str or Path): Directory containing images
        label_dir (str or Path): Directory containing YOLO format labels
        transform (callable, optional): Transformations to apply to images
    """
    
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = [f for f in self.image_dir.glob("*.jpg")]
        self.image_files.sort()
        
        # Filter images that have corresponding label files and can be opened
        self.valid_images = []
        for img_path in self.image_files:
            label_path = self.label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                # Try to open image to verify it's valid
                try:
                    with Image.open(img_path) as test_img:
                        test_img.verify()
                    self.valid_images.append(img_path)
                except Exception:
                    # Skip corrupted images
                    continue
        
        print(f"Found {len(self.valid_images)} valid images with labels")
    
    def __len__(self):
        return len(self.valid_images)
    
    def __getitem__(self, idx):
        """
        Get image and its label.
        
        Returns:
            image: Transformed image tensor
            label: Class label (0 for 1D, 1 for 2D)
        """
        # Load image
        img_path = self.valid_images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # If image fails to load, return a blank image
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Load label from YOLO format
        label_path = self.label_dir / f"{img_path.stem}.txt"
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # For classification, use the class of the first barcode
        first_line = lines[0].strip().split()
        label = int(first_line[0])
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms():
    """
    Get standard transforms for training and validation.
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform


def load_split_files(data_path):
    """
    Load train/val/test splits from text files.
    
    Args:
        data_path (Path): Path to dataset directory
        
    Returns:
        tuple: (train_filenames, val_filenames, test_filenames)
    """
    split_names = ['train', 'val', 'test']
    split_files = {}
    
    for split in split_names:
        split_path = data_path / f"{split}.txt"
        if split_path.exists():
            with open(split_path, 'r') as f:
                split_files[split] = [line.strip() for line in f.readlines()]
        else:
            split_files[split] = []
    
    return split_files['train'], split_files['val'], split_files['test']


def create_dataloaders(data_path, batch_size=32, num_workers=0):
    """
    Create train, val, and test DataLoaders.
    
    Args:
        data_path (Path): Path to dataset directory
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of worker processes
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create dataset
    dataset = BarcodeDataset(
        image_dir=data_path / "images",
        label_dir=data_path / "labels",
        transform=val_transform
    )
    
    # Load splits
    train_files, val_files, test_files = load_split_files(data_path)
    
    # Get indices for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    for idx, img_path in enumerate(dataset.valid_images):
        if img_path.name in train_files:
            train_indices.append(idx)
        elif img_path.name in val_files:
            val_indices.append(idx)
        elif img_path.name in test_files:
            test_indices.append(idx)
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    return train_loader, val_loader, test_loader


def get_dataloaders_from_config(config_path):
    """
    Create DataLoaders using configuration file.
    
    Args:
        config_path (str or Path): Path to config.yaml file
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    training_config = config['training']
    
    # Construct data path
    project_root = Path.cwd().parent
    data_path = project_root / data_config['dataset_path']
    
    return create_dataloaders(
        data_path=data_path,
        batch_size=training_config['batch_size'],
        num_workers=training_config.get('num_workers', 0)
    )


if __name__ == "__main__":
    # Test the data loader
    project_root = Path.cwd().parent
    data_path = project_root / "data" / "raw" / "dataset"
    
    print("Testing data loader...")
    train_loader, val_loader, test_loader = create_dataloaders(data_path)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")