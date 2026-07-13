"""
Unit tests for data loader module.

This module tests the BarcodeDataset class and data loading functions.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import BarcodeDataset, create_dataloaders, get_transforms


class TestBarcodeDataset(unittest.TestCase):
    """Test cases for BarcodeDataset class."""
    
    def setUp(self):
        """Set up test environment with temporary files."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.image_dir = Path(self.test_dir) / "images"
        self.label_dir = Path(self.test_dir) / "labels"
        self.image_dir.mkdir()
        self.label_dir.mkdir()
        
        # Create sample images and labels
        self.create_sample_data()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def create_sample_data(self):
        """Create sample images and label files for testing."""
        # Create 3 sample images
        for i in range(3):
            img = Image.new('RGB', (224, 224), color=(255, 0, 0))
            img.save(self.image_dir / f"test_{i}.jpg")
        
        # Create corresponding label files
        # test_0.jpg -> 1D barcode
        with open(self.label_dir / "test_0.txt", 'w') as f:
            f.write("0 0.5 0.5 0.3 0.1\n")
        
        # test_1.jpg -> 2D barcode
        with open(self.label_dir / "test_1.txt", 'w') as f:
            f.write("1 0.5 0.5 0.4 0.4\n")
        
        # test_2.jpg -> multiple barcodes (1D and 2D)
        with open(self.label_dir / "test_2.txt", 'w') as f:
            f.write("0 0.3 0.3 0.2 0.1\n")
            f.write("1 0.7 0.7 0.2 0.2\n")
    
    def test_dataset_initialization(self):
        """Test dataset initialization and filtering."""
        transform, _ = get_transforms()
        dataset = BarcodeDataset(
            image_dir=self.image_dir,
            label_dir=self.label_dir,
            transform=transform
        )
        
        # Should find all 3 valid images
        self.assertEqual(len(dataset), 3)
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        transform, _ = get_transforms()
        dataset = BarcodeDataset(
            image_dir=self.image_dir,
            label_dir=self.label_dir,
            transform=transform
        )
        
        # Test first sample (1D barcode)
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(label, 0)  # 1D barcode
        
        # Test second sample (2D barcode)
        image, label = dataset[1]
        self.assertEqual(label, 1)  # 2D barcode
    
    def test_missing_label_filtering(self):
        """Test that images without labels are filtered out."""
        # Create an image without a label
        img = Image.new('RGB', (224, 224), color=(0, 255, 0))
        img.save(self.image_dir / "no_label.jpg")
        
        transform, _ = get_transforms()
        dataset = BarcodeDataset(
            image_dir=self.image_dir,
            label_dir=self.label_dir,
            transform=transform
        )
        
        # Should still only have 3 valid images
        self.assertEqual(len(dataset), 3)
    
    def test_corrupted_image_handling(self):
        """Test handling of corrupted images."""
        # Create a corrupted image (empty file)
        with open(self.image_dir / "corrupted.jpg", 'w') as f:
            f.write("not an image")
        
        # Create corresponding label
        with open(self.label_dir / "corrupted.txt", 'w') as f:
            f.write("0 0.5 0.5 0.3 0.1\n")
        
        transform, _ = get_transforms()
        dataset = BarcodeDataset(
            image_dir=self.image_dir,
            label_dir=self.label_dir,
            transform=transform
        )
        
        # Corrupted image should be skipped
        self.assertEqual(len(dataset), 3)


class TestDataLoaders(unittest.TestCase):
    """Test cases for data loader creation."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir)
        self.image_dir = self.data_dir / "images"
        self.label_dir = self.data_dir / "labels"
        self.image_dir.mkdir()
        self.label_dir.mkdir()
        
        # Create sample data
        for i in range(10):
            img = Image.new('RGB', (224, 224), color=(255, 0, 0))
            img.save(self.image_dir / f"img_{i}.jpg")
            with open(self.label_dir / f"img_{i}.txt", 'w') as f:
                f.write("0 0.5 0.5 0.3 0.1\n")
        
        # Create split files
        train_files = [f"img_{i}.jpg" for i in range(0, 7)]
        val_files = [f"img_{i}.jpg" for i in range(7, 9)]
        test_files = [f"img_{i}.jpg" for i in range(9, 10)]
        
        with open(self.data_dir / "train.txt", 'w') as f:
            f.write("\n".join(train_files))
        with open(self.data_dir / "val.txt", 'w') as f:
            f.write("\n".join(val_files))
        with open(self.data_dir / "test.txt", 'w') as f:
            f.write("\n".join(test_files))
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)
    
    def test_create_dataloaders(self):
        """Test DataLoader creation."""
        train_loader, val_loader, test_loader = create_dataloaders(
            self.data_dir, batch_size=2
        )
        
        self.assertEqual(len(train_loader.dataset), 7)
        self.assertEqual(len(val_loader.dataset), 2)
        self.assertEqual(len(test_loader.dataset), 1)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()