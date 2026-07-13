"""
Data preprocessing script for BarBeR-Project.

This script converts VIA JSON annotations to YOLO format
and creates train/val/test splits for the dataset.
"""

import os
import json
import random
import shutil
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
import yaml


def load_via_annotations(annotations_dir):
    """
    Load all VIA JSON annotation files.
    
    Args:
        annotations_dir (str or Path): Directory containing VIA JSON files
        
    Returns:
        list: List of (annotation_file, data) tuples
    """
    annotations_dir = Path(annotations_dir)
    annotation_data = []
    
    for ann_file in annotations_dir.glob("*.json"):
        with open(ann_file, 'r') as f:
            data = json.load(f)
        annotation_data.append((ann_file, data))
    
    print(f"Loaded {len(annotation_data)} annotation files")
    return annotation_data


def polygon_to_yolo(points, img_width, img_height):
    """
    Convert polygon points to YOLO format (normalized center + width + height).
    
    Args:
        points (list): List of (x, y) points
        img_width (int): Image width in pixels
        img_height (int): Image height in pixels
        
    Returns:
        tuple: (x_center, y_center, width, height) normalized to 0-1
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    x_center = x_min + bbox_width / 2
    y_center = y_min + bbox_height / 2
    
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = bbox_width / img_width
    height_norm = bbox_height / img_height
    
    # Clip values to ensure they are within [0, 1]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))
    
    return x_center_norm, y_center_norm, width_norm, height_norm


def convert_annotations_to_yolo(annotation_data, images_dir, labels_dir, skip_corrupted=True):
    """
    Convert VIA annotations to YOLO format.
    
    Args:
        annotation_data (list): List of (annotation_file, data) tuples
        images_dir (str or Path): Directory containing images
        labels_dir (str or Path): Output directory for YOLO labels
        skip_corrupted (bool): Whether to skip corrupted images
        
    Returns:
        tuple: (converted_count, total_annotations)
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    total_annotations = 0
    skipped_images = []
    
    print("Converting annotations to YOLO format...")
    
    for ann_file, data in tqdm(annotation_data, desc="Processing annotation files"):
        for img_key, img_data in data['_via_img_metadata'].items():
            filename = img_data['filename']
            image_path = images_dir / filename
            
            # Check if image exists
            if not image_path.exists():
                skipped_images.append(filename)
                continue
            
            # Get image dimensions
            try:
                img = cv2.imread(str(image_path))
                if img is None:
                    skipped_images.append(filename)
                    continue
                img_height, img_width = img.shape[:2]
            except Exception:
                skipped_images.append(filename)
                continue
            
            # Convert each region
            yolo_labels = []
            for region in img_data['regions']:
                shape = region['shape_attributes']
                if shape['name'] != 'polygon':
                    continue
                
                points = list(zip(shape['all_points_x'], shape['all_points_y']))
                
                # Determine class (1 for 2D, 0 for 1D)
                btype = region['region_attributes'].get('Type', '')
                is_2d = btype in ['QR', 'DATAMATRIX', 'PDF417', 'AZTEC']
                class_id = 1 if is_2d else 0
                
                # Convert to YOLO format
                x_c, y_c, w, h = polygon_to_yolo(points, img_width, img_height)
                yolo_labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                total_annotations += 1
            
            # Save YOLO labels
            if yolo_labels:
                label_filename = labels_dir / f"{filename.rsplit('.', 1)[0]}.txt"
                with open(label_filename, 'w') as f:
                    f.write("\n".join(yolo_labels))
                converted_count += 1
    
    if skipped_images:
        print(f"Warning: {len(skipped_images)} images were skipped (corrupted or missing)")
    
    return converted_count, total_annotations


def create_splits(all_images, output_dir, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Create train/val/test splits.
    
    Args:
        all_images (list): List of image filenames
        output_dir (str or Path): Output directory for split files
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_files, val_files, test_files)
    """
    output_dir = Path(output_dir)
    random.seed(seed)
    
    shuffled = all_images.copy()
    random.shuffle(shuffled)
    
    total = len(shuffled)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    
    train_files = shuffled[:train_size]
    val_files = shuffled[train_size:train_size + val_size]
    test_files = shuffled[train_size + val_size:]
    
    # Save split files
    split_names = ['train', 'val', 'test']
    split_files = [train_files, val_files, test_files]
    
    for name, files in zip(split_names, split_files):
        with open(output_dir / f"{name}.txt", 'w') as f:
            for filename in files:
                f.write(f"{filename}\n")
        print(f"{name.capitalize()}: {len(files)} images")
    
    return train_files, val_files, test_files


def create_data_yaml(data_path, train_path, val_path, test_path=None):
    """
    Create data.yaml file for YOLO.
    
    Args:
        data_path (str or Path): Base path for dataset
        train_path (str): Path to training images (relative to data_path)
        val_path (str): Path to validation images (relative to data_path)
        test_path (str, optional): Path to test images (relative to data_path)
    """
    data_config = {
        'path': str(data_path),
        'train': train_path,
        'val': val_path,
        'nc': 2,
        'names': ['1D', '2D']
    }
    
    if test_path:
        data_config['test'] = test_path
    
    with open(data_path / "data.yaml", 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"data.yaml created at {data_path / 'data.yaml'}")
    return data_config


def preprocess_dataset(data_root, annotations_dir=None, images_dir=None, labels_dir=None,
                       train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    Main preprocessing function.
    
    Args:
        data_root (str or Path): Root directory for data
        annotations_dir (str or Path): Directory containing VIA annotations
        images_dir (str or Path): Directory containing images
        labels_dir (str or Path): Output directory for YOLO labels
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        seed (int): Random seed
        
    Returns:
        dict: Preprocessing statistics
    """
    data_root = Path(data_root)
    
    # Set up directories
    if annotations_dir is None:
        annotations_dir = data_root / "Annotations" / "VIA"
    else:
        annotations_dir = Path(annotations_dir)
    
    if images_dir is None:
        images_dir = data_root / "dataset" / "images"
    else:
        images_dir = Path(images_dir)
    
    if labels_dir is None:
        labels_dir = data_root / "dataset" / "labels"
    else:
        labels_dir = Path(labels_dir)
    
    print(f"Data root: {data_root}")
    print(f"Annotations dir: {annotations_dir}")
    print(f"Images dir: {images_dir}")
    print(f"Labels dir: {labels_dir}")
    
    # Load annotations
    annotation_data = load_via_annotations(annotations_dir)
    
    # Convert to YOLO
    converted_count, total_annotations = convert_annotations_to_yolo(
        annotation_data, images_dir, labels_dir
    )
    
    print(f"Converted {converted_count} images with {total_annotations} barcodes")
    
    # Get all image filenames with labels
    all_images = [f.name for f in labels_dir.glob("*.txt")]
    all_images = [f.replace('.txt', '.jpg') for f in all_images]
    
    # Create splits
    train_files, val_files, test_files = create_splits(
        all_images, data_root / "dataset", train_ratio, val_ratio, seed
    )
    
    # Create data.yaml
    data_config = create_data_yaml(
        data_root / "dataset",
        train_path="images/train",
        val_path="images/val"
    )
    
    return {
        'converted_images': converted_count,
        'total_annotations': total_annotations,
        'train_count': len(train_files),
        'val_count': len(val_files),
        'test_count': len(test_files)
    }


def reorganize_for_yolo(data_path):
    """
    Reorganize dataset into YOLO's expected structure (images/train/, images/val/, etc.)
    
    Args:
        data_path (str or Path): Path to dataset directory
    """
    data_path = Path(data_path)
    images_dir = data_path / "images"
    labels_dir = data_path / "labels"
    
    # Create train/val subdirectories
    for split in ['train', 'val']:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Read split files
    for split in ['train', 'val']:
        split_path = data_path / f"{split}.txt"
        if not split_path.exists():
            print(f"Warning: {split_path} not found")
            continue
        
        with open(split_path, 'r') as f:
            filenames = [line.strip() for line in f.readlines()]
        
        print(f"Moving {len(filenames)} files for {split}...")
        
        for filename in filenames:
            # Move image
            src_img = images_dir / filename
            dst_img = images_dir / split / filename
            if src_img.exists() and not dst_img.exists():
                shutil.move(str(src_img), str(dst_img))
            
            # Move label
            label_name = filename.rsplit('.', 1)[0] + '.txt'
            src_lbl = labels_dir / label_name
            dst_lbl = labels_dir / split / label_name
            if src_lbl.exists() and not dst_lbl.exists():
                shutil.move(str(src_lbl), str(dst_lbl))
    
    print("✅ Dataset reorganized for YOLO!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess BarBeR dataset")
    parser.add_argument("--data_root", type=str, default="data/raw",
                        help="Root directory for data")
    parser.add_argument("--train_ratio", type=float, default=0.70,
                        help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--reorganize", action="store_true",
                        help="Reorganize dataset into YOLO structure")
    args = parser.parse_args()
    
    # Get absolute path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_root = project_root / args.data_root
    
    # Preprocess
    stats = preprocess_dataset(
        data_root=data_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    print("\n" + "=" * 40)
    print("Preprocessing completed!")
    print(f"Converted images: {stats['converted_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Train: {stats['train_count']}")
    print(f"Val: {stats['val_count']}")
    print(f"Test: {stats['test_count']}")
    print("=" * 40)
    
    if args.reorganize:
        reorganize_for_yolo(data_root / "dataset")