"""
Prediction module for BarBeR-Project.

This module contains functions for running inference on new images
using trained classification and detection models.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from ultralytics import YOLO
import yaml


def load_classification_model(model_path, model_class, device='cpu'):
    """
    Load a trained classification model.
    
    Args:
        model_path (str or Path): Path to model weights (.pth file)
        model_class (nn.Module): Model class to instantiate
        device (str): Device to use ('cpu' or 'cuda')
        
    Returns:
        nn.Module: Loaded model
    """
    device = torch.device(device)
    model = model_class(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def predict_image_classification(model, image_path, transform=None, device='cpu'):
    """
    Predict class of a single image using classification model.
    
    Args:
        model (nn.Module): Trained classification model
        image_path (str or Path): Path to image file
        transform (callable, optional): Transformations to apply
        device (str): Device to use
        
    Returns:
        tuple: (predicted_class, class_name, confidence)
    """
    device = torch.device(device)
    
    # Load and preprocess image
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ['1D', '2D']
    predicted_class = predicted.item()
    class_name = class_names[predicted_class]
    confidence_score = confidence.item()
    
    return predicted_class, class_name, confidence_score


def predict_batch_classification(model, image_paths, transform=None, device='cpu'):
    """
    Predict classes for multiple images.
    
    Args:
        model (nn.Module): Trained classification model
        image_paths (list): List of image paths
        transform (callable, optional): Transformations to apply
        device (str): Device to use
        
    Returns:
        list: List of predictions (class, class_name, confidence)
    """
    results = []
    for image_path in image_paths:
        pred_class, class_name, confidence = predict_image_classification(
            model, image_path, transform, device
        )
        results.append({
            'image': image_path,
            'class': pred_class,
            'class_name': class_name,
            'confidence': confidence
        })
    return results


def predict_detection(model, image_path, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run detection inference on a single image using YOLO model.
    
    Args:
        model (YOLO): Trained YOLO model
        image_path (str or Path): Path to image file
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        
    Returns:
        results: YOLO prediction results
    """
    results = model(image_path, conf=conf_threshold, iou=iou_threshold)
    return results


def predict_detection_batch(model, image_paths, conf_threshold=0.25, iou_threshold=0.45):
    """
    Run detection inference on multiple images.
    
    Args:
        model (YOLO): Trained YOLO model
        image_paths (list): List of image paths
        conf_threshold (float): Confidence threshold
        iou_threshold (float): IoU threshold for NMS
        
    Returns:
        list: List of prediction results
    """
    results = []
    for image_path in image_paths:
        result = predict_detection(model, image_path, conf_threshold, iou_threshold)
        results.append(result)
    return results


def visualize_predictions(results, image_path, save_path=None, class_names=['1D', '2D']):
    """
    Visualize detection predictions with bounding boxes.
    
    Args:
        results: YOLO prediction results
        image_path (str or Path): Path to image file
        save_path (str or Path, optional): Path to save the visualization
        class_names (list): List of class names
        
    Returns:
        numpy.ndarray: Image with bounding boxes drawn
    """
    # Load image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Draw bounding boxes and labels
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = f"{class_names[cls]} {conf:.2f}"
        
        cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_rgb, label, (int(x1), int(y1)-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"Detected: {len(results[0].boxes)} barcodes")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return img_rgb


def load_yolo_model(model_path):
    """
    Load a trained YOLO detection model.
    
    Args:
        model_path (str or Path): Path to model weights (.pt file)
        
    Returns:
        YOLO: Loaded YOLO model
    """
    model = YOLO(str(model_path))
    print(f"YOLO model loaded from {model_path}")
    return model


def predict_from_config(config_path, model_type='classification', image_path=None, image_dir=None):
    """
    Run prediction using configuration file.
    
    Args:
        config_path (str or Path): Path to config.yaml file
        model_type (str): 'classification' or 'detection'
        image_path (str or Path, optional): Single image path
        image_dir (str or Path, optional): Directory of images
        
    Returns:
        predictions: Prediction results
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    project_root = Path.cwd().parent
    
    if model_type == 'classification':
        # Load model
        model_path = project_root / "experiments" / "baseline_cnn" / "checkpoints" / "best_model.pth"
        from src.models import BasicCNN
        device = config['training']['device']
        model = load_classification_model(model_path, BasicCNN, device)
        
        # Run prediction
        if image_path:
            return predict_image_classification(model, image_path, device=device)
        elif image_dir:
            image_paths = list(Path(image_dir).glob("*.jpg"))
            return predict_batch_classification(model, image_paths, device=device)
    
    elif model_type == 'detection':
        # Load YOLO model
        model_path = project_root / "experiments" / "yolo" / "checkpoints" / "best.pt"
        # If not in experiments, use the trained path from 02 notebook
        if not model_path.exists():
            model_path = Path("/Users/israelm/Desktop/DATA SCIENCE/runs/detect/train-3/weights/best.pt")
        model = load_yolo_model(model_path)
        
        # Run prediction
        if image_path:
            return predict_detection(model, image_path)
        elif image_dir:
            image_paths = list(Path(image_dir).glob("*.jpg"))
            return predict_detection_batch(model, image_paths)
    
    return None


if __name__ == "__main__":
    print("Prediction module loaded successfully!")
    print("Usage: from src.predict import predict_image_classification, predict_detection")