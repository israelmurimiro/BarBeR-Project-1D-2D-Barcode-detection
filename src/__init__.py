"""
BarBeR-Project Source Package.

This package contains all source code for the project including
data loading, model architectures, training, evaluation, and prediction.
"""

from .data_loader import BarcodeDataset, create_dataloaders, get_transforms
from .models import BasicCNN, create_mobilenetv2, create_efficientnet_b0, get_model
from .train import train_model, evaluate, train_one_epoch
from .evaluate import evaluate_model, generate_confusion_matrix, get_predictions
from .predict import predict_image_classification, predict_detection
from .utils import set_seed, get_device, load_config, count_parameters

__version__ = "1.0.0"