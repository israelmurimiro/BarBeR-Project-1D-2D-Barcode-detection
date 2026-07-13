"""
Dataset download script for BarBeR-Project.

This script provides utilities to download the BarBeR dataset
from the official website.
"""

import os
import sys
import webbrowser
from pathlib import Path
import requests
from tqdm import tqdm


# Dataset information
DATASET_URL = "https://ditto.ing.unimore.it/barber/"
DATASET_NAME = "BarBeR Dataset"
EXPECTED_SIZE = "~5 GB"  # approximate


def open_download_page():
    """
    Open the BarBeR dataset download page in the default browser.
    """
    print(f"Opening download page for {DATASET_NAME}...")
    webbrowser.open(DATASET_URL)
    print(f"Please download the dataset from the website.")
    print(f"Expected size: {EXPECTED_SIZE}")
    print("After downloading, extract the ZIP file and place the contents in:")
    print("  data/raw/")
    print("The extracted folder should contain 'Annotations/' and 'dataset/' directories.")
    return DATASET_URL


def check_dataset(data_path):
    """
    Check if the dataset is already present at the given path.
    
    Args:
        data_path (str or Path): Path to check for dataset
        
    Returns:
        bool: True if dataset is present, False otherwise
    """
    data_path = Path(data_path)
    annotations_path = data_path / "Annotations" / "VIA"
    dataset_path = data_path / "dataset"
    
    if annotations_path.exists() and dataset_path.exists():
        print("✅ Dataset already present!")
        print(f"  Annotations: {annotations_path}")
        print(f"  Dataset: {dataset_path}")
        return True
    else:
        print("❌ Dataset not found at the expected location.")
        return False


def get_dataset_info():
    """
    Print dataset information and download instructions.
    """
    print("=" * 60)
    print(f"{DATASET_NAME} Download Instructions")
    print("=" * 60)
    print(f"Download URL: {DATASET_URL}")
    print(f"Expected Size: {EXPECTED_SIZE}")
    print("")
    print("Steps:")
    print("1. Visit the URL above")
    print("2. Download the dataset (ZIP file)")
    print("3. Extract the ZIP file")
    print("4. Place the extracted contents in 'data/raw/'")
    print("   - You should have 'Annotations/' and 'dataset/' folders")
    print("")
    print("Example structure:")
    print("  BarBeR-Project/")
    print("  └── data/")
    print("      └── raw/")
    print("          ├── Annotations/")
    print("          │   └── VIA/")
    print("          │       ├── Artelab.json")
    print("          │       ├── Artelab Extended.json")
    print("          │       └── ... (other JSON files)")
    print("          └── dataset/")
    print("              ├── data.yaml")
    print("              ├── images/")
    print("              │   └── *.jpg")
    print("              └── labels/")
    print("                  └── *.txt")
    print("=" * 60)


def download_dataset(data_path=None, auto_open=True):
    """
    Main function to download the dataset.
    
    Args:
        data_path (str or Path, optional): Path to data directory
        auto_open (bool): Whether to automatically open the download page
    """
    # Determine data path
    if data_path is None:
        # Try to find project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        data_path = project_root / "data" / "raw"
    else:
        data_path = Path(data_path)
    
    print(f"Checking dataset at: {data_path}")
    print("-" * 40)
    
    if check_dataset(data_path):
        print("Dataset already present. No action needed.")
        return
    
    # Print instructions
    get_dataset_info()
    
    if auto_open:
        # Ask user if they want to open the download page
        response = input("\nDo you want to open the download page in your browser? (y/n): ").strip().lower()
        if response == 'y' or response == 'yes':
            open_download_page()
    
    print("\nAfter downloading, place the extracted files in the location shown above.")
    print("Then run this script again to verify the dataset is present.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download BarBeR dataset")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to data directory (default: ../data/raw)")
    parser.add_argument("--no_open", action="store_true",
                        help="Do not automatically open the download page")
    args = parser.parse_args()
    
    download_dataset(data_path=args.data_path, auto_open=not args.no_open)