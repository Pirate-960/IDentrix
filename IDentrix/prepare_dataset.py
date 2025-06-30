# prepare_dataset.py
"""
This script automates the setup of the Market-1501 dataset for the project.

It performs the following steps:
1.  Downloads the dataset from Kaggle using the Kaggle API.
2.  Extracts the contents from the downloaded ZIP file.
3.  Organizes the images into the required `data/gallery` and `data/query`
    directories for training and evaluation.

Prerequisites:
- The `kaggle` Python package must be installed (`pip install kaggle`).
- A `kaggle.json` API token must be configured in its default location
  (e.g., `~/.kaggle/kaggle.json` on Linux/macOS or
  `C:\\Users\\<User>\\.kaggle\\kaggle.json` on Windows).
"""

import os
import shutil
import zipfile
import subprocess
from pathlib import Path
import logging

# --- Basic logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# --- Path Configuration ---
# Use pathlib for robust and OS-agnostic path handling.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
DATASET_NAME = "market-1501-v15-09-15"
ZIP_NAME = f"{DATASET_NAME}.zip"
ZIP_PATH = RAW_DIR / ZIP_NAME
EXTRACTED_DIR = RAW_DIR / "Market-1501-v15.09.15"

# Destination directories for the final dataset structure
GALLERY_DST = DATA_DIR / "gallery"
QUERY_DST = DATA_DIR / "query"

# --- Step 1: Download from Kaggle ---
def download_dataset():
    """Downloads the Market-1501 dataset from Kaggle if not already present."""
    os.makedirs(RAW_DIR, exist_ok=True)
    
    if ZIP_PATH.exists():
        logging.info(f"Dataset ZIP file already exists at '{ZIP_PATH}'. Skipping download.")
        return

    logging.info("⬇️  Downloading Market-1501 from Kaggle...")
    
    # Command to download the dataset using the Kaggle CLI
    # Format: kaggle datasets download -d <dataset-slug> -p <path> --unzip
    command = [
        "kaggle", "datasets", "download",
        "-d", "iiierie/market-1501-v15-09-15",
        "-p", str(RAW_DIR)
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logging.info(result.stdout)
        
        # The Kaggle API downloads with a different name, so we rename it.
        downloaded_zip = RAW_DIR / "market-1501-v15-09-15.zip"
        if downloaded_zip.exists():
            downloaded_zip.rename(ZIP_PATH)
            logging.info(f"✅ Download complete. File saved to '{ZIP_PATH}'.")
        else:
            raise FileNotFoundError("Kaggle CLI ran but the expected zip file was not found.")

    except subprocess.CalledProcessError as e:
        logging.error("❌ Kaggle download failed.")
        logging.error("--- Kaggle CLI Error Output ---")
        logging.error(e.stderr)
        logging.error("--- End of Error Output ---")
        logging.error("Please ensure the Kaggle API is installed and your `kaggle.json` token is configured correctly.")
        exit(1)
    except FileNotFoundError:
        logging.error("❌ 'kaggle' command not found.")
        logging.error("Please install the Kaggle API with: pip install kaggle")
        exit(1)


# --- Step 2: Extract the ZIP file ---
def extract_zip():
    """Extracts the dataset from the ZIP file if not already extracted."""
    if EXTRACTED_DIR.exists():
        logging.info(f"🗂️  Dataset already extracted at '{EXTRACTED_DIR}'. Skipping extraction.")
        return

    if not ZIP_PATH.exists():
        logging.error(f"ZIP file not found at '{ZIP_PATH}'. Cannot extract.")
        return

    logging.info(f"🧩 Extracting '{ZIP_PATH}'...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)
    logging.info(f"✅ Extraction complete. Data is in '{EXTRACTED_DIR}'.")

# --- Step 3: Copy and Organize Images ---
def copy_images(src_folder: Path, dst_folder: Path):
    """
    Copies all .jpg images from a source to a destination folder.

    Args:
        - src_folder (Path): The source directory containing images.
        - dst_folder (Path): The destination directory.
    """
    if not src_folder.exists():
        logging.warning(f"Source folder '{src_folder}' not found. Cannot copy images.")
        return
        
    os.makedirs(dst_folder, exist_ok=True)
    
    image_files = list(src_folder.glob("*.jpg"))
    if not image_files:
        logging.warning(f"No .jpg files found in '{src_folder}'.")
        return

    logging.info(f"Copying {len(image_files)} images from '{src_folder.name}' to '{dst_folder.name}/'...")
    for f in image_files:
        # shutil.copy2 preserves metadata
        shutil.copy2(f, dst_folder / f.name)
    logging.info(f"📁 Copied {len(image_files)} images to '{dst_folder.name}/'.")

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Dataset Preparation ---")
    
    # Execute each step in order
    download_dataset()
    extract_zip()

    # Define source folders from the extracted Market-1501 dataset
    gallery_src = EXTRACTED_DIR / "bounding_box_test"
    query_src = EXTRACTED_DIR / "query"
    
    # Organize into final project structure
    copy_images(gallery_src, GALLERY_DST)
    copy_images(query_src, QUERY_DST)
    
    logging.info("🎉 Dataset is ready for use in the project!")