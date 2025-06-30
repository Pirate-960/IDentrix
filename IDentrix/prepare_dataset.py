# prepare_dataset.py

"""
Automatically downloads, extracts, and formats the Market-1501 dataset
for the IDentrix person Re-ID project using the Kaggle API.

Ensure you have:
1. Installed the kaggle package: pip install kaggle
2. Placed your kaggle.json in ~/.kaggle or configured the environment.
"""

import os
import shutil
import zipfile
import subprocess
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
ZIP_PATH = RAW_DIR / "market-1501.zip"
EXTRACTED_DIR = RAW_DIR / "Market-1501-v15.09.15"

GALLERY_DST = BASE_DIR / "data" / "gallery"
QUERY_DST = BASE_DIR / "data" / "query"

# Step 1: Download from Kaggle
def download_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("⬇️  Downloading Market-1501 from Kaggle...")

    # Run kaggle CLI to download the dataset
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "iiierie/market-1501-v15-09-15", "-p", str(RAW_DIR)],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("❌ Kaggle download failed. Make sure kaggle.json is configured properly.")

    # Find the downloaded zip
    for f in RAW_DIR.glob("*.zip"):
        f.rename(ZIP_PATH)
    print("✅ Download complete.")

# Step 2: Extract the ZIP file
def extract_zip():
    if EXTRACTED_DIR.exists():
        print("🗂️  Dataset already extracted.")
        return

    print("🧩 Extracting ZIP...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)
    print("✅ Extraction complete.")

# Step 3: Copy images
def copy_images(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    files = list(Path(src_folder).glob("*.jpg"))
    for f in files:
        shutil.copy2(f, dst_folder / f.name)
    print(f"📁 Copied {len(files)} images to {dst_folder.name}/")

# --- Main ---
if __name__ == "__main__":
    download_dataset()
    extract_zip()
    copy_images(EXTRACTED_DIR / "bounding_box_test", GALLERY_DST)
    copy_images(EXTRACTED_DIR / "query", QUERY_DST)
    print("\n🎉 Dataset is ready for training in IDentrix!")
