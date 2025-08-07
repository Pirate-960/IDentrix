# precompute_gallery.py
"""
Pre-computes and saves gallery embeddings for the IDentrix Streamlit app.

This script addresses a critical performance bottleneck by separating the slow,
one-time task of embedding generation from the interactive application startup.
It loads the final trained model, iterates through all images in the specified
gallery directory, computes an embedding for each, and saves the entire
collection of embeddings as a single NumPy array file.

Workflow:
1.  Train your model using `train.py`.
2.  Ensure the best model is saved as `checkpoints/best_model.pth`.
3.  Run this script once: `python precompute_gallery.py`.
4.  Run the Streamlit app: `streamlit run app.py`.

The app will now load the pre-computed embeddings, reducing its startup
time from over an hour to a few seconds.
"""
import os
import torch
import numpy as np
import logging
from pathlib import Path

from model import DualBackboneNet
from utils import get_embeddings

# --- Logging Configuration ---
# Configure logging to provide clear, timestamped status updates.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def run_precomputation():
    """
    Main function to orchestrate the embedding generation and saving process.
    """
    # --- Configuration ---
    # Define paths for the model, data, and output file.
    MODEL_PATH = Path("checkpoints/best_model.pth")
    GALLERY_DIR = Path("data/gallery")
    OUTPUT_FILE = Path("data/gallery_embeddings.npy")

    # --- Device Setup ---
    # Automatically select CUDA if available for significantly faster processing.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # --- Model Loading ---
    logging.info(f"Loading model from '{MODEL_PATH}'...")
    if not MODEL_PATH.is_file():
        logging.error(f"Model file not found. Please ensure '{MODEL_PATH}' exists.")
        return

    model = DualBackboneNet()
    try:
        # Load the model's learned weights.
        # `map_location` ensures the model can be loaded regardless of the device
        # it was trained on. `weights_only=True` is a security measure.
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.to(device)  # Move the model to the selected device (GPU or CPU)
        model.eval()      # Set the model to evaluation mode
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model state dict: {e}")
        return

    # --- Gallery Path Loading ---
    logging.info(f"Scanning for images in '{GALLERY_DIR}'...")
    if not GALLERY_DIR.is_dir():
        logging.error(f"Gallery directory not found. Please ensure '{GALLERY_DIR}' exists.")
        return

    # Create a sorted list of all .jpg image paths. Sorting is critical to ensure
    # that the order of the saved embeddings matches the order of the paths that are loaded by the app.
    gallery_paths = sorted([str(p) for p in GALLERY_DIR.glob("*.jpg")])

    if not gallery_paths:
        logging.error(f"No .jpg images found in '{GALLERY_DIR}'.")
        return
    logging.info(f"Found {len(gallery_paths)} images to process.")

    # --- Embedding Generation (The Slow Step) ---
    logging.info("Starting embedding generation. This may take a significant amount of time...")
    gallery_embs = get_embeddings(model, gallery_paths)
    logging.info(f"Successfully generated {gallery_embs.shape[0]} embeddings.")

    # --- Save Embeddings ---
    # Create the parent directory for the output file if it doesn't exist.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the NumPy array to a binary file (.npy).
    np.save(OUTPUT_FILE, gallery_embs)
    logging.info(f"✅ Gallery embeddings have been successfully pre-computed and saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    run_precomputation()