# app.py
"""
Streamlit web application for interactive person re-identification.

This script provides a user-friendly web interface for the IDentrix model.
It allows a user to upload a query image and visualizes the top-k most similar
individuals retrieved from a large gallery, along with their similarity scores.

================================================================================
                            --- EXECUTION PATH ---
================================================================================
This project requires a specific sequence of steps to run correctly.
Follow this path from a clean state:

1.  SETUP THE ENVIRONMENT:
    - Ensure Python, Kaggle API, and W&B are configured.
    - Create a virtual environment and activate it.
    - Run: pip install -r requirements.txt
    - Run: wandb login

2.  PREPARE THE DATASET:
    - This script downloads the Market-1501 dataset from Kaggle.
    - Run: python prepare_dataset.py

3.  TRAIN THE MODEL:
    - This trains the DualBackboneNet and saves checkpoints.
    - Run: python train.py

4.  PRE-COMPUTE GALLERY EMBEDDINGS (CRITICAL FOR PERFORMANCE):
    - This step is essential to avoid a >1 hour startup time for the app.
    - It generates embeddings for all gallery images just once.
    - Ensure 'checkpoints/best_model.pth' exists from the previous step.
    - Run: python precompute_gallery.py

5.  RUN THE APPLICATION:
    - This command starts the interactive web server.
    - Run: streamlit run app.py
================================================================================
"""

# --- Standard Library Imports ---
import os
import logging
from unittest import result

# --- Third-Party Imports ---
import streamlit as st
import torch
import numpy as np
from PIL import Image

# --- Local Application Imports ---
from model import DualBackboneNet
from utils import preprocess_image, match_topk

# --- Logging Configuration ---
# Set up a basic logger to output informational messages to the console,
# helping to trace the application's execution flow and debug issues.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ------------------------
# Load Model and Gallery
# ------------------------

@st.cache_resource
def load_model():
    """Loads the pre-trained DualBackboneNet model into memory.

    This function is decorated with `@st.cache_resource`, a Streamlit feature
    that caches "heavy" objects like ML models. The model is loaded from disk
    only on the first run and is then reused from memory for all subsequent
    users and sessions, making the app feel much faster.

    Returns:
        Optional[DualBackboneNet]: The loaded PyTorch model in evaluation mode,
                                   or None if the model file is not found.
    """
    logging.info("Attempting to load the model...")
    model_path = "checkpoints/best_model.pth"

    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please ensure the model has been trained and placed correctly.")
        logging.error(f"Model file not found at '{model_path}'.")
        return None

    model = DualBackboneNet()
    # Load the learned parameters into the model architecture.
    # `map_location='cpu'` ensures the app works on machines without a GPU.
    # `weights_only=True` is a security best practice.
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    
    # Set the model to evaluation mode. This disables layers like Dropout
    # and BatchNorm updates, which is crucial for consistent inference results.
    model.eval()
    
    logging.info("Model loaded successfully and cached.")
    return model


@st.cache_data
def load_gallery_assets():
    """Loads gallery image paths and their pre-computed embeddings.

    Decorated with `@st.cache_data`, which caches serializable data objects
    like lists and NumPy arrays. This function loads the file paths for all
    gallery images and the corresponding embeddings from the pre-computed .npy
    file. This is the key to the app's fast startup.

    Returns:
        Tuple[List[str], np.ndarray]: A tuple containing:
            - A sorted list of file paths for all gallery images.
            - A NumPy array of gallery embeddings, where each row corresponds
              to the image at the same index in the paths list.
    """
    gallery_dir = './data/gallery'
    embeddings_path = './data/gallery_embeddings.npy'

    logging.info(f"Loading gallery assets from '{gallery_dir}' and '{embeddings_path}'...")

    # Validate that the necessary files and directories exist.
    if not os.path.exists(embeddings_path):
        st.error(f"Embeddings file not found: '{embeddings_path}'")
        st.warning("Please run `python precompute_gallery.py` to generate it.", icon="⚠️")
        logging.error("Embeddings file not found.")
        return [], np.array([]) # Return empty objects on failure

    # Load the image paths and sort them alphabetically. This is CRITICAL to
    # ensure the paths correctly align with the pre-computed embeddings.
    gallery_paths = sorted([os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.endswith('.jpg')])
    
    # Load the embeddings from the binary file. This is extremely fast.
    gallery_embs = np.load(embeddings_path)

    # Perform a sanity check to catch potential data mismatches.
    if len(gallery_paths) != len(gallery_embs):
        st.error("Mismatch between image count and embedding count. The gallery may be out of sync.")
        st.warning("Please re-run `python precompute_gallery.py`.", icon="🔄")
        return [], np.array([])

    logging.info(f"Gallery assets loaded and cached: {len(gallery_paths)} items.")
    return gallery_paths, gallery_embs


# --- Main Application Logic ---

def main():
    """The main function that orchestrates the Streamlit application UI and logic."""
    
    st.set_page_config(page_title="IDentrix Re-ID", layout="wide")
    st.title("🔍 Person Re-Identification Demo")

    # --- Load Resources ---
    # Display a spinner to the user during the initial resource loading.
    with st.spinner("Warming up the engine..."):
        model = load_model()
        gallery_paths, gallery_embs = load_gallery_assets()

    # If loading fails, stop the app execution gracefully.
    if model is None or gallery_embs.size == 0:
        st.stop()

    # --- User Input Section ---
    st.subheader("1. Upload a Query Image")
    uploaded_file = st.file_uploader(
        "Choose an image of a person to search for in the gallery.",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        query_image = Image.open(uploaded_file).convert('RGB')

        # --- Display Query and Results ---
        col1, col2 = st.columns([1, 3]) # Create two columns for layout

        with col1:
            st.subheader("2. Your Query")
            st.image(query_image, caption="Uploaded Query Image", use_container_width=True)

        with col2:
            st.subheader("3. Top 5 Matches Found in Gallery")
            
            # --- Inference and Matching ---
            with st.spinner("Analyzing image and searching for matches..."):
                # Use a context manager to ensure no gradients are computed,
                # which saves memory and speeds up inference.
                with torch.no_grad():
                    query_tensor = preprocess_image(query_image)
                    query_embedding = model(query_tensor).detach().numpy()[0]

                top_paths, top_scores = match_topk(
                    query_embedding, gallery_embs, gallery_paths, k=5
                )

            # --- Display Results Grid ---
            # Create 5 columns to display the top 5 matches side-by-side.
            result_cols = st.columns(5)
            for i in range(5):
                with result_cols[i]:
                    st.image(top_paths[i], use_container_width=True)
                    # Display the similarity score as a caption. The `delta`
                    # provides a subtle visual indicator (green for high scores).
                    st.metric(label="Similarity", value=f"{top_scores[i]:.3f}", delta=f"Rank {i+1}")

if __name__ == "__main__":
    # Start the Streamlit app by calling the main function.
    # This is the entry point for the application.
    logging.info("Starting Streamlit app...!")
    st.set_page_config(page_title="IDentrix Re-ID", layout="wide")
    logging.info("Streamlit app started.")
    main()