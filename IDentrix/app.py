# app.py
"""
Streamlit app for interactive person re-identification.

This script launches a web-based user interface where users can:
- Upload a query image of a person.
- See the model retrieve the top-k most similar individuals from a pre-computed gallery.
- View the similarity scores for each match.

The application relies on a pre-trained model and a gallery of images, which are
loaded into memory for fast retrieval.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
from model import DualBackboneNet
from utils import preprocess_image, get_embeddings, match_topk
import os
import logging

# --- Basic logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ------------------------
# Load Model and Gallery
# ------------------------
st.title("🔍 Person Re-Identification Demo")
logging.info("Streamlit app started.")

@st.cache_resource
def load_model():
    """
    Loads the pre-trained DualBackboneNet model.
    - Uses Streamlit's caching to load the model only once.
    - Sets the model to evaluation mode.
    - Maps the model to the CPU for broader compatibility.

    Returns:
        DualBackboneNet: The loaded and initialized model.
    """
    logging.info("Loading model...")
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please train the model first.")
        logging.error(f"Model file not found at '{model_path}'.")
        return None
    
    model = DualBackboneNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    logging.info("Model loaded successfully.")
    return model

@st.cache_data
def load_gallery(_model):
    """
    Loads gallery images and computes their embeddings.
    - Uses Streamlit's data caching to avoid re-computing embeddings on every run.
    - Scans the specified gallery directory for .jpg files.
    - Uses the provided model to generate an embedding for each image.

    Args:
        _model (torch.nn.Module): The model to use for generating embeddings.

    Returns:
        tuple[list, np.ndarray]: A tuple containing:
            - A list of gallery image paths.
            - A NumPy array of gallery embeddings.
    """
    gallery_dir = './data/gallery'
    logging.info(f"Loading gallery from '{gallery_dir}'...")
    if not os.path.isdir(gallery_dir) or not os.listdir(gallery_dir):
        st.error(f"Gallery directory '{gallery_dir}' is empty or not found. Please prepare the dataset.")
        logging.error(f"Gallery directory '{gallery_dir}' is empty or not found.")
        return [], []

    gallery_paths = [os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.endswith('.jpg')]
    gallery_embs = get_embeddings(_model, gallery_paths)
    logging.info(f"Gallery loaded with {len(gallery_paths)} images.")
    return gallery_paths, gallery_embs

# Execute loading functions
with st.spinner("Warming up the engine... (Loading model and gallery)"):
    model = load_model()
    if model:
        gallery_paths, gallery_embs = load_gallery(model)
    else:
        # Stop the app if the model failed to load
        st.stop()


# ------------------------
# Main Application UI
# ------------------------
st.subheader("Upload a Query Image")
file = st.file_uploader("Choose an image of a person", type=['jpg', 'jpeg', 'png'])

if file:
    logging.info(f"User uploaded file: {file.name}")
    query_img = Image.open(file).convert('RGB')

    # --- Display Query Image ---
    st.image(query_img, caption="Query Image", width=256)

    # --- Generate Embedding for Query ---
    with st.spinner("Analyzing image..."):
        with torch.no_grad():
            # Preprocess the image and get its embedding
            query_tensor = preprocess_image(query_img)
            query_emb = model(query_tensor).detach().numpy()[0]
        logging.info("Generated embedding for the query image.")

    # --- Find and Display Matches ---
    with st.spinner("Searching for matches..."):
        top_paths, top_scores = match_topk(query_emb, gallery_embs, gallery_paths, k=5)
        logging.info(f"Found top 5 matches. Best score: {top_scores[0]:.4f}")

    st.markdown("### Top 5 Matches from Gallery")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(top_paths[i], caption=f"Score: {top_scores[i]:.2f}", use_column_width=True)