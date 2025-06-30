# app.py
"""
Streamlit app for interactive person re-identification.
Users can upload a query image and retrieve the top-k most similar individuals
from the gallery using the trained model.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
from model import DualBackboneNet
from utils import preprocess_image, get_embeddings, match_topk
import os

# ------------------------
# Load model and gallery
# ------------------------
st.title("🔍 Person Re-Identification Demo")

@st.cache_resource
def load_model():
    model = DualBackboneNet()
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location='cpu'))
    model.eval()
    return model

@st.cache_data
def load_gallery():
    gallery_dir = './data/gallery'
    gallery_paths = [os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.endswith('.jpg')]
    gallery_embs = get_embeddings(model, gallery_paths)
    return gallery_paths, gallery_embs

model = load_model()
gallery_paths, gallery_embs = load_gallery()

# ------------------------
# Upload and Predict
# ------------------------
st.subheader("Upload a Query Image")
file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if file:
    query_img = Image.open(file).convert('RGB')
    st.image(query_img, caption="Query Image", width=256)

    # Generate query embedding
    with torch.no_grad():
        query_tensor = preprocess_image(query_img)
        query_emb = model(query_tensor).detach().numpy()[0]

    # Match top-k
    top_paths, top_scores = match_topk(query_emb, gallery_embs, gallery_paths, k=5)

    st.markdown("### Top-5 Matches")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(top_paths[i], caption=f"Score: {top_scores[i]:.2f}", use_column_width=True)
