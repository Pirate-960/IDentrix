# utils.py
"""
This utility module provides helper functions for preprocessing images,
generating embeddings from the model, computing similarity scores,
and selecting top-k matches based on cosine similarity.

These functions support both offline evaluation and real-time/demo use cases.
"""

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# Image Preprocessing
# ------------------------
# Define a standard transform pipeline compatible with ViT and ResNet inputs
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def preprocess_image(img):
    """
    Applies necessary transforms to a single PIL image.

    Args:
        img (PIL.Image): Input image.

    Returns:
        Tensor: Transformed image tensor [1, C, H, W].
    """
    return transform(img).unsqueeze(0)

# ------------------------
# Embedding Extraction
# ------------------------
def get_embeddings(model, image_paths):
    """
    Extracts embeddings from a list of image paths using the provided model.

    Args:
        model (nn.Module): Trained model to extract features.
        image_paths (List[str]): List of image file paths.

    Returns:
        np.ndarray: Array of embeddings [N, embedding_dim].
    """
    model.eval()
    embeddings = []
    device = next(model.parameters()).device
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img_tensor = preprocess_image(img).to(device)
        with torch.no_grad():
            emb = model(img_tensor).cpu().numpy()
        embeddings.append(emb[0])
    return np.array(embeddings)

# ------------------------
# Matching by Cosine Similarity
# ------------------------
def match_topk(query_emb, gallery_embs, gallery_paths, k=5):
    """
    Matches a single query embedding to a gallery using cosine similarity.

    Args:
        query_emb (np.ndarray): Query embedding of shape [embedding_dim].
        gallery_embs (np.ndarray): Gallery embeddings of shape [N, embedding_dim].
        gallery_paths (List[str]): Corresponding image paths for the gallery.
        k (int): Number of top matches to return.

    Returns:
        Tuple[List[str], List[float]]: Top-k image paths and their similarity scores.
    """
    sims = cosine_similarity([query_emb], gallery_embs)[0]  # [N]
    top_k_idx = np.argsort(sims)[::-1][:k]  # Indices of top-k matches
    return [gallery_paths[i] for i in top_k_idx], [sims[i] for i in top_k_idx]
