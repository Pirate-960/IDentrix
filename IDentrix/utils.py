# utils.py
"""
This utility module provides helper functions for the person re-identification pipeline.

It includes functions for:
- Preprocessing images to make them compatible with the model.
- Generating batches of embeddings from image files using a trained model.
- Calculating similarity scores and finding the top-k matches for a query.

These functions are designed to be reusable across different scripts like training,
evaluation, and the interactive app.
"""

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from tqdm import tqdm
import logging

# ------------------------
# Image Preprocessing
# ------------------------
# A standard transform pipeline for images. It is compatible with the input
# requirements of both ResNet and ViT models from `timm`.
# - Resizes images to a fixed size (224x224).
# - Converts images to PyTorch Tensors.
# - Normalizes pixel values to the range [-1, 1], a common practice for many pre-trained models.
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # Normalize to [-1, 1]
])

def preprocess_image(img: Image.Image) -> torch.Tensor:
    """
    Applies the standard transformation pipeline to a single PIL image.

    Args:
        - img (PIL.Image): The input image to process.

    Returns:
        - torch.Tensor: The transformed image tensor, with a batch dimension added ([1, C, H, W]).
    """
    return transform(img).unsqueeze(0)

# ------------------------
# Embedding Extraction
# ------------------------
def get_embeddings(model: torch.nn.Module, image_paths: List[str]) -> np.ndarray:
    """
    Extracts embeddings for a list of images using a given model.

    Args:
        - model (torch.nn.Module): The trained model (e.g., DualBackboneNet).
        - image_paths (List[str]): A list of file paths to the images.

    Returns:
        - np.ndarray: A NumPy array of shape [N, embedding_dim], where N is the number of images.
    """
    model.eval()
    embeddings = []
    device = next(model.parameters()).device # Get model's device (cpu or cuda)

    # Use tqdm for a progress bar, as this can be a slow process
    logging.info(f"Extracting embeddings for {len(image_paths)} images...")
    for path in tqdm(image_paths, desc="Generating Embeddings"):
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = preprocess_image(img).to(device)
            with torch.no_grad():
                emb = model(img_tensor).cpu().numpy()
            embeddings.append(emb[0])
        except Exception as e:
            logging.error(f"Could not process image {path}: {e}")
            continue # Skip corrupted or unreadable images

    return np.array(embeddings)

# ------------------------
# Matching by Cosine Similarity
# ------------------------
def match_topk(query_emb: np.ndarray, gallery_embs: np.ndarray, gallery_paths: List[str], k: int = 5) -> Tuple[List[str], List[float]]:
    """
    Finds the top-k most similar gallery images for a single query embedding.

    Args:
        - query_emb (np.ndarray): The embedding of the query image, shape [embedding_dim].
        - gallery_embs (np.ndarray): A 2D array of gallery embeddings, shape [N, embedding_dim].
        - gallery_paths (List[str]): A list of file paths corresponding to the gallery embeddings.
        - k (int): The number of top matches to return.

    Returns:
        - Tuple[List[str], List[float]]: A tuple containing two lists:
            - The file paths of the top-k matched images.
            - The corresponding cosine similarity scores for those matches.
    """
    # Calculate cosine similarity between the single query and all gallery embeddings
    # query_emb must be 2D for the function, so we wrap it in a list.
    sims = cosine_similarity([query_emb], gallery_embs)[0]  # Shape: [N]

    # Get the indices of the top-k scores in descending order
    top_k_idx = np.argsort(sims)[::-1][:k]

    # Retrieve the paths and scores for the top-k indices
    top_paths = [gallery_paths[i] for i in top_k_idx]
    top_scores = [sims[i] for i in top_k_idx]

    return top_paths, top_scores