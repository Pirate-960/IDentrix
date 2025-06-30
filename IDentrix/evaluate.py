# evaluate.py
"""
This script evaluates the performance of a trained person re-identification model.

It uses two standard metrics:
- Rank-1 Accuracy: The percentage of queries for which the top-ranked gallery
  image has the correct person ID.
- Mean Average Precision (mAP): A more comprehensive metric that considers the
  ranking of all correct gallery images for each query.

W&B Integration:
- Downloads the model to be evaluated from W&B Artifacts.
- Logs the final evaluation scores to a new W&B run for comparison.
- Creates a results table in the W&B dashboard.

The script requires a `query/` folder and a `gallery/` folder, populated with
images following the Market-1501 naming convention (`personID_... .jpg`).
"""

import os
import numpy as np
from PIL import Image
import torch
from model import DualBackboneNet
from utils import preprocess_image
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import logging
from pathlib import Path
import wandb # NEW: Import Weights & Biases

# --- Basic logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ------------------------
# Load Image Paths and Labels
# ------------------------
def load_image_paths_and_labels(folder: str) -> Tuple[List[str], List[int]]:
    """
    Loads image paths and their corresponding person IDs from a directory.

    Args:
        - folder (str): The path to the directory (e.g., './data/query').

    Returns:
        - Tuple[List[str], List[int]]: A tuple containing:
            - A list of full image paths.
            - A list of corresponding integer labels (person IDs).
    """
    image_paths, labels = [], []
    logging.info(f"Loading images and labels from '{folder}'...")
    if not os.path.isdir(folder):
        logging.error(f"Directory not found: {folder}")
        return [], []
        
    for fname in sorted(os.listdir(folder)): # Sort for consistent order
        if fname.endswith('.jpg'):
            try:
                # Assumes Market-1501 naming: `0001_c1s1_001051_00.jpg` -> label 0001
                label = int(fname.split('_')[0])
                path = os.path.join(folder, fname)
                image_paths.append(path)
                labels.append(label)
            except (ValueError, IndexError):
                logging.warning(f"Skipping file with invalid name format: {fname}")
                continue
                
    logging.info(f"Found {len(image_paths)} images in '{folder}'.")
    return image_paths, labels

# ------------------------
# Embedding Extraction
# ------------------------
def extract_embeddings(model: torch.nn.Module, image_paths: List[str], device: str) -> np.ndarray:
    """
    Extracts embeddings for a list of images using the provided model.

    Args:
        - model (torch.nn.Module): The trained model.
        - image_paths (List[str]): List of image file paths.
        - device (str): The device ('cpu' or 'cuda') to use for inference.

    Returns:
        - np.ndarray: A NumPy array of embeddings [N, embedding_dim].
    """
    model.eval()
    embeddings = []
    logging.info(f"Extracting embeddings for {len(image_paths)} images...")
    
    with torch.no_grad():
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                tensor = preprocess_image(img).to(device)
                emb = model(tensor).cpu().numpy()
                embeddings.append(emb[0])
            except Exception as e:
                logging.error(f"Failed to process image {path}: {e}")
                
    return np.array(embeddings)

# ------------------------
# Evaluation Metrics
# ------------------------
def rank1_accuracy(sim_matrix: np.ndarray, query_labels: List[int], gallery_labels: List[int]) -> float:
    """
    Calculates Rank-1 accuracy.

    Args:
        - sim_matrix (np.ndarray): Similarity matrix [num_queries, num_gallery].
        - query_labels (List[int]): Labels for the query images.
        - gallery_labels (List[int]): Labels for the gallery images.

    Returns:
        - float: The Rank-1 accuracy score (0.0 to 1.0).
    """
    correct_matches = 0
    # Get the index of the highest similarity score for each query
    top1_indices = np.argmax(sim_matrix, axis=1)
    
    for i, top1_idx in enumerate(top1_indices):
        # Check if the label of the top-ranked gallery image matches the query label
        if gallery_labels[top1_idx] == query_labels[i]:
            correct_matches += 1
            
    return correct_matches / len(query_labels)

def mean_average_precision(sim_matrix: np.ndarray, query_labels: List[int], gallery_labels: List[int]) -> float:
    """
    Calculates Mean Average Precision (mAP).

    Args:
        - sim_matrix (np.ndarray): Similarity matrix [num_queries, num_gallery].
        - query_labels (List[int]): Labels for the query images.
        - gallery_labels (List[int]): Labels for the gallery images.

    Returns:
        - float: The mAP score (0.0 to 1.0).
    """
    average_precisions = []
    gallery_labels = np.array(gallery_labels) # Convert to numpy for efficient boolean indexing

    for i, sims in enumerate(sim_matrix):
        query_label = query_labels[i]
        
        # Sort gallery indices by similarity in descending order
        sorted_indices = np.argsort(sims)[::-1]
        
        # Create a boolean mask indicating correct matches
        is_match = (gallery_labels[sorted_indices] == query_label)
        
        # Get the rank of each correct match
        correct_match_ranks = np.where(is_match)[0] + 1
        
        if len(correct_match_ranks) == 0:
            average_precisions.append(0.0)
            continue
        
        # Calculate precision at each correct match
        # Precision@k = (number of correct matches up to rank k) / k
        precision_at_k = np.arange(1, len(correct_match_ranks) + 1) / correct_match_ranks
        
        # Average precision for this query is the mean of precision@k values
        ap = np.mean(precision_at_k)
        average_precisions.append(ap)
        
    return np.mean(average_precisions)

# ------------------------
# Main Evaluation
# ------------------------
if __name__ == '__main__':
    # --- Configuration ---
    # Define paths relative to the script's location
    SCRIPT_DIR = Path(__file__).resolve().parent
    QUERY_DIR = SCRIPT_DIR / 'data' / 'query'
    GALLERY_DIR = SCRIPT_DIR / 'data' / 'gallery'
    
    # --- NEW: W&B Configuration ---
    # IMPORTANT: Replace "YOUR_ENTITY/IDentrix-ReID" with your W&B username/entity and project name.
    # We use ":latest" to get the model from the most recent training run.
    WANDB_PROJECT = "IDentrix-ReID"
    MODEL_ARTIFACT_NAME = "DualBackboneNet" # Should match the name used in train.py
    
    # --- Initialize W&B Run ---
    run = wandb.init(project=WANDB_PROJECT, job_type="evaluation")

    # --- Setup ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # --- NEW: Download Model from W&B Artifacts with a fallback ---
    try:
        # Find the latest artifact from the most recent run
        # This is more robust than hardcoding a run ID.
        api = wandb.Api()
        artifacts = api.artifacts(f"{WANDB_PROJECT}/{MODEL_ARTIFACT_NAME}")
        latest_artifact = artifacts[0] # The API returns artifacts sorted by creation time
        
        logging.info(f"Downloading latest model artifact from W&B: {latest_artifact.name}")
        run.config.update({"model_artifact": latest_artifact.name}) # Log which model we're using
        artifact_dir = latest_artifact.download()
        model_path = next(Path(artifact_dir).glob("*.pth")) # Find the .pth file in the downloaded dir
        logging.info(f"Model downloaded to {model_path}")
    except Exception as e:
        logging.error(f"Could not download model from W&B: {e}")
        logging.info("Falling back to local 'checkpoints/best_model.pth'")
        model_path = SCRIPT_DIR / 'checkpoints' / 'best_model.pth'
        
        if not model_path.exists():
            logging.error(f"Local model checkpoint not found at {model_path}. Please train a model first.")
            run.finish(exit_code=1)
            exit()

    model = DualBackboneNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    logging.info(f"Model loaded from {model_path}")

    # --- Load Data ---
    query_paths, query_labels = load_image_paths_and_labels(str(QUERY_DIR))
    gallery_paths, gallery_labels = load_image_paths_and_labels(str(GALLERY_DIR))
    
    if not query_paths or not gallery_paths:
        logging.error("Query or gallery data is missing. Please run prepare_dataset.py.")
        run.finish(exit_code=1)
        exit()

    # --- Extract Embeddings ---
    query_embs = extract_embeddings(model, query_paths, device)
    gallery_embs = extract_embeddings(model, gallery_paths, device)

    # --- Compute Similarity ---
    logging.info("Computing similarity matrix...")
    sim_matrix = cosine_similarity(query_embs, gallery_embs)
    logging.info(f"Similarity matrix computed with shape: {sim_matrix.shape}")

    # --- Calculate and Print Metrics ---
    r1 = rank1_accuracy(sim_matrix, query_labels, gallery_labels)
    map_score = mean_average_precision(sim_matrix, query_labels, gallery_labels)

    print("\n--- Evaluation Results ---")
    print(f"Rank-1 Accuracy: {r1:.4f}")
    print(f"Mean Average Precision (mAP): {map_score:.4f}")
    print("--------------------------\n")
    
    # --- NEW: Log metrics and summary table to W&B ---
    wandb.summary["rank1_accuracy"] = r1
    wandb.summary["map_score"] = map_score
    
    results_table = wandb.Table(columns=["Metric", "Score"])
    results_table.add_data("Rank-1 Accuracy", r1)
    results_table.add_data("mAP", map_score)
    run.log({"evaluation_results": results_table})
    
    logging.info("Results logged to W&B.")
    wandb.finish()