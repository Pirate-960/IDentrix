# evaluate.py
"""
This script evaluates the trained model on person re-identification tasks
using Rank-1 accuracy and Mean Average Precision (mAP).
You should prepare `query/` and `gallery/` folders with images and labels.
"""

import os
import numpy as np
from PIL import Image
import torch
from model import DualBackboneNet
from utils import preprocess_image
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------
# Load Image Paths and Labels
# ------------------------
def load_image_paths_and_labels(folder):
    image_paths, labels = [], []
    for fname in os.listdir(folder):
        if fname.endswith('.jpg'):
            label = int(fname.split('_')[0])  # Assuming Market-1501 naming convention
            path = os.path.join(folder, fname)
            image_paths.append(path)
            labels.append(label)
    return image_paths, labels

# ------------------------
# Embedding Extraction
# ------------------------
def extract_embeddings(model, image_paths):
    model.eval()
    embeddings = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        tensor = preprocess_image(img).to(next(model.parameters()).device)
        with torch.no_grad():
            emb = model(tensor).cpu().numpy()
        embeddings.append(emb[0])
    return np.array(embeddings)

# ------------------------
# Evaluation Metrics
# ------------------------
def rank1_accuracy(sim_matrix, query_labels, gallery_labels):
    correct = 0
    for i, sims in enumerate(sim_matrix):
        top1 = np.argmax(sims)
        if gallery_labels[top1] == query_labels[i]:
            correct += 1
    return correct / len(query_labels)

def mean_average_precision(sim_matrix, query_labels, gallery_labels):
    aps = []
    for i, sims in enumerate(sim_matrix):
        sorted_idx = np.argsort(sims)[::-1]
        correct = 0
        total = 0
        precision_at_k = []
        for rank, idx in enumerate(sorted_idx):
            if gallery_labels[idx] == query_labels[i]:
                correct += 1
                precision_at_k.append(correct / (rank + 1))
        ap = np.mean(precision_at_k) if precision_at_k else 0
        aps.append(ap)
    return np.mean(aps)

# ------------------------
# Main Evaluation
# ------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DualBackboneNet().to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))

    query_paths, query_labels = load_image_paths_and_labels('./data/query')
    gallery_paths, gallery_labels = load_image_paths_and_labels('./data/gallery')

    print("Extracting embeddings...")
    query_embs = extract_embeddings(model, query_paths)
    gallery_embs = extract_embeddings(model, gallery_paths)

    print("Computing similarity matrix...")
    sim_matrix = cosine_similarity(query_embs, gallery_embs)

    r1 = rank1_accuracy(sim_matrix, query_labels, gallery_labels)
    map_score = mean_average_precision(sim_matrix, query_labels, gallery_labels)

    print(f"Rank-1 Accuracy: {r1:.4f}")
    print(f"Mean Average Precision: {map_score:.4f}")
