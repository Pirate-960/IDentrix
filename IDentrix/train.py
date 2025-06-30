# train.py
"""
This script handles the training of the DualBackboneNet for person re-identification.

Key components:
- Triplet Loss: A metric learning loss function that aims to minimize the distance
  between an anchor and a positive sample (same person) while maximizing the
  distance between the anchor and a negative sample (different person).
- TripletDataset: A custom PyTorch Dataset class that dynamically samples
  triplets (anchor, positive, negative) from the training data for each iteration.
- Training Loop: A standard loop that iterates over epochs, processes batches of
  triplets, computes the loss, and updates the model weights.

This script is enhanced with:
- Multi-GPU Support: Automatically uses `torch.nn.DataParallel` if multiple GPUs are detected.
- Multiprocessing Data Loading: Uses multiple CPU workers to pre-fetch data, preventing
  bottlenecks and maximizing GPU utilization.
- W&B Integration: Logs metrics, hyperparameters, and model checkpoints to Weights & Biases
  for robust experiment tracking and visualization.

Make sure the dataset is prepared (e.g., via prepare_dataset.py) before running.
"""

import os
import torch
import torch.nn as nn # Imported for nn.DataParallel
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import logging
from tqdm import tqdm
from model import DualBackboneNet
from pathlib import Path
import wandb # Import Weights & Biases

# --- Suppress the symlink warning from huggingface_hub ---
# This environment variable is set to silence a benign warning on Windows systems
# where symbolic links are not enabled by default.
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- Basic logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ------------------------
# Triplet Loss Function
# ------------------------
def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Calculates the triplet loss.
    Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    
    Args:
        - anchor (torch.Tensor): Embeddings of the anchor samples.
        - positive (torch.Tensor): Embeddings of the positive samples.
        - negative (torch.Tensor): Embeddings of the negative samples.
        - margin (float): The desired margin between positive and negative distances.

    Returns:
        - torch.Tensor: The mean triplet loss for the batch.
    """
    # F.pairwise_distance computes the L2 distance between corresponding pairs
    # d_pos is the distance between anchor and positive samples
    d_pos = F.pairwise_distance(anchor, positive)
    # d_neg is the distance between anchor and negative samples
    d_neg = F.pairwise_distance(anchor, negative)
    
    # F.relu implements the max(0, x) part of the loss function
    loss = F.relu(d_pos - d_neg + margin)
    
    # Return the average loss over the batch
    return torch.mean(loss)

# ------------------------
# Triplet Dataset Class
# ------------------------
class TripletDataset(Dataset):
    """
    A PyTorch Dataset that generates triplets for training.
    For each item request, it randomly selects:
    1. An "anchor" person ID.
    2. Two different images of that person (anchor and positive).
    3. A "negative" person ID (different from the anchor).
    4. One image from the negative person's collection.
    """
    def __init__(self, image_dir: str, transform: T.Compose = None):
        """
        Initializes the dataset by indexing all images by their person ID.
        
        Args:
            - image_dir (str): Path to the directory containing training images.
            - transform (T.Compose): Torchvision transforms to apply to each image.
        """
        self.image_dir = image_dir
        self.transform = transform

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"The specified image directory does not exist: {self.image_dir}")

        self.label_to_images, self.labels = self._index_images()
        
        if not self.labels:
            raise ValueError(f"No valid images found or no IDs with >= 2 images in directory: {image_dir}")
        logging.info(f"Dataset initialized with {len(self.labels)} unique IDs for training.")

    def _index_images(self):
        """
        Scans the image directory and groups image paths by person ID.
        - Assumes Market-1501 naming convention: `personID_cameraID_... .jpg`
        - Ignores IDs with only one image, as they can't form a positive pair.
        """
        label_to_images = {}
        logging.info(f"Indexing images by person ID from '{self.image_dir}'...")
        for fname in os.listdir(self.image_dir):
            if not fname.endswith('.jpg'):
                continue
            try:
                # The label (person ID) is the first part of the filename
                label = int(fname.split('_')[0])
                path = os.path.join(self.image_dir, fname)
                label_to_images.setdefault(label, []).append(path)
            except (ValueError, IndexError):
                logging.warning(f"Could not parse label from filename: {fname}. Skipping.")
                continue
        
        # Filter out labels with fewer than 2 images
        filtered_label_to_images = {k: v for k, v in label_to_images.items() if len(v) >= 2}
        labels = list(filtered_label_to_images.keys())
        logging.info(f"Found {len(label_to_images)} total IDs, {len(labels)} usable for training (>= 2 images).")
        
        return filtered_label_to_images, labels

    def __len__(self):
        # We can sample a large number of triplets. This length defines an "epoch".
        # 10,000 samples per epoch is a reasonable starting point.
        return 10000

    def __getitem__(self, idx: int):
        # --- Step 1: Select anchor and negative person IDs ---
        anchor_label = random.choice(self.labels)
        negative_label = random.choice([l for l in self.labels if l != anchor_label])

        # --- Step 2: Select anchor and positive images ---
        # `random.sample` selects two unique images from the anchor person's list
        anchor_path, positive_path = random.sample(self.label_to_images[anchor_label], 2)
        
        # --- Step 3: Select a negative image ---
        negative_path = random.choice(self.label_to_images[negative_label])

        # --- Step 4: Load and transform images ---
        def load_image(path: str):
            img = Image.open(path).convert('RGB')
            return self.transform(img) if self.transform else img

        anchor_img = load_image(anchor_path)
        positive_img = load_image(positive_path)
        negative_img = load_image(negative_path)

        return anchor_img, positive_img, negative_img

# ------------------------
# Data Transforms
# ------------------------
# Apply data augmentation to make the model more robust to variations.
transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3) # Normalize to [-1, 1]
])

# ------------------------
# Training Function
# ------------------------
def run_training_epoch(model: torch.nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: str, epoch_num: int):
    """
    Executes a single training epoch.

    Args:
        - model: The model to train.
        - dataloader: DataLoader providing triplet batches.
        - optimizer: The optimization algorithm.
        - device: The device to train on ('cpu' or 'cuda').
        - epoch_num (int): The current epoch number, for display purposes.

    Returns:
        - float: The average training loss for the epoch.
    """
    model.train()  # Set the model to training mode
    total_loss = 0.0
    
    # Use tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch_num}", unit="batch")
    
    for anchor, pos, neg in progress_bar:
        # Move data to the selected device
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        
        # --- Forward pass ---
        anchor_emb = model(anchor)
        pos_emb = model(pos)
        neg_emb = model(neg)

        # --- Loss calculation and backward pass ---
        loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update model weights

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Log metrics to W&B for each batch
        wandb.log({"batch_loss": loss.item()})

    return total_loss / len(dataloader)

# ------------------------
# Main Script
# ------------------------
if __name__ == '__main__':
    # --- Define paths relative to the script's location ---
    # This makes the script runnable from any directory.
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # --- Configuration dictionary for W&B ---
    config = {
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "architecture": "DualBackboneNet (ResNet50+ViT)",
        "dataset": "Market-1501 (Gallery Split)",
    }
    
    # --- Parallelism Configuration ---
    # Use a sensible number of CPU cores for data loading.
    # `os.cpu_count()` gives the total number of cores available.
    # We cap it at 8 to prevent overwhelming the system.
    NUM_WORKERS = min(os.cpu_count(), 8)
    
    # Use the gallery set for training
    DATA_DIR = SCRIPT_DIR / 'data' / 'gallery'
    CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
    
    # --- Initialize W&B Run ---
    wandb.init(
        project="IDentrix-ReID", # Your project name
        config=config,           # The hyperparameter configuration
        job_type="training"      # Categorize the run
    )
    # W&B injects its own config object, which is best practice to use
    config = wandb.config
    
    # --- Setup Device and Model Parallelism ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    model = DualBackboneNet()
    
    # --- Multi-GPU Data Parallelism ---
    if device == 'cuda' and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        logging.info(f"Multiple GPUs detected. Using {num_gpus} GPUs for Data Parallelism.")
        model = nn.DataParallel(model)
        # You can often increase the batch size when using multiple GPUs
        # For example: BATCH_SIZE = BATCH_SIZE * num_gpus
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # --- Dataset and DataLoader with Multiprocessing ---
    try:
        dataset = TripletDataset(image_dir=str(DATA_DIR), transform=transform)
        # Use `num_workers` to enable multiprocessing for data loading.
        # `pin_memory=True` can speed up CPU-to-GPU data transfer.
        logging.info(f"Initializing DataLoader with {NUM_WORKERS} worker processes.")
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
    except (ValueError, FileNotFoundError) as e:
        logging.error(f"Failed to create dataset: {e}")
        logging.error("Please ensure you have run 'prepare_dataset.py' and that the 'data/gallery' folder is populated.")
        # End W&B run with a failure code
        wandb.finish(exit_code=1)
        exit()

    # --- Training Loop ---
    logging.info("Starting training...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Log learning rate and watch the model
    wandb.watch(model, log="all", log_freq=100)
    
    for epoch in range(config.num_epochs):
        epoch_num = epoch + 1
        logging.info(f"--- Epoch {epoch + 1}/{config.num_epochs} ---")
        avg_loss = run_training_epoch(model, dataloader, optimizer, device, epoch_num)
        logging.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
        
        # Log epoch-level summary metrics
        wandb.log({"epoch": epoch_num, "avg_epoch_loss": avg_loss, "learning_rate": config.learning_rate})

        # --- Save Checkpoint ---
        checkpoint_path = CHECKPOINT_DIR / f'epoch_{epoch_num}_model.pth'
        
        # --- Handle saving for both DataParallel and single-GPU models ---
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, str(checkpoint_path))
            
        logging.info(f"Checkpoint saved locally to {checkpoint_path}")
        
        # --- Log model as a W&B Artifact ---
        artifact = wandb.Artifact(
            name=f'DualBackboneNet-{wandb.run.id}', # Unique name for each run's model
            type='model',
            description=f'Model checkpoint after epoch {epoch_num}',
            metadata=config.as_dict() # Log hyperparameters with the model
        )
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact, aliases=[f"epoch-{epoch_num}", "latest"])
        logging.info("Checkpoint saved as W&B Artifact.")
    
    # End the W&B run
    wandb.finish()
    logging.info("Training finished.")