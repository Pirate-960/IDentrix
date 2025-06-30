# train.py
"""
This script trains the DualBackboneNet using triplet loss for person re-identification.
It handles dataset loading, training loop, logging, and checkpoint saving.

Make sure to prepare the dataset (e.g., Market-1501) under the proper directory before running.
"""

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from model import DualBackboneNet

# ------------------------
# Triplet Loss Function
# ------------------------
def triplet_loss(anchor, positive, negative, margin=1.0):
    d_pos = F.pairwise_distance(anchor, positive)
    d_neg = F.pairwise_distance(anchor, negative)
    return torch.mean(F.relu(d_pos - d_neg + margin))

# ------------------------
# Triplet Dataset Class
# ------------------------
class TripletDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.label_to_images = self._index_images()
        self.labels = list(self.label_to_images.keys())

    def _index_images(self):
        label_to_images = {}
        for fname in os.listdir(self.image_dir):
            if not fname.endswith('.jpg'): continue
            label = int(fname.split('_')[0])
            path = os.path.join(self.image_dir, fname)
            label_to_images.setdefault(label, []).append(path)
        return label_to_images

    def __len__(self):
        return 10000  # Arbitrary sample size for epoch length

    def __getitem__(self, idx):
        anchor_label = random.choice(self.labels)
        positive_label = anchor_label
        negative_label = random.choice([l for l in self.labels if l != anchor_label])

        anchor_img, positive_img = random.sample(self.label_to_images[anchor_label], 2)
        negative_img = random.choice(self.label_to_images[negative_label])

        def load(path):
            img = Image.open(path).convert('RGB')
            return self.transform(img) if self.transform else img

        return load(anchor_img), load(positive_img), load(negative_img)

# ------------------------
# Data Transforms
# ------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(0.2, 0.2, 0.2, 0.2),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# ------------------------
# Training Function
# ------------------------
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for anchor, pos, neg in dataloader:
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        anchor_emb = model(anchor)
        pos_emb = model(pos)
        neg_emb = model(neg)

        loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# ------------------------
# Main Script
# ------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DualBackboneNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Modify this path to your dataset directory
    dataset = TripletDataset(image_dir='./data/gallery', transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Training loop
    for epoch in range(10):
        avg_loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/epoch{epoch+1}_model.pth')
