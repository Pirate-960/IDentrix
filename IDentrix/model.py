# model.py
"""
This module defines the core deep learning model for person re-identification.
The model uses a dual-backbone architecture:
- A ResNet-50 CNN for capturing local spatial features.
- A Vision Transformer (ViT) for modeling global context and relationships.

The outputs of both backbones are fused using a small attention module that learns
how much to weight each feature set dynamically. The result is passed through
a final projection head to get the final embedding, which can be used in a
metric learning setup (e.g., triplet loss or cosine similarity).
"""

import torch
import torch.nn as nn
import timm

class DualBackboneNet(nn.Module):
    def __init__(self, embedding_dim=128):
        """
        Initializes the dual-backbone model.

        Args:
            embedding_dim (int): Dimension of the final output embedding.
        """
        super().__init__()

        # CNN backbone (ResNet-50)
        self.cnn = timm.create_model('resnet50', pretrained=True, num_classes=0)

        # Vision Transformer backbone (ViT)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

        # Projection layers to reduce both backbones to the same embedding_dim
        self.fc_cnn = nn.Linear(self.cnn.num_features, embedding_dim)
        self.fc_vit = nn.Linear(self.vit.num_features, embedding_dim)

        # Attention mechanism to weight the CNN and ViT features
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 2),  # Outputs 2 scores: one for CNN, one for ViT
            nn.Softmax(dim=1)             # Normalize to get attention weights
        )

        # Final output projection head (can be identity or another linear layer)
        self.output_head = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input image tensor of shape [B, C, H, W].

        Returns:
            Tensor: Embedding vector of shape [B, embedding_dim].
        """
        # Extract features from both backbones
        f_cnn = self.fc_cnn(self.cnn(x))  # [B, embedding_dim]
        f_vit = self.fc_vit(self.vit(x))  # [B, embedding_dim]

        # Concatenate features and compute attention weights
        weights = self.attention(torch.cat([f_cnn, f_vit], dim=1))  # [B, 2]

        # Weighted fusion of features
        fused = weights[:, 0].unsqueeze(1) * f_cnn + weights[:, 1].unsqueeze(1) * f_vit

        # Final embedding
        return self.output_head(fused)  # [B, embedding_dim]
