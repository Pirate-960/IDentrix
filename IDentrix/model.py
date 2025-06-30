# model.py
"""
This module defines the core deep learning model for person re-identification.

The model uses a dual-backbone architecture to leverage the strengths of both
Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs):
- A ResNet-50 CNN excels at capturing local, texture-based, and spatial features.
- A Vision Transformer (ViT) excels at modeling global context and long-range
  dependencies between image patches.

The features from both backbones are fused using a small attention module. This
module learns to dynamically weight the importance of CNN vs. ViT features for
a given input image. The final fused vector is projected to produce the output
embedding, which is optimized for metric learning tasks.
"""

import torch
import torch.nn as nn
import timm

class DualBackboneNet(nn.Module):
    """
    A dual-backbone network combining a CNN and a ViT for feature extraction.
    """
    def __init__(self, embedding_dim=128):
        """
        Initializes the layers of the dual-backbone model.

        Args:
            embedding_dim (int): The target dimension for the final output embedding.
        """
        super().__init__()

        # --- Backbone 1: CNN (ResNet-50) ---
        # - Pre-trained on ImageNet for strong initial feature extraction.
        # - `num_classes=0` removes the final classification layer, turning it into a feature extractor.
        self.cnn = timm.create_model('resnet50', pretrained=True, num_classes=0)

        # --- Backbone 2: Vision Transformer (ViT) ---
        # - Pre-trained on ImageNet.
        # - Also used as a feature extractor with `num_classes=0`.
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

        # --- Projection Heads ---
        # - These linear layers reduce the feature dimensions from both backbones
        #   to a common, manageable size (`embedding_dim`).
        self.fc_cnn = nn.Linear(self.cnn.num_features, embedding_dim)
        self.fc_vit = nn.Linear(self.vit.num_features, embedding_dim)

        # --- Attention Fusion Module ---
        # - A small MLP that takes concatenated features from both backbones.
        # - It outputs two weights (one for CNN, one for ViT) that sum to 1.
        # - This allows the model to learn the optimal blend of local and global features.
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), # Input is concatenated features
            nn.ReLU(),
            nn.Linear(embedding_dim, 2),                  # Outputs 2 scores
            nn.Softmax(dim=1)                             # Normalize scores into weights [w_cnn, w_vit]
        )

        # --- Final Output Head ---
        # - A final linear layer to project the fused features. This can add
        #   extra modeling capacity before the final output.
        self.output_head = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (Tensor): Input image tensor with shape [B, C, H, W].

        Returns:
            Tensor: The final embedding vector with shape [B, embedding_dim].
        """
        # --- Step 1: Extract features from both backbones ---
        # The output of each backbone is passed through its projection head.
        f_cnn = self.fc_cnn(self.cnn(x))  # Shape: [B, embedding_dim]
        f_vit = self.fc_vit(self.vit(x))  # Shape: [B, embedding_dim]

        # --- Step 2: Compute attention weights ---
        # Concatenate the features and pass them to the attention module.
        combined_features = torch.cat([f_cnn, f_vit], dim=1) # Shape: [B, embedding_dim * 2]
        weights = self.attention(combined_features)          # Shape: [B, 2]

        # --- Step 3: Fuse features using the learned weights ---
        # Perform a weighted sum of the CNN and ViT features.
        # `weights[:, 0]` is the weight for the CNN, `weights[:, 1]` for the ViT.
        # `unsqueeze(1)` is used to make the weights broadcastable.
        fused = weights[:, 0].unsqueeze(1) * f_cnn + weights[:, 1].unsqueeze(1) * f_vit

        # --- Step 4: Final projection ---
        # Pass the fused features through the final head to get the embedding.
        embedding = self.output_head(fused)  # Shape: [B, embedding_dim]

        return embedding