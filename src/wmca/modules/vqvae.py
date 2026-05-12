"""VQ-VAE for Crafter frames — Path C of Plan 0.

Vector-quantized autoencoder: 64×64×3 → 16×16 token grid → 64×64×3.
Codebook size V ∈ {256, 512}, embedding dim = 64.
Straight-through gradient + commitment loss + EMA codebook updates.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector-quantized bottleneck with EMA codebook updates.

    Encoder output (B, D, H, W) is quantized to nearest codebook entry.
    Straight-through estimator for backprop through the quantization step.
    Codebook maintained via exponential moving average of assigned embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25, decay: float = 0.99,
                 eps: float = 1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        # Codebook: (V, D)
        embed = torch.randn(num_embeddings, embedding_dim)
        embed = F.normalize(embed, dim=-1) * (1.0 / (embedding_dim ** 0.25))
        self.register_buffer("embedding", embed)

        # EMA accumulators
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed_avg", embed.clone())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous latents.

        Args:
            z: (B, D, H, W) continuous encoder output.

        Returns:
            z_q: (B, D, H, W) quantized output (straight-through).
            indices: (B, H, W) codebook indices for each spatial position.
            loss: scalar commitment + codebook loss.
        """
        B, D, H, W = z.shape
        # Flatten spatial dims: (B, D, H, W) → (B*H*W, D)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e.T
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)                  # (B*H*W, 1)
        e_sq = (self.embedding ** 2).sum(dim=1).unsqueeze(0)           # (1, V)
        dist = z_sq + e_sq - 2 * (z_flat @ self.embedding.T)           # (B*H*W, V)

        # Nearest neighbour
        encoding_indices = dist.argmin(dim=1)                          # (B*H*W,)
        z_q_flat = self.embedding[encoding_indices]                    # (B*H*W, D)

        # Straight-through: output = z_q + (z - z_q).detach()
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2)
        z_q_st = z + (z_q - z).detach()

        # Commitment loss: encoder should commit to codebook
        commitment_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())

        # Update EMA (only in training, detached)
        if self.training:
            with torch.no_grad():
                # One-hot encoding: (B*H*W, V)
                enc_onehot = F.one_hot(encoding_indices, self.num_embeddings).float()

                # Cluster sizes
                self.ema_cluster_size.mul_(self.decay).add_(
                    enc_onehot.sum(0), alpha=1 - self.decay
                )

                # Embedding averages
                embed_sum = enc_onehot.T @ z_flat  # (V, D)
                self.ema_embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # Laplace smoothing: normalise cluster sizes
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.num_embeddings * self.eps)
                    * n
                )

                # Normalised embedding
                embed_normalised = self.ema_embed_avg / cluster_size.unsqueeze(1)
                self.embedding.copy_(embed_normalised)

        indices = encoding_indices.view(B, H, W)
        # Note: codebook_loss is zero-gradient with EMA updates (embedding is a buffer).
        # Only commitment_loss provides encoder gradient. Codebook updates via EMA.
        loss = commitment_loss
        return z_q_st, indices, loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token indices to continuous embeddings.

        Args:
            indices: (B, H, W) long — token grid.

        Returns:
            z_q: (B, D, H, W) continuous embedding grid.
        """
        B, H, W = indices.shape
        z_q_flat = self.embedding[indices.view(-1)]                     # (B*H*W, D)
        return z_q_flat.view(B, H, W, -1).permute(0, 3, 1, 2)          # (B, D, H, W)

    def codebook_utilization(self) -> float:
        """Fraction of codebook entries that have been used.

        Returns float in [0, 1]. Dead codes are those whose EMA cluster
        size remains at (or near) zero.
        """
        alive = (self.ema_cluster_size > self.eps).float().sum().item()
        return alive / self.num_embeddings


class VQEncoder(nn.Module):
    """Encoder: (B, 3, 64, 64) → (B, D, 16, 16) continuous latent."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),       # 64 → 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),               # 32 → 16
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, padding=1),                  # 16 → 16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VQDecoder(nn.Module):
    """Decoder: (B, D, 16, 16) → (B, 3, 64, 64) reconstruction."""

    def __init__(self, out_channels: int = 3, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, padding=1),                   # 16 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),      # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),  # 32 → 64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VQVAE(nn.Module):
    """Vector-quantized variational autoencoder for Crafter frames.

    Encodes 64×64×3 RGB frames → 16×16 discrete token grid (from codebook
    of size num_embeddings) → reconstructs 64×64×3 frames.

    Training loss: L_recon + β * L_commit + L_codebook (EMA).
    """

    def __init__(self, num_embeddings: int = 512, embed_dim: int = 64,
                 in_channels: int = 3, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim

        self.encoder = VQEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embed_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = VQDecoder(out_channels=in_channels, embed_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: (B, 3, 64, 64) input frames in [0, 1].

        Returns:
            recon: (B, 3, 64, 64) reconstructed frames.
            indices: (B, 16, 16) codebook token indices.
            vq_loss: scalar VQ loss (commitment + codebook).
        """
        z = self.encoder(x)
        z_q, indices, vq_loss = self.vq(z)
        recon = self.decoder(z_q)
        return recon, indices, vq_loss

    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frames to discrete token grid.

        Args:
            x: (B, 3, 64, 64) frames.

        Returns:
            tokens: (B, 16, 16) long — codebook indices.
        """
        z = self.encoder(x)
        _, indices, _ = self.vq(z)
        return indices

    def decode_from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode token grid back to frames.

        Args:
            indices: (B, 16, 16) long — codebook indices.

        Returns:
            frames: (B, 3, 64, 64) reconstructed frames.
        """
        z_q = self.vq.decode_indices(indices)
        return self.decoder(z_q)

    def encode_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to pre-quantization continuous latent (for baseline).

        Args:
            x: (B, 3, 64, 64) frames.

        Returns:
            z: (B, D, 16, 16) continuous latent (no VQ).
        """
        return self.encoder(x)

    def codebook_utilization(self) -> float:
        """Fraction of codebook entries that are alive."""
        return self.vq.codebook_utilization()

    def param_count(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        vq_bufs = sum(b.numel() for b in self.vq.buffers())
        return {
            "encoder": enc,
            "decoder": dec,
            "vq_buffers": vq_bufs,
            "total": enc + dec + vq_bufs,
        }
