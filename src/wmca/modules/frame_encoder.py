from __future__ import annotations

import torch
import torch.nn as nn


class FrameEncoder(nn.Module):
    """CNN encoder: (B, 3, 64, 64) -> (B, 1, 16, 16)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrameDecoder(nn.Module):
    """CNN decoder: (B, 1, 16, 16) -> (B, 3, 64, 64)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrameAutoencoder(nn.Module):
    """Full autoencoder for 64x64 RGB frames."""

    def __init__(self):
        super().__init__()
        self.encoder = FrameEncoder()
        self.decoder = FrameDecoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def param_count(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        return {"encoder": enc, "decoder": dec, "total": enc + dec}


class FrozenFrameEncoder(nn.Module):
    """Frozen encoder loaded from a trained FrameAutoencoder checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        super().__init__()
        autoencoder = FrameAutoencoder()
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        autoencoder.load_state_dict(state)
        self.encoder = autoencoder.encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
