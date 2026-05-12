"""DiscreteRescor — CML+NCA world model for discrete token prediction.

Path C of Plan 0: Predicts discrete VQ-VAE token sequences autoregressively.
Embeds token indices + action indices into continuous space, runs CML reservoir,
applies NCA correction in embedding space, outputs vocabulary logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wmca.modules.hybrid import CML2DMultiR


class DiscreteRescor(nn.Module):
    """Discrete-token world model for VQ-VAE token sequence prediction.

    Takes discrete token indices (long tensor) and action index as input.
    Embeds both, projects to 1-channel continuous CML input, runs CML+NCA
    in embedding space, outputs per-position vocabulary logits.

    Architecture:
        tokens (B,H,W) + action (B,) → embed → project → [0,1] → CML → NCA → logits
    """

    def __init__(
        self,
        vocab_size: int = 512,
        n_actions: int | None = None,
        embed_dim: int = 64,
        hidden_ch: int = 16,
        cml_K: int = 32,
        cml_steps: int = 15,
        r_lo: float = 3.57,
        r_hi: float = 3.99,
        use_sigmoid: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_actions = n_actions
        self.embed_dim = embed_dim
        self.use_sigmoid = use_sigmoid

        # Token embedding: V → embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Action embedding (optional)
        if n_actions is not None and n_actions > 0:
            self.action_embed = nn.Embedding(n_actions, embed_dim)
            input_ch = embed_dim * 2  # token + action
        else:
            self.action_embed = None
            input_ch = embed_dim

        # Project combined embeddings → 1ch CML input
        self.input_proj = nn.Conv2d(input_ch, 1, 1)

        # CML reservoir: K parallel CML passes with different r values, uniform 1/K averaged
        self.cml = CML2DMultiR(
            in_channels=1,
            K=cml_K,
            steps=cml_steps,
            r_lo=r_lo,
            r_hi=r_hi,
            seed=seed,
            gate_mode="uniform",
        )

        # NCA correction: [cml_input (1ch), cml_mean (1ch)] → correction in embed_dim
        self.nca = nn.Sequential(
            nn.Conv2d(1 + 1, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, embed_dim, 1),
        )

        # CML output projection: 1ch → embed_dim (learned, not zero-padded)
        self.cml_proj = nn.Conv2d(1, embed_dim, 1)

        # Output head: embedding space → vocabulary logits
        self.output_head = nn.Conv2d(embed_dim, vocab_size, 1)

    def forward(self, tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass — discrete token prediction.

        Args:
            tokens: (B, H, W) long — discrete token index grid.
            action: (B,) long — discrete action index, or None for no action.

        Returns:
            logits: (B, V, H, W) raw logits over vocabulary (no sigmoid/softmax).
        """
        if self.use_sigmoid:
            raise ValueError(
                "DiscreteRescor outputs vocabulary logits for CE loss. "
                "Do not apply sigmoid. Set use_sigmoid=False."
            )

        B, H, W = tokens.shape

        # Embed tokens: (B, H, W) → (B, embed_dim, H, W)
        tok_emb = self.token_embed(tokens).permute(0, 3, 1, 2)

        # Embed action (if present) and concatenate
        if self.action_embed is not None and action is not None:
            act_emb = self.action_embed(action)                                # (B, embed_dim)
            act_emb = act_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            combined = torch.cat([tok_emb, act_emb], dim=1)                   # (B, 2*embed_dim, H, W)
        else:
            combined = tok_emb

        # Project to 1ch CML input, squash to [0,1] for logistic map domain
        cml_input = torch.sigmoid(self.input_proj(combined))                  # (B, 1, H, W)

        # Frozen CML reservoir forward (no gradients through CML)
        cml_out = self.cml(cml_input)                                          # (B, 1, H, W)

        # NCA correction in embedding space
        nca_in = torch.cat([cml_input, cml_out], dim=1)                       # (B, 2, H, W)
        correction = self.nca(nca_in)                                          # (B, embed_dim, H, W)

        # Project CML output to embedding space (learned, not zero-padded)
        cml_emb = self.cml_proj(cml_out)                                      # (B, embed_dim, H, W)

        # Residual in embedding space: cml embedding + NCA correction
        embedding = correction + cml_emb                                       # (B, embed_dim, H, W)

        # Project to vocabulary logits
        logits = self.output_head(embedding)                                   # (B, V, H, W)

        return logits

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.buffers())
        return {"trained": trained, "frozen": frozen}

    def get_alpha_params(self) -> list[nn.Parameter]:
        return []


class DiscreteRescorMamba(nn.Module):
    """Discrete-token world model with per-cell Mamba SSM for temporal context.

    Processes K=4 consecutive token frames through a Mamba block before
    feeding to the CML+NCA pipeline. The Mamba provides temporal context
    beyond the single-frame CML reservoir.
    """

    def __init__(
        self,
        vocab_size: int = 512,
        n_actions: int | None = None,
        embed_dim: int = 64,
        hidden_ch: int = 16,
        cml_K: int = 32,
        cml_steps: int = 15,
        r_lo: float = 3.57,
        r_hi: float = 3.99,
        mamba_context: int = 4,
        use_sigmoid: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_actions = n_actions
        self.embed_dim = embed_dim
        self.mamba_context = mamba_context
        self.use_sigmoid = use_sigmoid

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Action embedding
        if n_actions is not None and n_actions > 0:
            self.action_embed = nn.Embedding(n_actions, embed_dim)
        else:
            self.action_embed = None

        # Mamba block for temporal context over K frames
        from wmca.modules.mamba_block import MinimalMambaBlock

        self.mamba = MinimalMambaBlock(
            d_model=embed_dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Project Mamba output (+ optional action) → 1ch CML input
        mamba_out_ch = embed_dim + (embed_dim if self.action_embed else 0)
        self.input_proj = nn.Conv2d(mamba_out_ch, 1, 1)

        # CML reservoir
        self.cml = CML2DMultiR(
            in_channels=1,
            K=cml_K,
            steps=cml_steps,
            r_lo=r_lo,
            r_hi=r_hi,
            seed=seed,
            gate_mode="uniform",
        )

        # NCA correction
        self.nca = nn.Sequential(
            nn.Conv2d(1 + 1, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, embed_dim, 1),
        )

        # CML output projection: 1ch → embed_dim (learned)
        self.cml_proj = nn.Conv2d(1, embed_dim, 1)

        # Output head
        self.output_head = nn.Conv2d(embed_dim, vocab_size, 1)

    def forward(
        self,
        token_history: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with K-frame token history.

        Args:
            token_history: (B, K, H, W) long — K consecutive token grids.
            action: (B,) long — current action index.

        Returns:
            logits: (B, V, H, W) raw logits over vocabulary.
        """
        if self.use_sigmoid:
            raise ValueError("DiscreteRescorMamba: set use_sigmoid=False for CE loss.")

        B, K, H, W = token_history.shape
        assert K == self.mamba_context, f"Expected K={self.mamba_context}, got {K}"

        # Embed tokens: (B, K, H, W) → (B, K, H*W, embed_dim)
        tok_flat = token_history.view(B, K, -1)                                # (B, K, H*W)
        tok_emb = self.token_embed(tok_flat)                                   # (B, K, H*W, embed_dim)

        # Reshape for Mamba: (B * H*W, K, embed_dim)
        tok_seq = tok_emb.permute(0, 2, 1, 3).contiguous()                     # (B, H*W, K, embed_dim)
        tok_seq = tok_seq.view(B * H * W, K, self.embed_dim)

        # Mamba scan over K temporal steps
        mamba_out = self.mamba(tok_seq)                                         # (B*H*W, K, embed_dim)
        mamba_last = mamba_out[:, -1, :]                                       # (B*H*W, embed_dim)
        mamba_last = mamba_last.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)  # (B, embed_dim, H, W)

        # Concat with action embedding
        if self.action_embed is not None and action is not None:
            act_emb = self.action_embed(action)                                # (B, embed_dim)
            act_emb = act_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            combined = torch.cat([mamba_last, act_emb], dim=1)
        else:
            combined = mamba_last

        # Project to 1ch CML input
        cml_input = torch.sigmoid(self.input_proj(combined))

        # CML forward
        cml_out = self.cml(cml_input)

        # NCA correction
        nca_in = torch.cat([cml_input, cml_out], dim=1)
        correction = self.nca(nca_in)

        # CML output projection (learned)
        cml_emb = self.cml_proj(cml_out)

        # Residual + output
        embedding = correction + cml_emb
        logits = self.output_head(embedding)

        return logits

    def param_count(self) -> dict[str, int]:
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(b.numel() for b in self.buffers())
        return {"trained": trained, "frozen": frozen}

    def get_alpha_params(self) -> list[nn.Parameter]:
        return []
