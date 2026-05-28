"""Entity-transformer policy + value network for Orbit Wars.

Inputs:
  entities:  (B, N, E)   per-entity feature vectors, E=ENTITY_DIM
  mask:      (B, N) bool — True = real entity, False = padding
  globals_:  (B, G)      global feature vector

Outputs (per-step action sampling happens externally via sample_action):
  - per-owned-planet logits:
      launch_logit   (B, N)         emit fleet from this planet this turn?
      target_logits  (B, N, N)      which entity to target (mask out self & invalid)
      fraction_logits(B, N, F)      fraction-of-garrison bin (F=5)
  - value:           (B,) scalar V(s)

The action is factored geometrically: angle = atan2(target.y - src.y,
target.x - src.x) computed outside the net, with optional orbital
prediction for moving targets. This embeds geometric reasoning into the
architecture and drastically shrinks the action space.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.features import ENTITY_DIM, GLOBAL_DIM, MAX_ENTITIES


SHIP_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 0.95)
N_FRACTIONS = len(SHIP_FRACTIONS)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: (B, N) bool — True for valid. PyTorch wants True for "padding/ignore".
        return self.encoder(x, src_key_padding_mask=~mask)


class OrbitWarsPolicy(nn.Module):
    """Entity transformer with pointer-style target selection."""

    def __init__(
        self,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        global_dim: int = GLOBAL_DIM,
        entity_dim: int = ENTITY_DIM,
        n_fractions: int = N_FRACTIONS,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_fractions = n_fractions

        self.entity_proj = nn.Sequential(
            nn.Linear(entity_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers)

        # Action heads (applied per-entity, only used for owned planets)
        self.launch_head = nn.Linear(d_model, 1)
        self.fraction_head = nn.Linear(d_model, n_fractions)

        # Pointer head: query (src) vs keys (targets). Bilinear scoring.
        self.target_query = nn.Linear(d_model, d_model)
        self.target_key = nn.Linear(d_model, d_model)

        # Value head: pool entities (masked mean) + global token → scalar
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, entities: torch.Tensor, mask: torch.Tensor, globals_: torch.Tensor):
        """Returns dict with per-entity logits + value.

        Shapes (B=batch, N=entities, F=fractions):
          launch_logit: (B, N)
          target_logits: (B, N, N)
          fraction_logits: (B, N, F)
          value: (B,)
        """
        # Project entities + add global as a learned bias (broadcast into each token)
        x = self.entity_proj(entities)  # (B, N, d)
        g = self.global_proj(globals_)  # (B, d)
        x = x + g.unsqueeze(1)
        h = self.encoder(x, mask)  # (B, N, d)

        launch_logit = self.launch_head(h).squeeze(-1)  # (B, N)
        fraction_logits = self.fraction_head(h)         # (B, N, F)

        q = self.target_query(h)  # (B, N, d)
        k = self.target_key(h)    # (B, N, d)
        # target_logits[b, i, j] = <q[b,i], k[b,j]> / sqrt(d)
        target_logits = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(self.d_model)

        # Mask out invalid keys (padding columns)
        # (B, N) → (B, 1, N)
        key_pad = ~mask  # True = invalid
        target_logits = target_logits.masked_fill(key_pad.unsqueeze(1), float("-inf"))
        # Also mask self-loops (i == j)
        diag = torch.eye(target_logits.shape[1], device=target_logits.device, dtype=torch.bool)
        target_logits = target_logits.masked_fill(diag.unsqueeze(0), float("-inf"))

        # Value: masked mean pool over real entities
        m = mask.float().unsqueeze(-1)  # (B, N, 1)
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)
        value = self.value_head(pooled).squeeze(-1)

        return {
            "launch_logit": launch_logit,
            "target_logits": target_logits,
            "fraction_logits": fraction_logits,
            "value": value,
        }
