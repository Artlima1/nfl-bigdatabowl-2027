"""PlayToVec: encode a player's movement on one play into a fixed-size embedding.

Input is the kinematic tensor from notebooks/DataTransformation.ipynb:
    X    (B, T, C) float32, t = 0 at the snap, 10 Hz, zero-padded
    mask (B, T)    bool, True where a frame exists
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUEncoder(nn.Module):
    """GRU over a masked movement track -> one embedding per track.

    Each valid frame is projected to `hidden_dim`, then a (optionally bidirectional)
    GRU runs over the track up to its last valid frame (packed, so trailing padding
    is never seen). The final hidden state of the top layer (both directions
    concatenated if bidirectional) is projected to `embed_dim`.

    The mask is also fed as an input channel, so the GRU can tell an interior
    missing frame (hole left by `align_frames`) from a real zero reading.
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels + 1, hidden_dim),  # +1 for the mask channel
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        n_dirs = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(n_dirs * hidden_dim),
            nn.Linear(n_dirs * hidden_dim, embed_dim),
        )
        self.n_dirs = n_dirs
        self.embed_dim = embed_dim

    @staticmethod
    def lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
        """Length = index of the last valid frame + 1 (at least 1, so empty tracks don't break packing)."""
        T = mask.shape[1]
        idx = torch.arange(1, T + 1, device=mask.device).expand_as(mask)
        return (idx * mask).amax(dim=1).clamp(min=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C), mask: (B, T) bool -> (B, embed_dim)."""
        m = mask.unsqueeze(-1).to(x.dtype)
        h = self.input_proj(torch.cat([x * m, m], dim=-1))

        lengths = self.lengths_from_mask(mask).cpu()
        packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)  # (num_layers * n_dirs, B, hidden_dim)

        last = h_n[-self.n_dirs:]  # top layer, all directions
        last = last.transpose(0, 1).reshape(x.shape[0], -1)  # (B, n_dirs * hidden_dim)
        return self.head(last)


class MLPFrameDecoder(nn.Module):
    """Small MLP that reconstructs a frame from the track embedding and the frame's time index.

    For each queried frame t, the input is [z, time_emb(t)] and the output is the
    `out_channels` values at that frame. Because the decoder sees only z and t (no
    neighbouring frames), everything about the track has to pass through z, which
    is what pushes the encoder to put the movement into the embedding.

    Training use: hide some valid frames from the encoder (set them False in its
    mask), encode, then ask the decoder for exactly those hidden frames.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        out_channels: int = 9,
        max_len: int = 50,
        time_dim: int = 16,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.time_emb = nn.Embedding(max_len, time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )
        self.max_len = max_len

    def forward(self, z: torch.Tensor, t_idx: torch.Tensor | None = None) -> torch.Tensor:
        """z: (B, E); t_idx: (B, K) long frame indices, or None for all frames -> (B, K, C)."""
        B = z.shape[0]
        if t_idx is None:
            t_idx = torch.arange(self.max_len, device=z.device).expand(B, -1)
        K = t_idx.shape[1]
        h = torch.cat([z.unsqueeze(1).expand(B, K, -1), self.time_emb(t_idx)], dim=-1)
        return self.mlp(h)


def masked_recon_step(enc, dec, x, mask, hidden) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked-frame objective: encode visible frames, decode all frames, score the hidden ones.

    Returns (loss, per-frame squared error (B, T, C)). Lives here, not in train.py,
    so both train.py and eval.py can use it without importing each other.
    """
    z = enc(x, mask & ~hidden)
    pred = dec(z)  # (B, T, C)
    sq_err = (pred - x) ** 2
    w = hidden.unsqueeze(-1).to(x.dtype)
    loss = (sq_err * w).sum() / (w.sum() * x.shape[-1]).clamp(min=1)
    return loss, sq_err
