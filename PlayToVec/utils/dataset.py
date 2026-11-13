"""Dataset for PlayToVec: load the kinematic tensor, split by game, standardize, sample hidden frames."""

from torch.utils.data import DataLoader, TensorDataset
import torch
from dataclasses import dataclass
import pandas as pd
import numpy as np
from os import path

@dataclass
class ChannelStats:
    """Per-channel mean/std over valid frames, fit on the training split only."""
    mean: torch.Tensor  # (C,)
    std: torch.Tensor   # (C,)

    def normalize(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        Xn = (X - self.mean) / self.std
        return Xn * mask.unsqueeze(-1)  # keep padding / holes at exactly 0

    def denormalize(self, Xn: torch.Tensor) -> torch.Tensor:
        return Xn * self.std.to(Xn.device) + self.mean.to(Xn.device)


def load_tracking_tensor(kin_dir: str) -> tuple[torch.Tensor, torch.Tensor, list[str], pd.DataFrame]:
    """Load X, mask, channels and the row index written by DataTransformation.ipynb."""
    d = torch.load(path.join(kin_dir, "tracking_tensor.pt"))
    index = pd.read_parquet(path.join(kin_dir, "tracking_index.parquet"))
    assert len(index) == d["X"].shape[0], "tensor and index are out of sync"
    return d["X"], d["mask"], list(d["channels"]), index


def split_by_game(
    index: pd.DataFrame,
    keep: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Row indices per split, grouped by game_id so plays from one game never straddle splits."""
    games = np.sort(index["game_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val, n_test = int(len(games) * val_frac), int(len(games) * test_frac)
    split_of_game = {g: "test" for g in games[:n_test]}
    split_of_game.update({g: "val" for g in games[n_test:n_test + n_val]})
    split = index["game_id"].map(split_of_game).fillna("train").to_numpy()
    return {s: np.flatnonzero((split == s) & keep) for s in ("train", "val", "test")}


def fit_channel_stats(X: torch.Tensor, mask: torch.Tensor) -> ChannelStats:
    v = X[mask]  # (n_valid_frames, C)
    return ChannelStats(mean=v.mean(0), std=v.std(0).clamp(min=1e-6))


def make_datasets(
    kin_dir: str,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    min_frames: int = 5,
    seed: int = 0,
) -> dict:
    """Load the tensor, split by game, standardize with train stats.

    Tracks with fewer than `min_frames` valid frames are left out of every split
    (too short to mask anything meaningful). Returns a dict with the normalized
    tensor for *all* rows (so `embed` can cover the full index), the split row
    indices, the stats and the metadata.
    """
    X, mask, channels, index = load_tracking_tensor(kin_dir)
    keep = (mask.sum(1) >= min_frames).numpy()
    splits = split_by_game(index, keep, val_frac, test_frac, seed)
    stats = fit_channel_stats(X[splits["train"]], mask[splits["train"]])
    return {
        "X_norm": stats.normalize(X, mask),
        "mask": mask,
        "splits": splits,
        "stats": stats,
        "channels": channels,
        "index": index,
    }


def make_loader(data: dict, split: str, batch_size: int = 512, shuffle: bool | None = None) -> DataLoader:
    idx = torch.from_numpy(data["splits"][split])
    ds = TensorDataset(data["X_norm"][idx], data["mask"][idx])
    shuffle = (split == "train") if shuffle is None else shuffle
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def sample_hidden_frames(mask: torch.Tensor, mask_ratio: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Pick ~mask_ratio of the valid frames to hide. Always leaves >= 1 visible frame per track."""
    r = torch.rand(mask.shape, generator=generator, device="cpu").to(mask.device)
    hidden = mask & (r < mask_ratio)
    all_hidden = (mask & ~hidden).sum(1) == 0
    if all_hidden.any():
        first_valid = mask.float().argmax(1)
        rows = all_hidden.nonzero(as_tuple=True)[0]
        hidden[rows, first_valid[rows]] = False
    return hidden
