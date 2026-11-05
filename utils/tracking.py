"""Preprocess in-game tracking into a padded movement tensor (notebooks/KinematicModel.ipynb).

Pipeline per season: load frames -> join play filters / play_direction -> temporal
alignment (sorted, t=0 at ball_snap, pre-snap dropped) -> geometric normalization
(every play moves toward +x, origin at the snap position) -> padded (N, T, C) tensor.
"""
from os import path

import numpy as np
import pandas as pd
import torch

KEYS = ["game_id", "play_id", "nfl_id"]
SKILL_POSITIONS = ("WR", "TE", "RB", "FB", "CB", "SS", "FS", "DB")
FIELD_LENGTH, FIELD_WIDTH = 120.0, 53.3
DT = 0.1  # seconds between frames (10 Hz)
CHANNELS = ["x_rel", "y_rel", "s", "a", "dis", "dir_sin", "dir_cos", "o_sin", "o_cos"]

_DTYPES = {"game_id": "int64", "play_id": "int32", "nfl_id": "int32",
           **{c: "float32" for c in ("x", "y", "s", "a", "dis", "o", "dir")}}


def load_tracking(data_dir: str, season: int, nfl_ids) -> pd.DataFrame:
    """One season of game tracking, restricted to `nfl_ids`, with `time` parsed."""
    df = pd.read_csv(path.join(data_dir, f"game_tracking_{season}.csv"), dtype=_DTYPES)
    df = df[df["nfl_id"].isin(nfl_ids)].copy()
    df["time"] = pd.to_datetime(df["time"])
    return df


def align_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Strict temporal order per track, t=0 at the ball_snap frame, pre-snap frames dropped.

    `frame` = round(t / DT) is the time index used in the tensor, so a missing
    raw frame leaves a hole (masked) instead of shifting the rest of the track.
    """
    df = df.sort_values(KEYS + ["time"], kind="stable").reset_index(drop=True)

    snap = df.loc[df["event"] == "ball_snap"].groupby(KEYS)["time"].min().rename("snap_time")
    df = df.join(snap, on=KEYS, how="inner")  # drops tracks without a snap

    df["t"] = (df["time"] - df["snap_time"]).dt.total_seconds()
    df = df[df["t"] >= 0].drop(columns="snap_time").reset_index(drop=True)
    df["frame"] = np.rint(df["t"] / DT).astype("int32")

    same_track = (df[KEYS].shift() == df[KEYS]).all(axis=1)
    assert (df["frame"].diff()[same_track] > 0).all(), "frames not strictly increasing within a track"
    return df


def normalize_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Rotate `left` plays 180 deg so all plays move toward +x, then put the snap position at (0, 0).

    Requires a `play_direction` column ("left"/"right"); tracks without it are dropped.
    Angles follow the NGS convention (0 deg = +y, clockwise), so a 180 deg rotation adds 180.
    """
    df = df[df["play_direction"].isin(["left", "right"])].copy()
    left = df["play_direction"] == "left"
    df.loc[left, "x"] = FIELD_LENGTH - df.loc[left, "x"]
    df.loc[left, "y"] = FIELD_WIDTH - df.loc[left, "y"]
    for col in ("dir", "o"):
        df.loc[left, col] = (df.loc[left, col] + 180) % 360

    # after align_frames, the first row of each track is the snap frame (t = 0)
    start = df.groupby(KEYS, sort=False)[["x", "y"]].transform("first")
    df["x_rel"] = df["x"] - start["x"]
    df["y_rel"] = df["y"] - start["y"]
    return df


def build_tracking_series(
    data_dir: str,
    seasons,
    plays_df: pd.DataFrame,
    players_df: pd.DataFrame,
    positions=SKILL_POSITIONS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Long-format, aligned and normalized frames for skill players across `seasons`.

    Keeps every season type (preseason included); drops plays nullified by penalty.
    """
    nfl_ids = players_df.loc[players_df["nfl_position"].isin(positions), "nfl_id"].unique()
    plays = plays_df.loc[
        plays_df["play_nullified_by_penalty"] == "N", KEYS + ["play_direction"]
    ].drop_duplicates(KEYS)

    out = []
    for season in seasons:
        raw = load_tracking(data_dir, season, nfl_ids)
        n_raw = raw.groupby(KEYS).ngroups
        df = raw.merge(plays, on=KEYS, how="inner")
        n_plays = df.groupby(KEYS).ngroups
        df = align_frames(df)
        n_snap = df.groupby(KEYS).ngroups
        df = normalize_geometry(df)
        n_dir = df.groupby(KEYS).ngroups
        if verbose:
            print(f"{season}: {n_raw} skill tracks -> {n_plays} not nullified -> "
                  f"{n_snap} with snap -> {n_dir} with play_direction")
        out.append(df)
        del raw
    return pd.concat(out, ignore_index=True)


def to_tensor(series_df: pd.DataFrame, max_len: int = 100) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Pack frames into X (N, max_len, C) float32 and mask (N, max_len) bool; zero-padded, truncated at max_len."""
    df = series_df[series_df["frame"] < max_len]
    track = df.groupby(KEYS, sort=True).ngroup().to_numpy()
    frame = df["frame"].to_numpy()

    dir_rad, o_rad = np.deg2rad(df["dir"].to_numpy()), np.deg2rad(df["o"].to_numpy())
    values = np.column_stack([
        df["x_rel"], df["y_rel"], df["s"], df["a"], df["dis"],
        np.sin(dir_rad), np.cos(dir_rad), np.sin(o_rad), np.cos(o_rad),
    ]).astype(np.float32)

    n = track.max() + 1
    X = np.zeros((n, max_len, len(CHANNELS)), dtype=np.float32)
    mask = np.zeros((n, max_len), dtype=bool)
    X[track, frame] = values
    mask[track, frame] = True
    return torch.from_numpy(X), torch.from_numpy(mask), list(CHANNELS)
