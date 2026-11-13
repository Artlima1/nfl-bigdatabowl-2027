"""Build the kinematic tensor from raw tracking: game_tracking_*.csv -> data/processed/kinematics/.

    python PlayToVec/preprocess.py                 # skips if the tensor already exists
    python PlayToVec/preprocess.py --force --max-len 60

Writes tracking_tensor.pt (X, mask, channels; no per-sample metadata),
tracking_index.parquet (game_id, play_id, nfl_id per row) and preprocess_config.json.
The visual checks behind these steps are in notebooks/DataTransformation.ipynb.
"""

import argparse
import json
import sys
from datetime import datetime
from os import path
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PlayToVec.utils import tracking as trk  # noqa: E402
from PlayToVec.utils.paths import KIN_DIR  # noqa: E402

COMPETITION = "nfl-big-data-bowl-2027"


def default_data_dir() -> str:
    import kagglehub

    return path.join(kagglehub.competition_download(COMPETITION), COMPETITION)


def preprocess(data_dir: str, seasons, max_len: int, out_dir: str, positions=trk.SKILL_POSITIONS) -> dict:
    players_df, plays_df = trk.load_play_tables(data_dir)
    series_df = trk.build_tracking_series(data_dir, seasons, plays_df, players_df, positions=positions)

    X, mask, channels = trk.to_tensor(series_df, max_len=max_len)
    trk.check_tensor(X, mask, channels, max_len)
    index_df = trk.track_index(series_df)
    out_path = trk.save_tensor(out_dir, X, mask, channels, index_df)

    config = {
        "seasons": list(seasons),
        "max_len": max_len,
        "positions": list(positions),
        "channels": channels,
        "n_tracks": int(X.shape[0]),
        "padding": round(1 - mask.float().mean().item(), 4),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(path.join(out_dir, "preprocess_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"saved {out_path}: X {tuple(X.shape)}, padding {config['padding']:.1%}, "
          f"{X.numel() * 4 / 1e9:.2f} GB")
    return config


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="raw competition files (default: kagglehub cache)")
    p.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025])
    p.add_argument("--max-len", type=int, default=50, help="frames kept after the snap (10 Hz)")
    p.add_argument("--out-dir", default=str(KIN_DIR))
    p.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    args = p.parse_args(argv)

    if not args.force and path.exists(path.join(args.out_dir, "tracking_tensor.pt")):
        print(f"{args.out_dir}/tracking_tensor.pt exists; skipping (use --force to rebuild)")
        return
    preprocess(args.data_dir or default_data_dir(), args.seasons, args.max_len, args.out_dir)


if __name__ == "__main__":
    main()
