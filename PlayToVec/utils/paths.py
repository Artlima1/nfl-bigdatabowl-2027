"""Default locations, resolved from this file so scripts work from any working directory."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KIN_DIR = REPO / "data" / "processed" / "kinematics"
MODEL_DIR = REPO / "data" / "processed" / "playtovec"
