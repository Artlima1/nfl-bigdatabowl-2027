"""Evaluate a trained PlayToVec checkpoint and export embeddings.

    python PlayToVec/eval.py                       # uses data/processed/playtovec/model.pt
    python PlayToVec/eval.py --split val --mask-ratio 0.3

Writes, next to the checkpoint:
    metrics.json        masked-reconstruction loss, per-channel MSE/RMSE, error by frame
    embeddings.parquet  game_id, play_id, nfl_id, split, z_0..z_{E-1} for every tensor row
The functions can also be imported from a notebook: evaluate, reconstruct, embed.
"""

import argparse
import json
import sys
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PlayToVec.models.PlayToVec import masked_recon_step  # noqa: E402
from PlayToVec.utils.checkpoint import load_model  # noqa: E402
from PlayToVec.utils.dataset import make_datasets, make_loader, sample_hidden_frames  # noqa: E402
from PlayToVec.utils.paths import KIN_DIR, MODEL_DIR  # noqa: E402


@torch.no_grad()
def evaluate(
    enc,
    dec,
    data: dict,
    split: str = "test",
    mask_ratio: float = 0.5,
    batch_size: int = 1024,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    """Masked-reconstruction error on hidden frames for one split (fixed masking seed).

    Returns:
        loss            mean MSE over hidden frames and channels (standardized units; = training loss)
        mse_norm        per-channel MSE, standardized units. Predicting the channel mean
                        scores ~1.0, so 1 - mse_norm is roughly the R^2 per channel.
        rmse            per-channel RMSE in original units (yards, yd/s, ...)
        mse_by_frame    loss per frame index (T,), to see how error grows with time after snap
    """
    device = device or next(enc.parameters()).device
    enc.eval()
    dec.eval()
    g = torch.Generator().manual_seed(seed)
    std = data["stats"].std.to(device)
    C, T = len(data["channels"]), data["X_norm"].shape[1]

    se_ch = torch.zeros(C, device=device)
    se_t = torch.zeros(T, device=device)
    n_t = torch.zeros(T, device=device)
    for x, mask in make_loader(data, split, batch_size, shuffle=False):
        x, mask = x.to(device), mask.to(device)
        hidden = sample_hidden_frames(mask, mask_ratio, g)
        _, sq_err = masked_recon_step(enc, dec, x, mask, hidden)
        sq_err = sq_err * hidden.unsqueeze(-1)
        se_ch += sq_err.sum((0, 1))
        se_t += sq_err.sum((0, 2))
        n_t += hidden.sum(0)

    mse_norm = se_ch / n_t.sum().clamp(min=1)
    return {
        "loss": mse_norm.mean().item(),
        "mse_norm": pd.Series(mse_norm.cpu().numpy(), index=data["channels"]),
        "rmse": pd.Series((mse_norm.sqrt() * std).cpu().numpy(), index=data["channels"]),
        "mse_by_frame": (se_t / (n_t * C).clamp(min=1)).cpu().numpy(),
    }


@torch.no_grad()
def reconstruct(enc, dec, data: dict, rows, device: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full (unmasked) reconstruction of specific rows, in original units. For plotting tracks.

    Returns (X_true, X_pred, mask), each with leading dim len(rows).
    """
    device = device or next(enc.parameters()).device
    enc.eval()
    dec.eval()
    rows = torch.as_tensor(rows)
    x, mask = data["X_norm"][rows].to(device), data["mask"][rows].to(device)
    pred = dec(enc(x, mask))
    st = data["stats"]
    return st.denormalize(x).cpu().numpy(), st.denormalize(pred).cpu().numpy(), mask.cpu().numpy()


@torch.no_grad()
def embed(enc, X_norm: torch.Tensor, mask: torch.Tensor, batch_size: int = 2048, device: str | None = None) -> np.ndarray:
    """Embeddings (N, embed_dim) for every row, in tensor order (aligns with tracking_index.parquet).

    Rows with no valid frame get the embedding of an empty track; filter them with mask.any(1).
    """
    device = device or next(enc.parameters()).device
    enc.eval()
    out = []
    for i in range(0, len(X_norm), batch_size):
        out.append(enc(X_norm[i:i + batch_size].to(device), mask[i:i + batch_size].to(device)).cpu())
    return torch.cat(out).numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=str(MODEL_DIR / "model.pt"))
    p.add_argument("--kin-dir", default=str(KIN_DIR))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    enc, dec, ck = load_model(args.ckpt, args.device)
    data = make_datasets(args.kin_dir, **ck["data_config"])
    for s, idx in ck["splits"].items():
        assert np.array_equal(data["splits"][s], idx), f"{s} split differs from the one used in training"
    assert torch.allclose(data["stats"].mean, ck["stats"]["mean"]), "normalization stats differ from training"
    out_dir = path.dirname(args.ckpt)

    m = evaluate(enc, dec, data, args.split, args.mask_ratio, device=args.device)
    metrics = {
        "split": args.split,
        "mask_ratio": args.mask_ratio,
        "loss": m["loss"],
        "mse_norm": m["mse_norm"].round(5).to_dict(),
        "rmse": m["rmse"].round(5).to_dict(),
        "mse_by_frame": [round(float(v), 5) for v in m["mse_by_frame"]],
    }
    with open(path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"{args.split} loss {m['loss']:.4f}  (mean predictor ~ 1.0)")
    print(pd.DataFrame({"mse_norm": m["mse_norm"], "rmse": m["rmse"]}).round(4).to_string())

    Z = embed(enc, data["X_norm"], data["mask"], device=args.device)
    emb = data["index"].copy()
    emb["split"] = "excluded"  # rows below min_frames
    for s, idx in data["splits"].items():
        emb.loc[idx, "split"] = s
    emb = pd.concat([emb, pd.DataFrame(Z, columns=[f"z_{i}" for i in range(Z.shape[1])])], axis=1)
    emb.to_parquet(path.join(out_dir, "embeddings.parquet"), index=False)
    print(f"saved metrics.json and embeddings.parquet {Z.shape} to {out_dir}")


if __name__ == "__main__":
    main()
