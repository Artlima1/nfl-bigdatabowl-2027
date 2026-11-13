"""Train PlayToVec (GRU encoder + MLP frame decoder) with masked-frame reconstruction.

    python PlayToVec/train.py                      # skips if the checkpoint already exists
    python PlayToVec/train.py --force --epochs 50 --embed-dim 32

Reads data/processed/kinematics/ (run preprocess.py first) and writes the best
checkpoint (by val loss) to data/processed/playtovec/model.pt plus history.json.
The functions can also be imported from a notebook: build_model, train.
"""

import argparse
import json
import sys
from os import path
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PlayToVec.eval import evaluate  # noqa: E402
from PlayToVec.models.PlayToVec import masked_recon_step  # noqa: E402
from PlayToVec.utils.checkpoint import _cpu_state, build_from_config, model_config, save_checkpoint  # noqa: E402
from PlayToVec.utils.dataset import make_datasets, make_loader, sample_hidden_frames  # noqa: E402
from PlayToVec.utils.paths import KIN_DIR, MODEL_DIR  # noqa: E402


def build_model(data: dict, embed_dim: int = 64, enc_hidden: int = 128, dec_hidden: int = 128, **enc_kwargs):
    """Encoder/decoder sized to the data. Returns (enc, dec, config); keep config for save_checkpoint."""
    cfg = model_config(len(data["channels"]), data["X_norm"].shape[1], embed_dim, enc_hidden, dec_hidden, **enc_kwargs)
    enc, dec = build_from_config(cfg)
    return enc, dec, cfg


def train(
    enc,
    dec,
    data: dict,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    mask_ratio: float = 0.5,
    grad_clip: float = 1.0,
    patience: int = 5,
    device: str | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """AdamW + cosine LR, early stopping on val loss. Best weights are restored at the end.

    Validation uses a fixed masking seed every epoch so val losses are comparable.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc.to(device)
    dec.to(device)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    train_loader = make_loader(data, "train", batch_size)
    g = torch.Generator().manual_seed(seed)

    history = {"train_loss": [], "val_loss": []}
    best, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        enc.train()
        dec.train()
        total, n = 0.0, 0
        for x, mask in train_loader:
            x, mask = x.to(device), mask.to(device)
            hidden = sample_hidden_frames(mask, mask_ratio, g)
            loss, _ = masked_recon_step(enc, dec, x, mask, hidden)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            total += loss.item() * len(x)
            n += len(x)
        sched.step()

        val_loss = evaluate(enc, dec, data, "val", mask_ratio, batch_size, device, seed)["loss"]
        history["train_loss"].append(total / n)
        history["val_loss"].append(val_loss)
        if verbose:
            print(f"epoch {epoch + 1:3d}  train {total / n:.4f}  val {val_loss:.4f}", flush=True)

        if val_loss < best - 1e-5:
            best, bad_epochs = val_loss, 0
            best_state = {"enc": _cpu_state(enc), "dec": _cpu_state(dec)}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch + 1}, best val {best:.4f}")
                break

    enc.load_state_dict(best_state["enc"])
    dec.load_state_dict(best_state["dec"])
    return history


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kin-dir", default=str(KIN_DIR))
    p.add_argument("--out", default=str(MODEL_DIR / "model.pt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--enc-hidden", type=int, default=128)
    p.add_argument("--dec-hidden", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--unidirectional", action="store_true")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--min-frames", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="retrain even if the checkpoint exists")
    args = p.parse_args(argv)

    if not args.force and path.exists(args.out):
        print(f"{args.out} exists; skipping (use --force to retrain)")
        return

    data_cfg = {"val_frac": args.val_frac, "test_frac": args.test_frac,
                "min_frames": args.min_frames, "seed": args.seed}
    data = make_datasets(args.kin_dir, **data_cfg)
    print({s: len(i) for s, i in data["splits"].items()})

    enc, dec, model_cfg = build_model(data, args.embed_dim, args.enc_hidden, args.dec_hidden,
                                      num_layers=args.num_layers, bidirectional=not args.unidirectional)
    history = train(enc, dec, data, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    weight_decay=args.weight_decay, mask_ratio=args.mask_ratio, patience=args.patience,
                    device=args.device, seed=args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    train_cfg = {k: getattr(args, k) for k in ("epochs", "batch_size", "lr", "weight_decay", "mask_ratio", "patience")}
    save_checkpoint(args.out, enc, dec, data, model_cfg, data_cfg, history)
    with open(path.join(path.dirname(args.out), "history.json"), "w") as f:
        json.dump({"train_config": train_cfg, **history}, f, indent=2)
    print(f"saved {args.out} (best val {min(history['val_loss']):.4f})")


if __name__ == "__main__":
    main()
