"""Save / load a PlayToVec checkpoint (weights + everything needed to rebuild and reuse the model)."""

import torch

from PlayToVec.models.PlayToVec import GRUEncoder, MLPFrameDecoder


def model_config(in_channels: int, max_len: int, embed_dim: int, enc_hidden: int, dec_hidden: int, **enc_kwargs) -> dict:
    return {"in_channels": in_channels, "max_len": max_len, "embed_dim": embed_dim,
            "enc_hidden": enc_hidden, "dec_hidden": dec_hidden, "enc_kwargs": enc_kwargs}


def build_from_config(cfg: dict) -> tuple[GRUEncoder, MLPFrameDecoder]:
    enc = GRUEncoder(in_channels=cfg["in_channels"], hidden_dim=cfg["enc_hidden"],
                     embed_dim=cfg["embed_dim"], **cfg["enc_kwargs"])
    dec = MLPFrameDecoder(embed_dim=cfg["embed_dim"], out_channels=cfg["in_channels"],
                          max_len=cfg["max_len"], hidden_dim=cfg["dec_hidden"])
    return enc, dec


def _cpu_state(m) -> dict:
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


def save_checkpoint(ckpt_path: str, enc, dec, data: dict, model_cfg: dict,
                    data_cfg: dict | None = None, history: dict | None = None) -> None:
    """`model_cfg` rebuilds the networks; `data_cfg` holds the make_datasets arguments (seed, fractions)."""
    torch.save({
        "enc": _cpu_state(enc),
        "dec": _cpu_state(dec),
        "model_config": model_cfg,
        "data_config": data_cfg or {},
        "stats": {"mean": data["stats"].mean, "std": data["stats"].std},
        "channels": data["channels"],
        "splits": data["splits"],
        "history": history,
    }, ckpt_path)


def load_model(ckpt_path: str, device: str = "cpu") -> tuple[GRUEncoder, MLPFrameDecoder, dict]:
    """Rebuild encoder/decoder from the stored config and load the weights. Returns (enc, dec, checkpoint)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc, dec = build_from_config(ck["model_config"])
    enc.load_state_dict(ck["enc"])
    dec.load_state_dict(ck["dec"])
    return enc.to(device).eval(), dec.to(device).eval(), ck
