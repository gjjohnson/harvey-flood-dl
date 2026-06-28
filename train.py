"""
train.py
--------
Usage:
    python train.py --config config.yaml

Trains the ZoneConditionedLSTM on the time-based train split, validates each
epoch, saves the best checkpoint (by validation loss) to
`models/harvey_lstm.pt`, and writes a training-curve plot to
`outputs/training_curves.png`.

The checkpoint bundles the model weights, the config, and the fitted
Normalizer, so evaluate.py / inference.py / demo.py never need to refit
normalization stats or guess hyperparameters -- everything needed to use
the model is in one file.
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import load_config, make_datasets
from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg):
    if cfg["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg["device"])


def run_epoch(model, loader, device, optimizer=None, flood_weight=0.3):
    """If optimizer is None, runs in eval mode (no grad, no weight update)."""
    is_train = optimizer is not None
    model.train(is_train)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    total_loss, total_mse, total_bce, n_batches = 0.0, 0.0, 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, zone_id, y_level, y_flood in loader:
            x, zone_id = x.to(device), zone_id.to(device)
            y_level, y_flood = y_level.to(device), y_flood.to(device)

            level_pred, flood_logits = model(x, zone_id)
            loss_mse = mse(level_pred, y_level)
            loss_bce = bce(flood_logits, y_flood)
            loss = loss_mse + flood_weight * loss_bce

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_bce += loss_bce.item()
            n_batches += 1

    n_batches = max(n_batches, 1)
    return total_loss / n_batches, total_mse / n_batches, total_bce / n_batches


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg)
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds, normalizer, _ = make_datasets(cfg)
    print(f"Train windows: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    model = build_model(cfg, num_zones=len(cfg["zones"])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    history = {"train_loss": [], "val_loss": [], "val_mse": [], "val_bce": []}
    best_val_loss = float("inf")
    ckpt_path = cfg["train"]["checkpoint_path"]
    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, device, optimizer, cfg["train"]["flood_loss_weight"]
        )
        val_loss, val_mse, val_bce = run_epoch(
            model, val_loader, device, optimizer=None, flood_weight=cfg["train"]["flood_loss_weight"]
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)
        history["val_bce"].append(val_bce)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "normalizer": normalizer.to_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)

        if epoch == 1 or epoch % 5 == 0 or epoch == cfg["train"]["epochs"]:
            print(f"epoch {epoch:3d}/{cfg['train']['epochs']}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"(val_mse={val_mse:.4f} val_bce={val_bce:.4f})")

    print(f"\nBest val_loss={best_val_loss:.4f} -> saved to {ckpt_path}")

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE + flood BCE)")
    plt.title("Training curves — Harvey flood forecaster")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150)
    print("Saved outputs/training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
