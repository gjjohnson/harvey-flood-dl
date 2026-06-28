"""
evaluate.py
-----------
Usage:
    python evaluate.py --config config.yaml --checkpoint models/harvey_lstm.pt

Loads a trained checkpoint, runs it over the held-out test split, and
reports, per zone:
  - RMSE / MAE on forecasted gage height (in feet, de-normalized)
  - precision/recall/F1 on flood-stage exceedance

Also saves a predicted-vs-actual hydrograph plot per zone to
outputs/hydrograph_<zone>.png so you can visually sanity-check the
forecasts against the real Harvey timeline.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from dataset import Normalizer, load_config, make_datasets, time_split
from model import build_model


def denorm(values, mean, std):
    return values * std + mean


def evaluate_zone_hydrograph(model, zone_name, df, zone_cfg, normalizer, cfg, device):
    """
    Roll the model forward over a zone's full timeline using *only the
    horizon-0 prediction* at each step (i.e. plot the model's one-step-ahead
    forecast continuously across the whole record) so we get a clean
    predicted-vs-actual hydrograph rather than horizon spaghetti.
    """
    lookback = cfg["windows"]["lookback_hours"]
    horizon = cfg["windows"]["horizon_hours"]

    rain = (df["rainfall_in"].values - normalizer.rain_mean) / normalizer.rain_std
    wind = (df["wind_speed_mph"].values - normalizer.wind_mean) / normalizer.wind_std
    level = df["gage_height_ft"].values
    n = len(df)

    preds_first_hour = np.full(n, np.nan)
    flood_prob_first_hour = np.full(n, np.nan)
    model.eval()
    with torch.no_grad():
        for start in range(0, n - lookback - 1):
            x = np.stack([rain[start:start + lookback], wind[start:start + lookback]], axis=-1)
            x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
            zone_id_t = torch.tensor([zone_cfg["id"]], dtype=torch.long).to(device)
            level_pred, flood_logits = model(x_t, zone_id_t)
            pred_ft = denorm(level_pred[0, 0].item(), normalizer.level_mean, normalizer.level_std)
            preds_first_hour[start + lookback] = pred_ft
            flood_prob_first_hour[start + lookback] = torch.sigmoid(flood_logits[0, 0]).item()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df.index, level, label="actual gage height", color="black", linewidth=1.2)
    ax1.plot(df.index, preds_first_hour, label="model forecast (+1h)", color="tab:red", linewidth=1.2)
    ax1.axhline(zone_cfg["flood_stage_ft"], color="orange", linestyle="--",
                label=f"flood stage ({zone_cfg['flood_stage_ft']} ft)")
    tag = " [SYNTHETIC DATA]" if df["synthetic"].iloc[0] else ""
    ax1.set_title(f"{zone_cfg['label']}{tag}")
    ax1.set_ylabel("gage height (ft)")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.bar(df.index, df["rainfall_in"], width=0.03, color="tab:blue", label="rainfall (in/hr)")
    ax2.set_ylabel("rainfall (in/hr)")
    ax2.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/hydrograph_{zone_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(config_path="config.yaml", checkpoint_path=None):
    cfg = load_config(config_path)
    checkpoint_path = checkpoint_path or cfg["train"]["checkpoint_path"]
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]  # use the exact config the model was trained with
    normalizer = Normalizer.from_dict(ckpt["normalizer"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, num_zones=len(cfg["zones"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    train_ds, val_ds, test_ds, _, zone_frames = make_datasets(cfg)
    loader_args = dict(batch_size=cfg["train"]["batch_size"], shuffle=False)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, **loader_args)

    all_level_true, all_level_pred = [], []
    all_flood_true, all_flood_pred = [], []
    all_zone_ids = []

    with torch.no_grad():
        for x, zone_id, y_level, y_flood in test_loader:
            x, zone_id = x.to(device), zone_id.to(device)
            level_pred, flood_logits = model(x, zone_id)
            all_level_true.append(y_level.numpy())
            all_level_pred.append(level_pred.cpu().numpy())
            all_flood_true.append(y_flood.numpy())
            all_flood_pred.append((torch.sigmoid(flood_logits) > 0.5).cpu().numpy().astype(float))
            all_zone_ids.append(zone_id.cpu().numpy())

    level_true = denorm(np.concatenate(all_level_true), normalizer.level_mean, normalizer.level_std)
    level_pred = denorm(np.concatenate(all_level_pred), normalizer.level_mean, normalizer.level_std)
    flood_true = np.concatenate(all_flood_true)
    flood_pred = np.concatenate(all_flood_pred)
    zone_ids = np.concatenate(all_zone_ids)

    print("\n=== Test-set metrics by zone ===")
    rows = []
    for zone_cfg in cfg["zones"]:
        mask = zone_ids == zone_cfg["id"]
        if mask.sum() == 0:
            continue
        rmse = float(np.sqrt(np.mean((level_true[mask] - level_pred[mask]) ** 2)))
        mae = float(np.mean(np.abs(level_true[mask] - level_pred[mask])))
        ft, fp = flood_true[mask].ravel(), flood_pred[mask].ravel()
        if ft.sum() > 0:
            prec = precision_score(ft, fp, zero_division=0)
            rec = recall_score(ft, fp, zero_division=0)
            f1 = f1_score(ft, fp, zero_division=0)
        else:
            prec, rec, f1 = float("nan"), float("nan"), float("nan")
        rows.append(dict(zone=zone_cfg["name"], rmse_ft=rmse, mae_ft=mae,
                          flood_precision=prec, flood_recall=rec, flood_f1=f1))
        print(f"  {zone_cfg['label']:35s}  RMSE={rmse:5.2f} ft  MAE={mae:5.2f} ft  "
              f"flood F1={f1:.2f} (P={prec:.2f} R={rec:.2f})")

    results_df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/test_metrics.csv", index=False)
    print("\nSaved outputs/test_metrics.csv")

    # Pooled-across-zone classification metrics. With only a single storm
    # event, a zone's time-based test window can legitimately contain zero
    # flood-stage crossings (the recession may have fully resolved before
    # the held-out window for that particular zone) -- this is a real
    # limitation of single-event evaluation, not a bug. Pooling across
    # zones gives one well-defined number whenever *any* zone's test window
    # contains a crossing.
    ft_all, fp_all = flood_true.ravel(), flood_pred.ravel()
    if ft_all.sum() > 0:
        pooled_f1 = f1_score(ft_all, fp_all, zero_division=0)
        pooled_p = precision_score(ft_all, fp_all, zero_division=0)
        pooled_r = recall_score(ft_all, fp_all, zero_division=0)
        print(f"\nPooled (all zones) flood-exceedance F1={pooled_f1:.2f} "
              f"(precision={pooled_p:.2f}, recall={pooled_r:.2f})")
    else:
        print("\nNo flood-stage crossings anywhere in the pooled test set — "
              "classification head not evaluable on this split.")
    print("Note: per-zone flood metrics above show 'nan' for zones whose "
          "test window happens to contain no flood-stage crossings -- this "
          "is expected with a single storm event split by time, not an error.")

    print("\nGenerating hydrograph plots (predicted vs actual) per zone...")
    for zone_cfg in cfg["zones"]:
        path = evaluate_zone_hydrograph(
            model, zone_cfg["name"], zone_frames[zone_cfg["name"]], zone_cfg, normalizer, cfg, device
        )
        print(f"  saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
