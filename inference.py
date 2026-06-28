"""
inference.py
------------
Scenario-based inference: given a hypothetical rainfall/wind scenario and a
zone, predict the gage-height trajectory for the next `horizon` hours.

This is the function the demo calls. It's deliberately decoupled from any
specific historical timeline so you can ask "what if this zone saw 4 in/hr
of rain and 60 mph wind sustained for the last 36 hours" and get a forecast,
not just replay history.

Usage (CLI):
    python inference.py --checkpoint models/harvey_lstm.pt --zone brays_bayou \
        --rainfall 3.5 --wind 45
"""

import argparse

import numpy as np
import torch

from dataset import Normalizer
from model import build_model


def load_checkpoint(checkpoint_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    normalizer = Normalizer.from_dict(ckpt["normalizer"])
    model = build_model(cfg, num_zones=len(cfg["zones"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, cfg, normalizer, device


def zone_lookup(cfg, zone_name):
    for z in cfg["zones"]:
        if z["name"] == zone_name:
            return z
    raise ValueError(f"Unknown zone '{zone_name}'. Options: {[z['name'] for z in cfg['zones']]}")


def predict_scenario(model, cfg, normalizer, device, zone_name,
                      rainfall_series=None, wind_series=None,
                      constant_rainfall=None, constant_wind=None):
    """
    Build a `lookback_hours`-long input window and return:
        hours (1..horizon), level_ft (array), flood_prob (array)

    Either pass explicit `rainfall_series` / `wind_series` arrays of length
    lookback_hours, OR pass `constant_rainfall` (in/hr) / `constant_wind`
    (mph) to hold both flat across the lookback window -- the simplest way
    to ask "what does sustained rain+wind of X look like for this zone".
    """
    zone_cfg = zone_lookup(cfg, zone_name)
    lookback = cfg["windows"]["lookback_hours"]
    horizon = cfg["windows"]["horizon_hours"]

    if rainfall_series is None:
        if constant_rainfall is None:
            raise ValueError("Provide either rainfall_series or constant_rainfall")
        rainfall_series = np.full(lookback, constant_rainfall)
    if wind_series is None:
        if constant_wind is None:
            raise ValueError("Provide either wind_series or constant_wind")
        wind_series = np.full(lookback, constant_wind)

    rainfall_series = np.asarray(rainfall_series, dtype=np.float32)[-lookback:]
    wind_series = np.asarray(wind_series, dtype=np.float32)[-lookback:]
    if len(rainfall_series) < lookback:
        pad = lookback - len(rainfall_series)
        rainfall_series = np.pad(rainfall_series, (pad, 0), mode="edge")
        wind_series = np.pad(wind_series, (pad, 0), mode="edge")

    rain_n = (rainfall_series - normalizer.rain_mean) / normalizer.rain_std
    wind_n = (wind_series - normalizer.wind_mean) / normalizer.wind_std
    x = np.stack([rain_n, wind_n], axis=-1).astype(np.float32)
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    zone_id_t = torch.tensor([zone_cfg["id"]], dtype=torch.long).to(device)

    with torch.no_grad():
        level_pred, flood_logits = model(x_t, zone_id_t)
    level_ft = (level_pred[0].cpu().numpy() * normalizer.level_std) + normalizer.level_mean
    flood_prob = torch.sigmoid(flood_logits[0]).cpu().numpy()

    hours = np.arange(1, horizon + 1)
    return hours, level_ft, flood_prob, zone_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/harvey_lstm.pt")
    parser.add_argument("--zone", required=True)
    parser.add_argument("--rainfall", type=float, required=True, help="constant in/hr over lookback window")
    parser.add_argument("--wind", type=float, required=True, help="constant mph over lookback window")
    args = parser.parse_args()

    model, cfg, normalizer, device = load_checkpoint(args.checkpoint)
    hours, level_ft, flood_prob, zone_cfg = predict_scenario(
        model, cfg, normalizer, device, args.zone,
        constant_rainfall=args.rainfall, constant_wind=args.wind,
    )

    print(f"\nScenario: {zone_cfg['label']} | sustained {args.rainfall} in/hr rain, {args.wind} mph wind")
    print(f"Flood stage: {zone_cfg['flood_stage_ft']} ft (placeholder — verify against NWS AHPS)\n")
    print(f"{'hour':>5} {'level_ft':>10} {'flood_prob':>11}")
    for h, lvl, fp in zip(hours, level_ft, flood_prob):
        flag = "  <-- ABOVE FLOOD STAGE" if lvl >= zone_cfg["flood_stage_ft"] else ""
        print(f"{h:5d} {lvl:10.2f} {fp:11.2f}{flag}")
