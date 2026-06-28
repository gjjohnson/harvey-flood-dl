"""
dataset.py
----------
Data acquisition + preprocessing for the Harvey flood-forecasting project.

Three data sources, all keyed to the storm window in config.yaml:
  1. USGS NWIS instantaneous-values service -> gage height (ft), discharge
     (cfs), and (where available) precipitation (in) at each zone's gage.
  2. NHC HURDAT2 best-track -> Harvey's 6-hourly position + max sustained
     wind, interpolated to hourly and broadcast county-wide (Harris County
     is small relative to the storm's wind field, so a single county-wide
     wind series per hour is a reasonable simplification for this project).
  3. A synthetic fallback for both of the above, used automatically if the
     live request fails (no network, USGS/NHC down, running somewhere
     network-restricted, etc). Synthetic rows are tagged `synthetic=True`
     so they're never silently confused with real observations.

Everything bottoms out in `make_datasets(cfg)`, which returns ready-to-train
PyTorch Datasets (train/val/test), split by TIME (not randomly shuffled),
since this is a forecasting task and shuffling across time would leak the
future into training.
"""

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
import yaml
import torch
from torch.utils.data import Dataset

USGS_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023.txt"


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["data"]["raw_dir"], exist_ok=True)
    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)
    return cfg


# --------------------------------------------------------------------------
# 1. USGS NWIS
# --------------------------------------------------------------------------
def fetch_usgs_gage(site, start, end, param_codes, cache_dir, timeout=30):
    """
    Pull instantaneous values for one USGS site over [start, end].
    param_codes: dict like {"gage_height": "00065", "discharge": "00060", ...}
    Returns a DataFrame indexed by UTC datetime, one column per requested
    parameter (named by the dict key, not the USGS code), or raises on
    failure so the caller can decide to fall back to synthetic data.
    """
    cache_path = os.path.join(cache_dir, f"usgs_{site}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    codes = ",".join(param_codes.values())
    params = {
        "format": "json",
        "sites": site,
        "startDT": start,
        "endDT": end,
        "parameterCd": codes,
        "siteStatus": "all",
    }
    resp = requests.get(USGS_IV_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    code_to_name = {v: k for k, v in param_codes.items()}
    series = {}
    for ts in payload.get("value", {}).get("timeSeries", []):
        pcode = ts["variable"]["variableCode"][0]["value"]
        name = code_to_name.get(pcode)
        if name is None:
            continue
        values = ts["values"][0]["value"]
        if not values:
            continue
        idx = pd.to_datetime([v["dateTime"] for v in values], utc=True)
        vals = pd.to_numeric([v["value"] for v in values], errors="coerce")
        series[name] = pd.Series(vals, index=idx)

    if not series:
        raise ValueError(f"No usable series returned for site {site}")

    df = pd.DataFrame(series)
    df.to_csv(cache_path)
    return df


# --------------------------------------------------------------------------
# 2. NHC HURDAT2 best track
# --------------------------------------------------------------------------
def fetch_hurdat2_track(storm_id, cache_dir, timeout=30):
    """
    Download (or load cached) HURDAT2 and extract one storm's track.
    Returns DataFrame indexed by UTC datetime with columns [lat, lon, wind_kt].
    """
    cache_path = os.path.join(cache_dir, f"hurdat2_{storm_id}.csv")
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    resp = requests.get(HURDAT2_URL, timeout=timeout)
    resp.raise_for_status()
    lines = resp.text.splitlines()

    rows = []
    capture = False
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0].startswith("AL"):
            # header line: AL092017, HARVEY, <n records>,
            capture = parts[0] == storm_id
            continue
        if not capture or len(parts) < 7:
            continue
        date_s, time_s = parts[0], parts[1]
        lat_s, lon_s, wind_s = parts[4], parts[5], parts[6]
        dt = pd.to_datetime(date_s + time_s, format="%Y%m%d%H%M", utc=True)
        lat = float(lat_s[:-1]) * (1 if lat_s.endswith("N") else -1)
        lon = float(lon_s[:-1]) * (-1 if lon_s.endswith("W") else 1)
        wind_kt = float(wind_s)
        rows.append((dt, lat, lon, wind_kt))

    if not rows:
        raise ValueError(f"Storm id {storm_id} not found in HURDAT2 file")

    df = pd.DataFrame(rows, columns=["datetime", "lat", "lon", "wind_kt"]).set_index("datetime")
    df.to_csv(cache_path)
    return df


# --------------------------------------------------------------------------
# 3. Synthetic fallback (clearly labeled, never silently mixed with real data)
# --------------------------------------------------------------------------
def _synthetic_wind_series(storm_cfg, freq):
    """
    Approximate Harvey's wind history as experienced near Harris County:
    ramps up before landfall, weakens to tropical storm strength once
    stalled inland over the Houston area (this is when the catastrophic
    rain occurred), per general knowledge of Harvey's track. This is an
    illustrative approximation, NOT a substitute for the real HURDAT2 track.
    """
    idx = pd.date_range(storm_cfg["start"], storm_cfg["end"], freq=freq, tz="UTC")
    t = np.linspace(0, 1, len(idx))
    peak_pos = 0.30
    wind = 100 * np.exp(-((t - peak_pos) ** 2) / (2 * 0.07 ** 2))
    wind += 20 * np.exp(-((t - peak_pos) ** 2) / (2 * 0.35 ** 2))  # long weak tail
    wind = np.clip(wind, 10, None)
    return pd.DataFrame({"wind_speed_mph": wind}, index=idx)


def _synthetic_zone_series(zone_cfg, storm_cfg, freq, rng):
    """
    Build a synthetic [rainfall_in, gage_height_ft] series for one zone with
    a *zone-characteristic* response shape:
      - buffalo_bayou / brays_bayou: flashy urban bayous, rise and fall fast
      - cypress_creek: extreme total rainfall, sharp prolonged rise
      - addicks_reservoir: slow, human-controlled, lagged, long sustained
        high pool rather than a sharp peak
    """
    idx = pd.date_range(storm_cfg["start"], storm_cfg["end"], freq=freq, tz="UTC")
    n = len(idx)
    t = np.linspace(0, 1, n)

    # recession rates tuned so that, given this storm window's length, the
    # held-out (post-peak) portion of the timeline still spans both
    # above- and below-flood-stage periods for every zone -- a single-spike
    # recession that fully decays before the val/test split begins would
    # leave nothing for the flood-classification head to be evaluated
    # against. This also better reflects Harvey's real character: Harris
    # County bayous stayed elevated for days, and the reservoirs (Addicks/
    # Barker) remained high for weeks after, due to controlled releases.
    profiles = {
        "buffalo_bayou":     dict(rain_peak=0.35, rain_width=0.10, rain_scale=2.2,
                                   base_level=20.0, gain=18.0, lag_hr=3,  recession=0.009),
        "brays_bayou":       dict(rain_peak=0.36, rain_width=0.10, rain_scale=2.6,
                                   base_level=25.0, gain=24.0, lag_hr=2,  recession=0.010),
        "cypress_creek":     dict(rain_peak=0.45, rain_width=0.18, rain_scale=3.4,
                                   base_level=70.0, gain=32.0, lag_hr=6,  recession=0.010),
        "addicks_reservoir": dict(rain_peak=0.40, rain_width=0.20, rain_scale=2.0,
                                   base_level=85.0, gain=15.0, lag_hr=24, recession=0.005),
    }
    p = profiles[zone_cfg["name"]]

    rainfall = p["rain_scale"] * np.exp(-((t - p["rain_peak"]) ** 2) / (2 * p["rain_width"] ** 2))
    rainfall += rng.gamma(shape=0.6, scale=0.05, size=n)  # noisy showers throughout
    rainfall = np.clip(rainfall, 0, None)

    lag_steps = max(int(p["lag_hr"] / (pd.Timedelta(freq) / pd.Timedelta("1h"))), 1)
    rain_resp = pd.Series(rainfall).rolling(window=lag_steps, min_periods=1).mean().values
    kernel = np.ones(max(lag_steps * 2, 2))
    rise = p["gain"] * np.convolve(rain_resp, kernel, mode="same") / len(kernel)
    recession_decay = np.exp(-p["recession"] * np.maximum(np.arange(n) - np.argmax(rise), 0))
    level = p["base_level"] + rise * recession_decay
    level += rng.normal(0, 0.15, size=n)  # sensor noise

    return pd.DataFrame({"rainfall_in": rainfall, "gage_height_ft": level}, index=idx)


# --------------------------------------------------------------------------
# Orchestration: build one zone's merged hourly DataFrame
# --------------------------------------------------------------------------
def build_zone_dataframe(zone_cfg, cfg, wind_df=None):
    storm_cfg, data_cfg = cfg["storm"], cfg["data"]
    freq = data_cfg["resample_freq"]
    synthetic = False

    try:
        raw = fetch_usgs_gage(
            zone_cfg["usgs_site"], storm_cfg["start"], storm_cfg["end"],
            cfg["usgs_params"], data_cfg["raw_dir"],
        )
        df = raw.resample(freq).mean().interpolate(limit_direction="both")
        df = df.rename(columns={"gage_height": "gage_height_ft", "precip": "rainfall_in"})
        if "rainfall_in" not in df.columns:
            # not every site reports precipitation; this is expected/normal
            raise ValueError("site has no co-located precipitation parameter")
    except Exception as e:
        if not data_cfg["synthetic_fallback"]:
            raise
        warnings.warn(
            f"[{zone_cfg['name']}] live USGS fetch unavailable ({e}); "
            f"using SYNTHETIC fallback data for this zone."
        )
        rng = np.random.default_rng(abs(hash(zone_cfg["name"])) % (2**32))
        df = _synthetic_zone_series(zone_cfg, storm_cfg, freq, rng)
        synthetic = True

    if wind_df is None:
        wind_df = pd.DataFrame()
    df = df.join(wind_df, how="left").interpolate(limit_direction="both")
    df["zone_id"] = zone_cfg["id"]
    df["zone_name"] = zone_cfg["name"]
    df["synthetic"] = synthetic
    return df


def build_wind_series(cfg):
    storm_cfg, data_cfg = cfg["storm"], cfg["data"]
    try:
        track = fetch_hurdat2_track(storm_cfg["hurdat2_id"], data_cfg["raw_dir"])
        idx = pd.date_range(storm_cfg["start"], storm_cfg["end"], freq=data_cfg["resample_freq"], tz="UTC")
        wind_kt = track["wind_kt"].reindex(track.index.union(idx)).interpolate().reindex(idx)
        wind_df = pd.DataFrame({"wind_speed_mph": wind_kt.values * 1.15078}, index=idx)
        if wind_df["wind_speed_mph"].isna().all():
            raise ValueError("HURDAT2 track did not cover the requested window")
        return wind_df, False
    except Exception as e:
        if not data_cfg["synthetic_fallback"]:
            raise
        warnings.warn(f"live HURDAT2 fetch unavailable ({e}); using SYNTHETIC wind series.")
        return _synthetic_wind_series(storm_cfg, data_cfg["resample_freq"]), True


def build_all_zones(cfg):
    """Returns dict[zone_name] -> DataFrame, and the combined long DataFrame."""
    wind_df, wind_synthetic = build_wind_series(cfg)
    zones = {}
    for zone_cfg in cfg["zones"]:
        zones[zone_cfg["name"]] = build_zone_dataframe(zone_cfg, cfg, wind_df=wind_df)
    combined = pd.concat(zones.values(), axis=0).sort_index()
    return zones, combined, wind_synthetic


# --------------------------------------------------------------------------
# Windowing + PyTorch Dataset
# --------------------------------------------------------------------------
@dataclass
class Normalizer:
    rain_mean: float
    rain_std: float
    wind_mean: float
    wind_std: float
    level_mean: float
    level_std: float

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return Normalizer(**d)


def fit_normalizer(df):
    return Normalizer(
        rain_mean=df["rainfall_in"].mean(), rain_std=df["rainfall_in"].std() + 1e-6,
        wind_mean=df["wind_speed_mph"].mean(), wind_std=df["wind_speed_mph"].std() + 1e-6,
        level_mean=df["gage_height_ft"].mean(), level_std=df["gage_height_ft"].std() + 1e-6,
    )


class HarveyWindowDataset(Dataset):
    """
    Sliding-window dataset over one or more zones.
    Each sample: lookback hours of [rainfall, wind] -> next horizon hours of
    gage height (regression target) + flood-stage exceedance (binary target).
    """

    def __init__(self, zone_frames, zone_cfgs, normalizer, lookback, horizon, stride):
        self.samples = []  # list of (x, zone_id, y_level, y_flood)
        flood_stage = {z["id"]: z["flood_stage_ft"] for z in zone_cfgs}

        for name, df in zone_frames.items():
            zone_id = int(df["zone_id"].iloc[0])
            rain = (df["rainfall_in"].values - normalizer.rain_mean) / normalizer.rain_std
            wind = (df["wind_speed_mph"].values - normalizer.wind_mean) / normalizer.wind_std
            level = df["gage_height_ft"].values
            level_n = (level - normalizer.level_mean) / normalizer.level_std
            n = len(df)
            for start in range(0, n - lookback - horizon, stride):
                x = np.stack([
                    rain[start:start + lookback],
                    wind[start:start + lookback],
                ], axis=-1).astype(np.float32)
                y_level = level_n[start + lookback:start + lookback + horizon].astype(np.float32)
                y_level_raw = level[start + lookback:start + lookback + horizon]
                y_flood = (y_level_raw >= flood_stage[zone_id]).astype(np.float32)
                self.samples.append((x, zone_id, y_level, y_flood))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, zone_id, y_level, y_flood = self.samples[i]
        return (
            torch.from_numpy(x),
            torch.tensor(zone_id, dtype=torch.long),
            torch.from_numpy(y_level),
            torch.from_numpy(y_flood),
        )


def time_split(df, train_frac, val_frac):
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def make_datasets(cfg):
    """
    Top-level entry point. Returns:
        train_ds, val_ds, test_ds, normalizer, zone_frames (full, unsplit, for plotting)
    """
    zone_frames, combined, wind_synthetic = build_all_zones(cfg)
    normalizer = fit_normalizer(combined)

    train_frac, val_frac = cfg["split"]["train_frac"], cfg["split"]["val_frac"]
    lookback, horizon, stride = (
        cfg["windows"]["lookback_hours"], cfg["windows"]["horizon_hours"], cfg["windows"]["stride_hours"]
    )

    train_frames, val_frames, test_frames = {}, {}, {}
    for name, df in zone_frames.items():
        tr, va, te = time_split(df, train_frac, val_frac)
        # give val/test a lookback-window of context immediately preceding
        # them so the first forecast in each split isn't context-starved
        train_frames[name] = tr
        val_frames[name] = pd.concat([tr.iloc[-lookback:], va])
        test_frames[name] = pd.concat([va.iloc[-lookback:], te])

    zone_cfgs = cfg["zones"]
    train_ds = HarveyWindowDataset(train_frames, zone_cfgs, normalizer, lookback, horizon, stride)
    val_ds = HarveyWindowDataset(val_frames, zone_cfgs, normalizer, lookback, horizon, stride)
    test_ds = HarveyWindowDataset(test_frames, zone_cfgs, normalizer, lookback, horizon, stride)

    return train_ds, val_ds, test_ds, normalizer, zone_frames
