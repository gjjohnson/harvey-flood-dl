"""
model.py
--------
A single shared LSTM encoder, conditioned on a learned per-zone embedding,
with two output heads:
  1. Regression head  -> forecasted gage height for the next `horizon` hours
  2. Classification head -> probability of exceeding flood stage at each of
     those hours

Why one shared model conditioned on zone embedding, rather than four
separate per-zone models: with only ~10 days of data per gage, a per-zone
model would have very little to learn from. Sharing the encoder across
zones lets the model learn general storm-response dynamics from all four
gages combined, while the zone embedding lets it specialize the response
shape per watershed -- which is also exactly the mechanism that makes the
demo's "same storm, different zone" comparison meaningful: swapping the
zone embedding with identical rainfall/wind input isolates the effect of
*location* on the predicted outcome.
"""

import torch
import torch.nn as nn


class ZoneConditionedLSTM(nn.Module):
    def __init__(self, num_zones, rainfall_wind_dim=2, zone_embed_dim=8,
                 hidden_size=64, num_layers=2, horizon=12, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.zone_embed = nn.Embedding(num_zones, zone_embed_dim)

        self.lstm = nn.LSTM(
            input_size=rainfall_wind_dim + zone_embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.level_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )
        self.flood_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),  # logits, one per forecast hour
        )

    def forward(self, x, zone_id):
        """
        x: (batch, lookback, rainfall_wind_dim)
        zone_id: (batch,) long
        returns: level_pred (batch, horizon) [normalized units],
                 flood_logits (batch, horizon)
        """
        batch, lookback, _ = x.shape
        z = self.zone_embed(zone_id)                       # (batch, zone_embed_dim)
        z_rep = z.unsqueeze(1).expand(-1, lookback, -1)     # broadcast over time
        lstm_in = torch.cat([x, z_rep], dim=-1)

        _, (h_n, _) = self.lstm(lstm_in)
        h_final = h_n[-1]                                   # last layer's final hidden state

        level_pred = self.level_head(h_final)
        flood_logits = self.flood_head(h_final)
        return level_pred, flood_logits


def build_model(cfg, num_zones):
    m = cfg["model"]
    return ZoneConditionedLSTM(
        num_zones=num_zones,
        rainfall_wind_dim=m["rainfall_wind_dim"],
        zone_embed_dim=m["zone_embed_dim"],
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        horizon=cfg["windows"]["horizon_hours"],
        dropout=m["dropout"],
    )
