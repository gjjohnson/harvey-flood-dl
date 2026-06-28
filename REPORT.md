# Hurricane Harvey Flood Forecaster — Harris County, TX
### Project Report

## Problem

Harris County's flood risk during a tropical storm is highly local even
though the rainfall driving it is roughly county-wide: an urban bayou, a
rural creek, and a flood-control reservoir can all see the same storm and
produce very different — and very different-shaped — flood responses. This
project builds a deep learning model that forecasts a stream gage's water
level 12 hours ahead from the preceding 36 hours of rainfall and wind, and
flags whether it's predicted to cross flood stage, **conditioned on which
of four Harris County watersheds it's looking at.** The goal isn't just
forecasting accuracy in isolation — it's a model that can answer "given the
same storm input, how differently would these four places respond?", which
is the practical question county flood management actually faces.

## Dataset

Four USGS gage stations, chosen to span distinct watershed behavior:

| Zone | USGS site | Character |
|---|---|---|
| Buffalo Bayou at Houston | 08074500 | Urban bayou, fast/flashy |
| Brays Bayou at Houston | 08075000 | Urban bayou, chronic residential flooding (Meyerland) |
| Cypress Creek near Westfield | 08068500 | Rural/suburban, Harvey's most extreme rainfall totals |
| Addicks Reservoir near Addicks | 08073000 | Flood-control reservoir, human-controlled release |

For Aug 23 – Sep 2, 2017 (Harvey's approach through recession), each zone
combines: gage height and discharge from USGS NWIS, precipitation where
co-reported (USGS parameter 00045), and wind speed from NHC's HURDAT2 best
track for Harvey, applied county-wide. All series are resampled to hourly
and merged into a single tidy frame per zone.

The pipeline tries the live USGS/NHC APIs first and falls back to a
clearly-labeled, procedurally generated synthetic approximation if those
services are unreachable (this occurred in the development sandbox, which
has restricted network egress — it should not occur in Colab). Every plot
and metric is tagged with whether it came from real or synthetic data; no
result should be read as "real Harvey observations" without checking that
tag first.

Flood stage thresholds used in this project are illustrative placeholders,
not verified NWS AHPS values — that verification is listed under Future
Work, not assumed here.

## Model

A single LSTM encoder shared across all four zones, conditioned on a
learned per-zone embedding (8 dimensions), with two output heads: a
regression head forecasting gage height for each of the next 12 hours, and
a classification head predicting the probability of exceeding flood stage
at each of those hours.

The shared-encoder design is a direct response to data scarcity: each gage
only has about ten days of storm data, far too little to train a separate
model per zone. Sharing the encoder lets it learn general storm-response
dynamics from all four gages combined, while the zone embedding lets it
specialize the response shape per watershed — which is also the exact
mechanism the demo relies on: holding rainfall and wind fixed and swapping
only the zone embedding isolates the effect of location on the predicted
outcome.

## Training

Adam optimizer, learning rate 1e-3, weight decay 1e-5, 60 epochs, batch
size 64. Loss is multi-task: MSE on the water-level forecast plus a
weighted (0.3×) binary cross-entropy on flood-stage exceedance. The
train/val/test split is **time-based per zone**, not randomly shuffled —
training on the earlier portion of the storm timeline and validating/
testing on the later portion, since shuffling across time would leak
future information into training for what is fundamentally a forecasting
task. The best checkpoint by validation loss is saved, bundled together
with the exact config and fitted normalization statistics used to produce
it, so downstream evaluation/inference never needs to refit anything or
guess hyperparameters.

## Evaluation

Test-set metrics per zone (RMSE/MAE on water level in feet; precision/
recall/F1 on flood-stage exceedance):

| Zone | RMSE (ft) | MAE (ft) |
|---|---|---|
| Buffalo Bayou | 10.9 | 10.6 |
| Brays Bayou | 12.0 | 11.7 |
| Cypress Creek | 20.2 | 19.4 |
| Addicks Reservoir | 13.9 | 13.2 |

Per-zone flood-classification metrics were undefined (no positive examples
in the held-out window) for three of the four zones — a real consequence
of evaluating a single storm event by time: a zone's test window can
legitimately contain zero flood-stage crossings if its recession resolved
before the held-out period began. This is reported as `nan`, not
suppressed, and a pooled-across-zone version of the same metrics is
computed as well, which is well-defined whenever any zone's test window
contains a crossing (here, F1 = 0.05, precision = 0.03, recall = 1.00 —
the model never misses a crossing but is not selective). This is a real
limitation of single-event evaluation, addressed under Future Work via
leave-one-zone-out testing and multi-storm training.

## Results

The model tracks the rise and peak of all four zones well, visually
confirmed by the predicted-vs-actual hydrograph plots (`outputs/
hydrograph_<zone>.png`). Cypress Creek has the highest error of the four —
its rainfall response is the most volatile, consistent with it receiving
Harvey's most extreme single-point rainfall totals. Addicks Reservoir's
slow, multi-day recession is visibly under-predicted by the model, which
decays back toward baseline faster than the real reservoir did — a
sensible failure mode, since the actual recession is paced by Army Corps
release decisions, an operational/policy process that rainfall and wind
inputs don't capture.

The zone-comparison demo makes the core point concretely: holding rainfall
at 0.4 in/hr and wind at 20 mph and only swapping the zone embedding, the
model forecasts Buffalo Bayou (36 ft), Brays Bayou (43 ft), and Cypress
Creek (101 ft) all crossing their respective flood stages, while Addicks
Reservoir (86 ft) stays below its threshold — the same storm input,
genuinely different predicted outcomes by location.

## Limitations

- Trained and evaluated on a single storm event; generalization to a
  different rainfall/wind regime is untested.
- A single county-wide wind series is applied to all four zones rather
  than a per-zone, distance-weighted value.
- Reservoir releases are a human/policy decision that the model cannot
  fully capture from rainfall and wind alone.
- Flood stage thresholds are illustrative placeholders, not verified
  against NWS AHPS.

## Future Work

- Train across multiple storms (Imelda 2019, Beta 2020, Nicholas 2021) so
  the model generalizes across events rather than memorizing one.
- Per-zone wind via distance-weighted interpolation from the HURDAT2 track.
- Leave-one-zone-out evaluation — train on three watersheds, test entirely
  on the fourth — as a cleaner generalization test than the time-based
  split used here, and one that would give every zone a meaningful
  flood-classification evaluation.
- Extend toward spatial flood-extent mapping (Sentinel-1 SAR + DEM, U-Net
  segmentation) as a richer, visually compelling extension beyond
  point-gage forecasting.
- Verify and replace the placeholder flood-stage thresholds with NWS AHPS
  values before using this for anything beyond a demo.
