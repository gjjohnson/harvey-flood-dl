# Hurricane Harvey Flood Forecaster — Harris County, TX

A small end-to-end deep learning project: forecast bayou/reservoir water
levels in Harris County during Hurricane Harvey (Aug 2017) from rainfall
and wind, and use it to compare how different watersheds responded to the
same storm.

**Run it in Google Colab:** [`notebooks/Harvey_Flood_Forecaster_Colab.ipynb`](notebooks/Harvey_Flood_Forecaster_Colab.ipynb)
— open it in Colab, set the runtime to GPU, run top to bottom. That's the
fastest path; everything below is the same pipeline run from a terminal.

---

## Problem

Given rainfall and wind over the preceding 36 hours, forecast a stream
gage's water level for the next 12 hours, **and** flag whether it's
predicted to cross flood stage. The interesting part isn't the forecasting
task in isolation — it's that the model is conditioned on *which* of four
Harris County watersheds it's looking at, so a single trained model can
answer "given the same rainfall and wind, how differently would Buffalo
Bayou, Brays Bayou, Cypress Creek, and Addicks Reservoir respond?" That's
the question Harris County actually has to answer every storm: rainfall is
roughly county-wide, but flood outcomes are extremely local.

## Dataset

Four USGS gage stations, each standing in for a different watershed
"personality" in Harris County:

| Zone | USGS site | Why this one |
|---|---|---|
| Buffalo Bayou at Houston | 08074500 | Urban bayou, fast/flashy response, downtown/west Houston |
| Brays Bayou at Houston | 08075000 | Urban bayou, infamous for repeated residential flooding (Meyerland / Texas Medical Center) |
| Cypress Creek near Westfield | 08068500 | Rural/suburban creek, NW Harris Co., received some of Harvey's single highest rainfall totals |
| Addicks Reservoir near Addicks | 08073000 | Flood-control reservoir — slow, *human-controlled* release rather than free-flowing channel |

For each zone, three signals over Aug 23 – Sep 2, 2017:
- **Gage height (ft)** and **discharge (cfs)** — [USGS NWIS instantaneous values service](https://waterservices.usgs.gov/nwis/iv/)
- **Precipitation (in)**, where the gage co-reports it — same USGS service, parameter code `00045`
- **Wind speed (mph)** — [NHC HURDAT2 best track](https://www.nhc.noaa.gov/data/hurdat/) for Harvey (AL092017), interpolated to hourly and applied county-wide (Harris County is small relative to the storm's wind field, so this is a reasonable simplification, not a per-zone signal)

**On synthetic fallback data:** `dataset.py` tries the live USGS/NHC APIs
first and falls back to a clearly-labeled (`synthetic=True`) procedurally
generated approximation if they're unreachable (this happens, for example,
in network-restricted sandboxes — it should *not* happen in Colab, which
has unrestricted internet). Every plot and printed summary tags whether a
zone's data is real or synthetic. Don't report results from a run as "real
Harvey data" without checking those tags first.

**Flood stage thresholds** in `config.yaml` are placeholders for
illustrating the pipeline — verify exact action/minor/moderate/major flood
stage values against [NWS AHPS](https://water.weather.gov/ahps/) for each
site before using this for anything beyond a demo.

## Model

A single shared LSTM encoder conditioned on a learned per-zone embedding,
with two output heads (`model.py`):
- **Regression head** — forecasted gage height for each of the next 12 hours
- **Classification head** — probability of exceeding flood stage at each hour

One model shared across all four zones (rather than four separate models)
because each zone only has ~10 days of data — a per-zone model would have
almost nothing to learn from. Sharing the encoder lets it learn general
storm-response dynamics from all four gages combined, while the zone
embedding specializes the response shape per watershed. This is also
exactly the mechanism the demo exploits: holding rainfall/wind fixed and
swapping the zone embedding isolates the effect of *location* on the
predicted outcome.

## Training

Time-based split per zone (60/15/25-ish depending on `config.yaml`) — never
random shuffling across time, since that would leak the future into
training for a forecasting task. Multi-task loss: MSE on water level +
weighted BCE on flood exceedance. See `train.py`.

```bash
python train.py --config config.yaml
```

Trains in well under a minute on CPU for this dataset size; GPU is not
required but Colab gives you one for free anyway.

## Evaluation

```bash
python evaluate.py --config config.yaml --checkpoint models/harvey_lstm.pt
```

Reports, per zone: RMSE/MAE on forecasted water level, and precision/
recall/F1 on flood-stage exceedance — plus a pooled-across-zone version of
the classification metrics. **Why pooled:** with a single storm event,
a zone's time-based test window can legitimately contain zero flood-stage
crossings (the recession may have fully resolved before the held-out window
starts for that zone) — `nan` for those zones is expected, not a bug.
Pooling across zones gives one well-defined classification number whenever
*any* zone's test window contains a crossing.

Also writes a predicted-vs-actual hydrograph plot per zone to
`outputs/hydrograph_<zone>.png`.

## Demo

```bash
python demo.py --checkpoint models/harvey_lstm.pt
```

A Gradio app: pick a zone, set sustained rainfall (in/hr) and wind (mph),
see the 12-hour forecast and whether it's predicted to cross flood stage.
**Hold rainfall/wind fixed and switch the zone dropdown** — that's the
comparison this whole project is built around.

In Colab this renders inline automatically; locally it opens in your
browser. There's also a no-UI CLI version for quick scripted comparisons:

```bash
python inference.py --zone brays_bayou --rainfall 3.0 --wind 50
python inference.py --zone addicks_reservoir --rainfall 3.0 --wind 50
```

## Repo structure

```
harvey-flood-dl/
├── data/raw/             cached downloads (USGS CSVs, HURDAT2 track)
├── data/processed/       (reserved; current pipeline works in-memory)
├── models/               trained checkpoint (model + config + normalizer bundled together)
├── notebooks/            Colab notebook (primary entry point)
├── outputs/              training curves, hydrograph plots, test_metrics.csv
├── dataset.py            USGS/HURDAT2 fetch + synthetic fallback + windowing
├── model.py              ZoneConditionedLSTM
├── train.py              training loop
├── evaluate.py           test-set metrics + hydrograph plots
├── inference.py          scenario-based "what if" prediction (used by demo.py)
├── demo.py               Gradio app
├── config.yaml           all zones/hyperparameters/paths — edit this, not the code
└── requirements.txt
```

## Limitations

- **Single storm event.** Everything here is trained and evaluated on one
  storm. The model has not seen how these watersheds behave in a different
  rainfall/wind regime, and the flood-classification head in particular is
  only lightly validated due to limited held-out positive examples for some
  zones (see the Evaluation section above).
- **County-wide wind, not per-zone wind.** A single hourly wind value is
  applied to all four zones; a more careful version would account for
  storm-center distance to each watershed.
- **Reservoir releases are policy decisions, not pure hydrology.** Addicks
  is operated by the Army Corps of Engineers — actual releases depend on
  human decisions (and, during Harvey, contributed to widely-reported
  controversy over upstream/downstream flooding) that rainfall and wind
  alone don't fully capture. The model visibly underestimates Addicks'
  slow recession relative to the bayous (see `outputs/hydrograph_addicks_reservoir.png`).
- **Placeholder flood stage thresholds** — see the Dataset section.

## Future work

- Add more storms (Imelda 2019, Beta 2020, Nicholas 2021) so the model
  generalizes across events, not just within one
- Per-zone wind via distance-weighted interpolation from the HURDAT2 track
  rather than a single county-wide series
- Spatial flood-extent mapping (Sentinel-1 SAR + DEM, U-Net segmentation)
  as a richer, visually compelling extension beyond point-gage forecasting
- Leave-one-zone-out evaluation, training on 3 watersheds and testing
  entirely on the 4th, as a cleaner generalization test than the time-based
  split used here
