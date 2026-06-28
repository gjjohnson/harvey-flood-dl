"""
demo.py
-------
Interactive demo: pick a zone, set a sustained rainfall rate and wind speed,
see the model's predicted gage-height trajectory for the next N hours and
whether it's predicted to cross flood stage.

This is the piece that lets you directly compare zones: hold rainfall/wind
fixed and swap the zone dropdown to see how the same storm input produces a
very different outcome at, say, Addicks Reservoir vs. Brays Bayou.

Run:
    python demo.py --checkpoint models/harvey_lstm.pt

In Colab, this opens an inline widget automatically (Gradio detects the
notebook environment); locally it opens in your browser. Pass --share if
you want a public Gradio link.
"""

import argparse

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from inference import load_checkpoint, predict_scenario


def build_demo(checkpoint_path):
    model, cfg, normalizer, device = load_checkpoint(checkpoint_path)
    zone_choices = [(z["label"], z["name"]) for z in cfg["zones"]]
    lookback = cfg["windows"]["lookback_hours"]
    horizon = cfg["windows"]["horizon_hours"]

    def run(zone_name, rainfall_in_hr, wind_mph):
        hours, level_ft, flood_prob, zone_cfg = predict_scenario(
            model, cfg, normalizer, device, zone_name,
            constant_rainfall=rainfall_in_hr, constant_wind=wind_mph,
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hours, level_ft, marker="o", color="tab:red", label="forecast gage height")
        ax.axhline(zone_cfg["flood_stage_ft"], color="orange", linestyle="--",
                   label=f"flood stage ({zone_cfg['flood_stage_ft']} ft)")
        ax.set_xlabel("hours ahead")
        ax.set_ylabel("gage height (ft)")
        ax.set_title(f"{zone_cfg['label']} — sustained {rainfall_in_hr} in/hr, {wind_mph} mph wind")
        ax.legend()
        fig.tight_layout()

        will_flood = bool(np.any(level_ft >= zone_cfg["flood_stage_ft"]))
        first_flood_hour = int(hours[level_ft >= zone_cfg["flood_stage_ft"]][0]) if will_flood else None
        peak_ft = float(np.max(level_ft))

        if will_flood:
            summary = (
                f"⚠️ Predicted to **exceed flood stage** within the {horizon}-hour forecast window "
                f"(first crossing at hour {first_flood_hour}). Peak forecast level: {peak_ft:.1f} ft "
                f"(flood stage: {zone_cfg['flood_stage_ft']} ft)."
            )
        else:
            summary = (
                f"✅ Not predicted to exceed flood stage in the {horizon}-hour forecast window. "
                f"Peak forecast level: {peak_ft:.1f} ft (flood stage: {zone_cfg['flood_stage_ft']} ft)."
            )

        note = ("\n\n*Note: flood-stage thresholds in this demo are illustrative placeholders — "
                "verify exact values against NWS AHPS before treating this as a real forecast.*")
        return fig, summary + note

    with gr.Blocks(title="Hurricane Harvey Flood Forecaster — Harris County") as demo:
        gr.Markdown(
            "# Hurricane Harvey Flood Forecaster (demo)\n"
            "Pick a zone (a Harris County watershed gage) and a sustained rainfall/wind "
            f"scenario over the past {lookback} hours, and see the model's {horizon}-hour "
            "gage-height forecast. **Hold the rainfall/wind the same and switch zones** to "
            "see how location changes the outcome — that's the point of this demo."
        )
        with gr.Row():
            with gr.Column(scale=1):
                zone_dd = gr.Dropdown(choices=zone_choices, value=zone_choices[0][1], label="Zone")
                rainfall_sl = gr.Slider(0, 6, value=2.0, step=0.1, label="Sustained rainfall (in/hr)")
                wind_sl = gr.Slider(0, 100, value=40, step=1, label="Sustained wind speed (mph)")
                run_btn = gr.Button("Run forecast", variant="primary")
            with gr.Column(scale=2):
                plot_out = gr.Plot(label="Forecast")
                text_out = gr.Markdown()

        run_btn.click(run, inputs=[zone_dd, rainfall_sl, wind_sl], outputs=[plot_out, text_out])
        demo.load(run, inputs=[zone_dd, rainfall_sl, wind_sl], outputs=[plot_out, text_out])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/harvey_lstm.pt")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo(args.checkpoint)
    demo.launch(share=args.share)
