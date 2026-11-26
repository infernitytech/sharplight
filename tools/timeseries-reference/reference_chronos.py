"""Chronos / Chronos-Bolt reference forecasts for tests/Sharplight.TimeSeries.Tests: the chronos-forecasting package's pipelines on a few synthetic series
(seasonal with trend and noise, with missing values, short, constant, longer than the context). Writes ~/models/timeseries/refs/<model>.json with per case the
history, the horizon and the quantile forecasts (levels 0.1 ... 0.9) plus the mean.

    python -I tools/timeseries-reference/reference_chronos.py chronos-bolt-tiny [chronos-bolt-small ...]

Needs torch (CPU) and chronos-forecasting (pip install chronos-forecasting); the models are expected under ~/models/timeseries/<name>.
For the sampling (token) models the forecasts are drawn with top_k = 1 (greedy), the deterministic case the test can compare exactly.
"""
import json, math, os, sys
import numpy as np
import torch
from chronos import BaseChronosPipeline, ChronosPipeline

ROOT = os.path.expanduser("~/models/timeseries")
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def cases():
    rng = np.random.default_rng(7)
    t = np.arange(400)
    seasonal = 10 + 0.02 * t + 3 * np.sin(2 * math.pi * t / 24) + rng.normal(0, 0.3, t.size)
    gaps = seasonal[:300].copy()
    gaps[[5, 6, 100, 101, 102, 250]] = np.nan
    gaps[:37] = np.nan
    long = 50 + 5 * np.sin(2 * math.pi * np.arange(2600) / 168) + rng.normal(0, 1, 2600)
    return [
        ("seasonal", seasonal[:300], 64),
        ("gaps", gaps, 64),
        ("short", seasonal[:21] * 100, 24),
        ("constant", np.full(50, 4.0), 16),
        ("long_context", long, 64),
        ("long_horizon", seasonal[:300], 150),
        ("interpolated_levels", seasonal[:200], 32),
    ]


def main():
    torch.set_num_threads(4)
    os.makedirs(ROOT + "/refs", exist_ok=True)
    for name in sys.argv[1:]:
        pipeline = BaseChronosPipeline.from_pretrained(ROOT + "/" + name, device_map="cpu", torch_dtype=torch.float32)
        token = isinstance(pipeline, ChronosPipeline)
        out = []
        for case, history, horizon in cases():
            context = torch.tensor(history, dtype=torch.float32)
            levels = [0.05, 0.25, 0.5, 0.75, 0.95] if case == "interpolated_levels" else LEVELS
            kwargs = {"top_k": 1, "num_samples": 1} if token else {}
            with torch.no_grad():
                quantiles, mean = pipeline.predict_quantiles(context, prediction_length=horizon, quantile_levels=levels, **kwargs)
            out.append({
                "name": case,
                "history": [None if math.isnan(v) else float(v) for v in history],
                "horizon": horizon,
                "levels": levels,
                "quantiles": quantiles[0].T.tolist(),   # [levels][horizon]
                "mean": mean[0].tolist(),
            })
        with open(f"{ROOT}/refs/{name}.json", "w") as f:
            json.dump({"model": name, "cases": out}, f)
        print(name, "ok")


if __name__ == "__main__":
    main()
