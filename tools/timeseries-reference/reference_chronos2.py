"""Chronos-2 reference forecasts for tests/Sharplight.TimeSeries.Tests: chronos-forecasting's Chronos2Pipeline (chronos-forecasting >= 2.0) on synthetic tasks:
univariate series (seasonal, gaps, short, constant, longer than the context, a horizon beyond one pass, interpolated levels), two targets forecast jointly,
and targets with past-only and known-future covariates (also over a horizon that unrolls). Writes ~/models/timeseries/refs/<model>.json with per case the
targets, the covariates (history, known future or null), the horizon, the levels and per target the quantile forecasts [levels][horizon] plus the mean.

    python -I tools/timeseries-reference/reference_chronos2.py chronos-2

Needs torch (CPU) and chronos-forecasting >= 2.0 (pip install "chronos-forecasting>=2.0"); the model is expected under ~/models/timeseries/<name>
(amazon/chronos-2: config.json + model.safetensors).
"""
import json, math, os, sys
import numpy as np
import torch
from chronos import Chronos2Pipeline

ROOT = os.path.expanduser("~/models/timeseries")
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def values(a):
    return [None if math.isnan(v) else float(v) for v in a]


def cases():
    rng = np.random.default_rng(11)
    t = np.arange(1400)
    seasonal = 10 + 0.02 * t + 3 * np.sin(2 * math.pi * t / 24) + rng.normal(0, 0.3, t.size)
    gaps = seasonal[:300].copy()
    gaps[[5, 6, 100, 101, 102, 250]] = np.nan
    gaps[:37] = np.nan
    gaps[148:190] = np.nan   # whole empty patches inside the context (patches 10 and 11 of 19)
    long = 50 + 5 * np.sin(2 * math.pi * np.arange(8500) / 168) + rng.normal(0, 1, 8500)
    # Two related targets: the second lags the first by 6 steps.
    a = 20 + 4 * np.sin(2 * math.pi * t / 48) + rng.normal(0, 0.2, t.size)
    b = 5 + 2 * np.sin(2 * math.pi * (t - 6) / 48) + rng.normal(0, 0.2, t.size)
    # A target driven by a known-future promotion flag and a temperature, plus a past-only related series.
    promo = (rng.random(t.size) < 0.15).astype(np.float64)
    temperature = 15 + 8 * np.sin(2 * math.pi * t / 96) + rng.normal(0, 0.5, t.size)
    related = 3 + np.sin(2 * math.pi * t / 24) + rng.normal(0, 0.1, t.size)
    sales = 100 + 30 * promo + 1.5 * temperature + 4 * related + rng.normal(0, 2, t.size)
    temperature_gaps = temperature.copy()
    temperature_gaps[[310, 311, 400]] = np.nan
    uni = lambda name, h, horizon, levels=LEVELS: (name, [h], [], horizon, levels)
    return [
        uni("seasonal", seasonal[:300], 64),
        uni("gaps", gaps, 64),
        uni("short", seasonal[:21] * 100, 24),
        uni("constant", np.full(50, 4.0), 16),
        uni("long_context", long, 64),
        uni("long_horizon", seasonal[:300], 1100),
        uni("interpolated_levels", seasonal[:200], 32, [0.025, 0.25, 0.5, 0.75, 0.975]),
        ("multivariate", [a[:400], b[:400]], [], 48, LEVELS),
        ("covariates", [sales[:300]], [
            ("related", related[:300], None),
            ("promo", promo[:300], promo[300:348]),
            ("temperature", temperature_gaps[:300], temperature_gaps[300:348]),
        ], 48, LEVELS),
        ("covariates_long_horizon", [sales[:200]], [
            ("promo", promo[:200], promo[200:1250]),
        ], 1050, LEVELS),
    ]


def main():
    torch.set_num_threads(8)
    os.makedirs(ROOT + "/refs", exist_ok=True)
    for name in sys.argv[1:]:
        pipeline = Chronos2Pipeline.from_pretrained(ROOT + "/" + name, device_map="cpu", torch_dtype=torch.float32)
        out = []
        for case, targets, covariates, horizon, levels in cases():
            task = {"target": np.stack(targets).astype(np.float32) if len(targets) > 1 else targets[0].astype(np.float32)}
            if covariates:
                task["past_covariates"] = {n: h.astype(np.float32) for n, h, _ in covariates}
                task["future_covariates"] = {n: f.astype(np.float32) for n, _, f in covariates if f is not None}
            with torch.no_grad():
                quantiles, mean = pipeline.predict_quantiles([task], prediction_length=horizon, quantile_levels=levels)
            q = quantiles[0].numpy()   # [targets, horizon, levels]
            m = mean[0].numpy()        # [targets, horizon]
            out.append({
                "name": case,
                "targets": [values(x) for x in targets],
                "covariates": [{"name": n, "history": values(h), "future": None if f is None else values(f)} for n, h, f in covariates],
                "horizon": horizon,
                "levels": levels,
                "forecasts": [{"quantiles": q[i].T.tolist(), "mean": m[i].tolist()} for i in range(len(targets))],
                # The first target in the univariate layout of the other reference files (TimeSeriesModels.Cases).
                "history": values(targets[0]),
                "quantiles": q[0].T.tolist(),
                "mean": m[0].tolist(),
            })
            print(name, case, "ok", flush=True)
        with open(f"{ROOT}/refs/{name}.json", "w") as f:
            json.dump({"model": name, "cases": out}, f)


if __name__ == "__main__":
    main()
