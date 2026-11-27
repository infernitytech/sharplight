"""TimesFM reference forecasts for tests/Sharplight.TimeSeries.Tests, in the format of reference_chronos.py (same synthetic series): per case the history,
the horizon, the quantile forecasts and the point forecast ("mean"). Writes ~/models/timeseries/refs/<model>.json.

    python -I tools/timeseries-reference/reference_timesfm.py timesfm-2.0-500m-pytorch timesfm-2.5-200m-pytorch

Needs torch (CPU), transformers (>= 5) and the timesfm package (pip install "timesfm[torch] @ git+https://github.com/google-research/timesfm");
the models are expected under ~/models/timeseries/<name>.

- TimesFM 2.0 (model_type timesfm): transformers' TimesFmModelForPrediction with frequency 0. Its forward predicts one output patch (128 steps);
  longer horizons append the mean forecast to the series and call it again, as the timesfm 1.x package's decoding loop does. Missing values are
  filled first as the timesfm package does (leading ones dropped, the others linearly interpolated), since transformers takes no missing values.
- TimesFM 2.5: the timesfm package's TimesFM_2p5_200M_torch with the flags its model card recommends (normalize_inputs, continuous quantile head,
  flip invariance, non-negativity, quantile-crossing fix) and max_context = 15360 (the context limit 16384 less the 1024 steps of the continuous quantile head, the context Sharplight uses).
Levels other than the native 0.1 ... 0.9 interpolate as Sharplight does (edge-padded native forecasts, torch.quantile's linear rule).
"""
import json, math, os, sys
import numpy as np
import torch

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
    signed = 3 * np.sin(2 * math.pi * t[:250] / 50) + rng.normal(0, 0.5, 250) - 1
    return [
        ("seasonal", seasonal[:300], 64),
        ("gaps", gaps, 64),
        ("short", seasonal[:21] * 100, 24),
        ("constant", np.full(50, 4.0), 16),
        ("long_context", long, 64),
        ("long_horizon", seasonal[:300], 300),
        ("interpolated_levels", seasonal[:200], 32),
        ("signed", signed, 200),
    ]


def fill(values):
    """The timesfm package's preprocessing: strip leading NaNs, linearly interpolate the others."""
    values = np.array(values, dtype=np.float64)
    nan = np.isnan(values)
    values = values[np.argmax(~nan):]
    nan = np.isnan(values)
    if nan.any():
        values[nan] = np.interp(np.nonzero(nan)[0], np.nonzero(~nan)[0], values[~nan])
    return values


def select(native, levels):
    """[horizon, 9] native quantiles -> [len(levels), horizon] as Sharplight's TimeSeriesQuantiles.Select."""
    if all(any(abs(l - n) < 1e-6 for n in LEVELS) for l in levels):
        return [native[:, LEVELS.index(next(n for n in LEVELS if abs(l - n) < 1e-6))].tolist() for l in levels]
    padded = np.sort(np.concatenate([native[:, :1], native, native[:, -1:]], axis=1), axis=1)
    out = []
    for level in levels:
        position = level * (padded.shape[1] - 1)
        low = int(math.floor(position))
        high = min(low + 1, padded.shape[1] - 1)
        out.append((padded[:, low] + (position - low) * (padded[:, high] - padded[:, low])).tolist())
    return out


def timesfm20(path):
    from transformers import TimesFmModelForPrediction
    model = TimesFmModelForPrediction.from_pretrained(path, torch_dtype=torch.float32).eval()

    def forecast(history, horizon):
        series = torch.tensor(fill(history), dtype=torch.float32)
        blocks = []
        while sum(b.shape[0] for b in blocks) < horizon:
            with torch.no_grad():
                full = model(past_values=[series], freq=[0], return_dict=True).full_predictions[0]   # [128, 10]: mean, then the levels
            blocks.append(full)
            series = torch.cat([series, full[:, 0]])
        full = torch.cat(blocks)[:horizon].double().numpy()
        return full[:, 1:], full[:, 0]

    return forecast


def timesfm25(path):
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
    model.load_checkpoint(path)
    compiled = {}

    def forecast(history, horizon):
        steps = 128 * ((horizon + 127) // 128)
        if steps not in compiled:
            model.compile(timesfm.ForecastConfig(max_context=15360, max_horizon=steps, normalize_inputs=True, use_continuous_quantile_head=True,
                                                 force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
            compiled[steps] = model.compiled_decode
        model.compiled_decode = compiled[steps]
        point, quantiles = model.forecast(horizon=horizon, inputs=[np.array(history, dtype=np.float64)])
        return quantiles[0][:, 1:].astype(np.float64), point[0].astype(np.float64)

    return forecast


def main():
    torch.set_num_threads(8)
    os.makedirs(ROOT + "/refs", exist_ok=True)
    for name in sys.argv[1:]:
        path = ROOT + "/" + name
        with open(path + "/config.json") as f:
            config = json.load(f)
        is25 = config.get("model_type") == "timesfm2_5" or "quantile_horizon_length" in config or "output_quantile_len" in config
        if is25 and config.get("model_type") == "timesfm2_5":
            raise SystemExit(f"{name}: the transformers layout of TimesFM 2.5 shares the reference of timesfm-2.5-200m-pytorch")
        forecast = timesfm25(path) if is25 else timesfm20(path)
        out = []
        for case, history, horizon in cases():
            levels = [0.05, 0.25, 0.5, 0.75, 0.95] if case == "interpolated_levels" else LEVELS
            native, point = forecast(history, horizon)
            out.append({
                "name": case,
                "history": [None if math.isnan(v) else float(v) for v in history],
                "horizon": horizon,
                "levels": levels,
                "quantiles": select(native, levels),   # [levels][horizon]
                "mean": point.tolist(),
            })
            print(name, case, "ok", flush=True)
        with open(f"{ROOT}/refs/{name}.json", "w") as f:
            json.dump({"model": name, "cases": out}, f)
        print(name, "ok")


if __name__ == "__main__":
    main()
