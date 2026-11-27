"""Moirai reference forecasts for tests/Sharplight.TimeSeries.Tests, in the format of reference_timesfm.py (same synthetic series). Writes
~/models/timeseries/refs/<model>.json.

    python -I tools/timeseries-reference/reference_moirai.py moirai-2.0-R-small moirai-1.1-R-small

Needs torch (CPU) and uni2ts (pip install --no-deps "uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts" plus einops, jaxtyping (with jax for
its PyTree), gluonts, lightning, huggingface_hub, safetensors, hydra-core); the models are expected under ~/models/timeseries/<name> (config.json and
model.safetensors of Salesforce/<name>). fedora-blackwell has the environment in ~/venvs/uni2ts.

- Moirai 2.0 (config with quantile_levels): Moirai2Forecast.predict with context_length = min(len(history), 1680) (the uni2ts notebook's context; no
  left padding of the context then), quantiles exact; the point forecast is the median. Levels other than the native 0.1 ... 0.9 interpolate as Sharplight
  does (edge-padded native forecasts, torch.quantile's linear rule).
- Moirai 1.1 (config with distr_output): MoiraiForecast at patch size 32 (the notebook's) and context_length = min(len(history), 1000), missing values
  zero-filled as gluonts' AddObservedValuesIndicator does. Per forecast step the mixture parameters after the domain maps ("params": weights (probs),
  student_t df/loc/scale, normal loc, negative_binomial total_count/logits, log_normal loc/scale) and the scaler's loc/scale, plus the quantiles and the mean
  of 20000 seeded samples (statistics only; torch's RNG is not reproduced).
- Moirai 1.1 with --auto (writes refs/<model>-auto.json): MoiraiForecast(patch_size="auto") at context_length 1000 (Sharplight's context), fed as
  gluonts feeds it: the history's last 1000 + horizon steps, left-padded with is_pad. Per case the validation loss of every patch size
  (MoiraiForecast._val_loss, SampleNLLLoss), the patch size forward() picks (argmin) and that size's distribution on the last 1000 steps as above
  (no samples).

    python -I tools/timeseries-reference/reference_moirai.py --auto moirai-1.1-R-small
"""
import json, math, os, sys
import numpy as np
import torch

ROOT = os.path.expanduser("~/models/timeseries")
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SAMPLES = 20000


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


def moirai2(path):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    module = Moirai2Module.from_pretrained(path).eval()

    def forecast(history, horizon, levels):
        series = np.array(history, dtype=np.float32)
        model = Moirai2Forecast(module=module, prediction_length=horizon, context_length=min(len(series), 1680), target_dim=1,
                                feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).eval()
        native = model.predict([series])[0].astype(np.float64)   # [9, horizon]
        native = native.T
        return {"quantiles": select(native, levels), "mean": native[:, LEVELS.index(0.5)].tolist()}

    return forecast


def moirai1(path, patch_size=32):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    module = MoiraiModule.from_pretrained(path).eval()

    def forecast(history, horizon, levels):
        series = np.array(history, dtype=np.float32)
        context = min(len(series), 1000)
        series = series[-context:]
        observed = ~np.isnan(series)
        series = np.where(observed, series, 0).astype(np.float32)
        model = MoiraiForecast(module=module, prediction_length=horizon, context_length=context, patch_size=patch_size, num_samples=SAMPLES,
                               target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).eval()
        with torch.no_grad():
            distr = model._get_distr(patch_size, torch.tensor(series)[None, :, None], torch.tensor(observed)[None, :, None],
                                     torch.zeros(1, context, dtype=torch.bool))
            start = math.ceil(context / patch_size)
            rows = [start + t // patch_size for t in range(horizon)]
            cols = [t % patch_size for t in range(horizon)]

            def at(x):
                return x[0, rows, cols].double().tolist()

            mixture = distr.base_dist
            student, normal, negbin, lognormal = mixture.components
            params = {
                "weights": mixture.weights.probs[0, rows, cols].double().tolist(),
                "student_t": {"df": at(student.df), "loc": at(student.loc), "scale": at(student.scale)},
                "normal": {"loc": at(normal.loc)},
                "negative_binomial": {"total_count": at(negbin.total_count), "logits": at(negbin.logits)},
                "log_normal": {"loc": at(lognormal.loc), "scale": at(lognormal.scale)},
            }
            loc = float(distr.loc[0, rows[0], 0])
            scale = float(distr.scale[0, rows[0], 0])
            torch.manual_seed(0)
            samples = distr.sample(torch.Size((SAMPLES,)))[:, 0, rows, cols].double().numpy()   # [samples, horizon]
        return {"quantiles": np.quantile(samples, levels, axis=0).tolist(),   # the samples' empirical quantiles, as for every sampling model
                "mean": samples.mean(axis=0).tolist(), "loc": loc, "scale": scale, "params": params, "patch_size": patch_size}

    return forecast


def moirai1_auto(path, context_length=1000):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    module = MoiraiModule.from_pretrained(path).eval()

    def forecast(history, horizon, levels):
        series = np.array(history, dtype=np.float32)
        length = context_length + horizon
        window = series[-length:]
        pad = length - window.size
        observed = np.concatenate([np.zeros(pad, dtype=bool), ~np.isnan(window)])
        target = np.concatenate([np.zeros(pad, dtype=np.float32), np.where(np.isnan(window), 0, window)]).astype(np.float32)
        is_pad = np.concatenate([np.ones(pad, dtype=bool), np.zeros(window.size, dtype=bool)])
        model = MoiraiForecast(module=module, prediction_length=horizon, context_length=context_length, patch_size="auto", num_samples=100,
                               target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0).eval()
        assert model.past_length == length
        t, o, p = torch.tensor(target)[None, :, None], torch.tensor(observed)[None, :, None], torch.tensor(is_pad)[None, :]
        with torch.no_grad():
            losses = torch.stack([model._val_loss(patch_size=ps, target=t, observed_target=o, is_pad=p) for ps in module.patch_sizes])
            best = int(module.patch_sizes[int(losses.argmin(dim=0)[0])])
            distr = model._get_distr(best, t[:, -context_length:], o[:, -context_length:], p[:, -context_length:])
            entry = {"val_loss": losses[:, 0].double().tolist(), "patch_size": best}
            entry.update(distribution(distr, context_length, horizon, best))
        return entry

    return forecast


def distribution(distr, context, horizon, patch_size):
    start = math.ceil(context / patch_size)
    rows = [start + t // patch_size for t in range(horizon)]
    cols = [t % patch_size for t in range(horizon)]

    def at(x):
        return x[0, rows, cols].double().tolist()

    mixture = distr.base_dist
    student, normal, negbin, lognormal = mixture.components
    params = {
        "weights": mixture.weights.probs[0, rows, cols].double().tolist(),
        "student_t": {"df": at(student.df), "loc": at(student.loc), "scale": at(student.scale)},
        "normal": {"loc": at(normal.loc)},
        "negative_binomial": {"total_count": at(negbin.total_count), "logits": at(negbin.logits)},
        "log_normal": {"loc": at(lognormal.loc), "scale": at(lognormal.scale)},
    }
    return {"loc": float(distr.loc[0, rows[0], 0]), "scale": float(distr.scale[0, rows[0], 0]), "params": params}


def main():
    torch.set_num_threads(8)
    os.makedirs(ROOT + "/refs", exist_ok=True)
    auto = "--auto" in sys.argv[1:]
    for name in [a for a in sys.argv[1:] if a != "--auto"]:
        path = ROOT + "/" + name
        with open(path + "/config.json") as f:
            config = json.load(f)
        forecast = moirai1_auto(path) if auto else moirai2(path) if "quantile_levels" in config else moirai1(path)
        out = []
        for case, history, horizon in cases():
            levels = [0.05, 0.25, 0.5, 0.75, 0.95] if case == "interpolated_levels" else LEVELS
            entry = {
                "name": case,
                "history": [None if math.isnan(v) else float(v) for v in history],
                "horizon": horizon,
                "levels": levels,
            }
            entry.update(forecast(history, horizon, levels))
            out.append(entry)
            print(name, case, "ok", flush=True)
        with open(f"{ROOT}/refs/{name}{'-auto' if auto else ''}.json", "w") as f:
            json.dump({"model": name, "cases": out}, f)
        print(name, "ok")


if __name__ == "__main__":
    main()
