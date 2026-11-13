"""Reference data (from onnxruntime) for the "model zoo" end-to-end tests (ZooTests.cs).

usage: make_zoo_refs.py [name ...]     (default: every configured model whose file exists)
Needs: python with onnx, onnxruntime, numpy.  Models live in ~/models/onnx/<model path>; references are written to
~/models/onnx/ref/zoo/<name>.json + <name>.b<B>.in<k>.bin / .out<k>.bin (float32; integer tensors are stored as float32 values).

Each CONFIG entry: model path (relative to ~/models/onnx), optional per-input shape overrides (-1 = batch), per-input
generators, the batches to run, and comparison hints for the C# side:
  rtol / atol   tolerance (atol is multiplied by the output magnitude)
  outputs       optional {output name: {"mode": "argmax"|"skip"|"sorted", ...}} hints
"""
import json, os, sys
import numpy as np
import onnx, onnxruntime as ort

ROOT = os.path.expanduser("~/models/onnx")
OUT = os.path.join(ROOT, "ref", "zoo")


def smooth_image(rs, shape, lo=0.0, hi=1.0):
    """Low-frequency 'photo like' noise: upsampled coarse noise + a bit of fine noise, in [lo, hi]."""
    n, c, h, w = shape
    coarse = rs.rand(n, c, max(h // 16, 2), max(w // 16, 2)).astype(np.float32)
    ys = np.linspace(0, coarse.shape[2] - 1, h); xs = np.linspace(0, coarse.shape[3] - 1, w)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, coarse.shape[2] - 1); x1 = np.minimum(x0 + 1, coarse.shape[3] - 1)
    fy = (ys - y0)[None, None, :, None]; fx = (xs - x0)[None, None, None, :]
    a = coarse[:, :, y0][:, :, :, x0]; b = coarse[:, :, y0][:, :, :, x1]
    c_ = coarse[:, :, y1][:, :, :, x0]; d = coarse[:, :, y1][:, :, :, x1]
    img = (a * (1 - fx) + b * fx) * (1 - fy) + (c_ * (1 - fx) + d * fx) * fy
    img = 0.9 * img + 0.1 * rs.rand(n, c, h, w)
    return (lo + (hi - lo) * img).astype(np.float32)


# name -> config.  "gen": {input name or "*": generator spec}; spec = "randn" | "img" | "imgnorm" | ("ints", lo, hi) | ("const", v) | ("randn", scale)
CONFIG = {
    "realesrgan-x4v3": dict(model="realesrgan-x4v3/model.onnx", gen={"*": "img"}, rtol=2e-3),
    "yolov8n": dict(model="yolov8n/yolov8n.onnx", gen={"*": "img"}),
    "yolo11n": dict(model="yolo11n/yolo11n.onnx", gen={"*": "img"}),
    "yolov8n-pose": dict(model="yolov8n-pose/yolov8n-pose.onnx", gen={"*": "img"}),
    "detr-resnet50": dict(model="detr-resnet50/model.onnx", fixbatch=True, shapes={"pixel_values": [-1, 3, 384, 384], "pixel_mask": [-1, 64, 64]},
                          gen={"pixel_values": "imgnorm", "pixel_mask": ("const", 1)}),
    "rtdetr-r18": dict(model="rtdetr-r18/model.onnx", fixbatch=True, shapes={"pixel_values": [-1, 3, 640, 640]}, gen={"*": "img"}),
    "mobilesam-encoder": dict(model="mobilesam/mobilesam.encoder.onnx", shapes={"input_image": [512, 640, 3]}, gen={"*": "img255_hwc"}),
    "mobilesam-decoder": dict(model="mobilesam/mobilesam.decoder.onnx", shapes={"point_coords": [1, 3, 2], "point_labels": [1, 3]}, gen={
        "image_embeddings": ("randn", 0.5), "point_coords": ("coords", 1024), "point_labels": ("ints", 0, 2),
        "mask_input": ("const", 0), "has_mask_input": ("const", 0), "orig_im_size": ("vals", [512, 640])}, pin=["orig_im_size"]),
    "tiny-sd-unet": dict(model="tiny-sd/unet.onnx", fixbatch=True, shapes={"sample": [-1, 4, 16, 16], "timestep": [], "encoder_hidden_states": [-1, 8, 32]}, gen={"*": ("randn", 0.7), "timestep": ("const", 500)}),
    "tiny-sd-vae": dict(model="tiny-sd/vae_decoder.onnx", shapes={"latent_sample": [-1, 4, 16, 16]}, gen={"*": ("randn", 0.7)}),
    "tiny-sd-text": dict(model="tiny-sd/text_encoder.onnx", fixbatch=True, shapes={"input_ids": [-1, 8]}, gen={"*": ("ints", 1, 1000)}),
    "sdxl-vae": dict(model="sdxl-vae/model.onnx", shapes={"latent_sample": [-1, 4, 32, 32]}, gen={"*": ("randn", 0.7)}, rtol=5e-3),
    "whisper-decoder": dict(model="whisper-tiny/decoder_model.onnx", shapes={"input_ids": [-1, 8], "encoder_hidden_states": [-1, 1500, 384]},
                            gen={"input_ids": ("ints", 1, 1000), "encoder_hidden_states": ("randn", 0.5)}),
    "clip-vision": dict(model="clip-b32/vision_model.onnx", fixbatch=True, shapes={"pixel_values": [-1, 3, 224, 224]}, gen={"*": "imgnorm"}),
    "clip-text": dict(model="clip-b32/text_model.onnx", fixbatch=True, shapes={"input_ids": [-1, 16]}, gen={"input_ids": ("ints", 1, 1000)}),
    "siglip-vision": dict(model="siglip-b16/vision_model.onnx", fixbatch=True, shapes={"pixel_values": [-1, 3, 224, 224]}, gen={"*": "imgnorm"}),
    "siglip-text": dict(model="siglip-b16/text_model.onnx", fixbatch=True, shapes={"input_ids": [-1, 16]}, gen={"input_ids": ("ints", 1, 1000)}),
    "convnext-tiny": dict(model="convnext-tiny/model.onnx", shapes={"pixel_values": [-1, 3, 224, 224]}, gen={"*": "imgnorm"}),
    "swin-tiny": dict(model="swin-tiny/model.onnx", fixbatch=True, shapes={"pixel_values": [-1, 3, 224, 224]}, gen={"*": "imgnorm"}),
    "swin2sr": dict(model="swin2sr/model.onnx", fixbatch=True, shapes={"pixel_values": [-1, 3, 64, 64]}, gen={"*": "img"}),
    "wav2vec2-base": dict(model="wav2vec2-base/model.onnx", fixbatch=True, shapes={"input_values": [-1, 16000]}, gen={"*": ("randn", 0.3)}),
    "gpt2": dict(model="gpt2/decoder_model.onnx", fixbatch=True, shapes={"input_ids": [-1, 12], "attention_mask": [-1, 12]},
                 gen={"input_ids": ("ints", 1, 5000), "attention_mask": ("const", 1)}),
    "animegan-v2": dict(model="animegan-v2/face_paint_512_v2_0.onnx", gen={"*": "imgpm1"}, rtol=1e-2, atol=4e-3),  # InstanceNorm epsilon=0 on 512x512 maps: f32 reduction order shows (GPU ~2.5e-3)
    "animegan-hayao": dict(model="animegan-hayao/generator_hayao.onnx", gen={"*": "imgpm1"}),
    "mosaic-9": dict(model="mosaic-9/mosaic-9.onnx", gen={"*": ("img255",)}),
    "raft-large": dict(model="raft-large/raft_large.onnx", rms=3e-2, gen={"*": "imgnorm"}),
    "tiny-mamba": dict(model="tiny-mamba/model.onnx", fixbatch=True, shapes={"input_ids": [1, 10], "in_conv_states": [2, 1, 32, 4], "in_ssm_states": [2, 1, 32, 16]}, gen={"input_ids": ("ints", 1, 100), "*": ("randn", 0.3)}),
    "silero-vad": dict(model="silero-vad/model.onnx", shapes={"input": [1, 576], "state": [2, 1, 128]}, fixbatch=True, pin=["sr"], gen={"input": ("randn", 0.1), "sr": ("const", 16000), "state": ("const", 0)}),
    "ppocr-det": dict(model="ppocr-v4/ch_PP-OCRv4_det_infer.onnx", shapes={"x": [-1, 3, 320, 320]}, gen={"*": "imgnorm"}),
    "ppocr-rec": dict(model="ppocr-v4/ch_PP-OCRv4_rec_infer.onnx", fixbatch=True, shapes={"x": [-1, 3, 48, 320]}, gen={"*": "imgnorm"}),
    "ppocr-rec-en-v3": dict(model="ppocr-v4/en_PP-OCRv3_rec_infer.onnx", fixbatch=True, shapes={"x": [-1, 3, 48, 320]}, gen={"*": "imgnorm"}),
    "ppocr-rec-v2": dict(model="ppocr-v4/ch_ppocr_mobile_v2.0_rec_infer.onnx", fixbatch=True, shapes={"x": [-1, 3, 32, 320]}, gen={"*": "imgnorm"}),
    "moonshine-encoder": dict(model="moonshine-tiny/encoder_model.onnx", shapes={"input_values": [-1, 16000]}, gen={"*": ("randn", 0.3)}),
    "yolos-tiny": dict(model="yolos-tiny/model.onnx", shapes={"pixel_values": [-1, 3, 320, 320]}, gen={"*": "imgnorm"}),
    "yolov10n-int8": dict(model="yolov10n-int8/model_int8.onnx", mode="scores6", fixbatch=True, rms=0.25, gen={"*": "img"}),
    "resnet50-int8": dict(model="resnet50-int8/resnet50-v1-12-int8.onnx", rms=0.25, gen={"*": "imgnorm"}),
    "mobilenetv2-int8": dict(model="mobilenetv2-int8/mobilenetv2-12-int8.onnx", rms=0.25, gen={"*": "imgnorm"}),
    "clip-vision-int8": dict(model="clip-b32/vision_model_int8.onnx", rms=0.25, fixbatch=True, shapes={"pixel_values": [-1, 3, 224, 224]}, gen={"*": "imgnorm"}),
    "convnext-tiny-q": dict(model="convnext-tiny/model_quantized.onnx", fixbatch=True, rms=0.25, shapes={"pixel_values": [-1, 3, 224, 224]}, gen={"*": "imgnorm"}),
    "mobilevit-small": dict(model="mobilevit-small/model.onnx", fixbatch=True, shapes={"pixel_values": [-1, 3, 256, 256]}, gen={"*": "img"}),
    "dinov2-small": dict(model="dinov2-small/model.onnx", fixbatch=True, shapes={"pixel_values": [-1, 3, 224, 224]}, gen={"*": "imgnorm"}),
    "depth-anything-v2-small": dict(model="depth-anything-v2-small/model.onnx", fixbatch=True, shapes={"pixel_values": [-1, 3, 252, 252]}, gen={"*": "imgnorm"}),
    "nomic-moe": dict(model="nomic-moe/model_int8.onnx", rms=0.25, fixbatch=True, shapes={"input_ids": [-1, 16], "attention_mask": [-1, 16]}, gen={"input_ids": ("ints", 1, 1000), "attention_mask": ("const", 1)}),
    "sam2-encoder": dict(model="sam2-tiny/vision_encoder.onnx", fixbatch=True, gen={"*": "imgnorm"}),
}


# decoders with a KV cache (past length fixed to 12 / 8 tokens; batch fixed to 1)
_gpt2_past = {f"past_key_values.{l}.{kv}": [1, 12, 12, 64] for l in range(12) for kv in ("key", "value")}
CONFIG["gpt2-past"] = dict(model="gpt2/decoder_with_past_model.onnx", fixbatch=True, shapes={**_gpt2_past, "input_ids": [1, 1], "attention_mask": [1, 13]},
                           gen={"*": ("randn", 0.5), "input_ids": ("ints", 1, 5000), "attention_mask": ("const", 1)})
_wh_past = {}
for _l in range(4):
    _wh_past[f"past_key_values.{_l}.decoder.key"] = [1, 6, 8, 64]; _wh_past[f"past_key_values.{_l}.decoder.value"] = [1, 6, 8, 64]
    _wh_past[f"past_key_values.{_l}.encoder.key"] = [1, 6, 1500, 64]; _wh_past[f"past_key_values.{_l}.encoder.value"] = [1, 6, 1500, 64]
CONFIG["whisper-decoder-past"] = dict(model="whisper-tiny/decoder_with_past_model.onnx", fixbatch=True, shapes={**_wh_past, "input_ids": [1, 1]},
                                      gen={"*": ("randn", 0.5), "input_ids": ("ints", 1, 1000)})
CONFIG["gpt2-merged"] = dict(model="gpt2/decoder_model_merged.onnx", fixbatch=True, pin=["use_cache_branch"],
                             shapes={**_gpt2_past, "input_ids": [1, 1], "attention_mask": [1, 13], "use_cache_branch": [1]},
                             gen={"*": ("randn", 0.5), "input_ids": ("ints", 1, 5000), "attention_mask": ("const", 1), "use_cache_branch": ("const", 1)})


def make(rs, spec, shape, elem):
    kind = spec[0] if isinstance(spec, tuple) else spec
    if kind == "img":   a = smooth_image(rs, shape)
    elif kind == "imgnorm": a = (smooth_image(rs, shape) - 0.45) / 0.25
    elif kind == "imgpm1": a = smooth_image(rs, shape) * 2 - 1
    elif kind == "img255_hwc": a = smooth_image(rs, (1, 3) + tuple(shape[:2]), 0, 255)[0].transpose(1, 2, 0)
    elif kind == "img255": a = smooth_image(rs, shape, 0, 255)
    elif kind == "randn": a = rs.randn(*shape) * (spec[1] if isinstance(spec, tuple) else 0.7)
    elif kind == "ints": a = rs.randint(spec[1], spec[2], size=shape)
    elif kind == "const": a = np.full(shape, spec[1])
    elif kind == "vals": a = np.array(spec[1]).reshape(shape)
    elif kind == "coords": a = rs.rand(*shape) * spec[1]
    else: raise ValueError(spec)
    return a.astype({1: np.float32, 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64}.get(elem, np.float32))


def run(name, cfg):
    path = os.path.join(ROOT, cfg["model"])
    if not os.path.exists(path):
        print("missing", path); return
    m = onnx.load(path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    ins = [i for i in m.graph.input if i.name not in inits]
    shapes = []
    for i in ins:
        sh = [(-1 if (d.dim_param or d.dim_value == 0) else d.dim_value) for d in i.type.tensor_type.shape.dim]
        ov = cfg.get("shapes", {})
        sh = ov.get(i.name, ov.get("*", sh))
        shapes.append(sh)
    if any(d == -1 for s in shapes for d in s[1:]):
        print("dynamic non-batch dims, add shape override:", name, [(i.name, s) for i, s in zip(ins, shapes)]); return
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    outs = [o.name for o in sess.get_outputs()]
    batches = cfg.get("batches") or ([1, 2] if shapes[0][0] == -1 else [1])
    os.makedirs(OUT, exist_ok=True)
    rs = np.random.RandomState(7)
    gen = cfg.get("gen", {})
    consts = {}
    for b in batches:
        feeds = {}
        for k, (i, s) in enumerate(zip(ins, shapes)):
            shp = [b if d == -1 else d for d in s]
            a = make(rs, gen.get(i.name, gen.get("*", "randn")), shp, i.type.tensor_type.elem_type)
            feeds[i.name] = a
            if i.name in cfg.get("pin", []): consts[i.name] = a
            a.astype(np.float32).tofile(f"{OUT}/{name}.b{b}.in{k}.bin")
        res = sess.run(outs, feeds)
        for k, o in enumerate(res):
            np.ascontiguousarray(o, dtype=np.float32).tofile(f"{OUT}/{name}.b{b}.out{k}.bin")
        print(name, "batch", b, [(r.shape, str(r.dtype)) for r in res])
    json.dump({"model": cfg["model"], "batches": batches,
               "inputs": [{"name": i.name, "shape": s, "elem": i.type.tensor_type.elem_type} for i, s in zip(ins, shapes)],
               "outputs": [{"name": o} for o in outs], "rtol": cfg.get("rtol", 1e-3), "atol": cfg.get("atol", 2e-4), "fixbatch": cfg.get("fixbatch", False), "rms": cfg.get("rms"), "mode": cfg.get("mode"), "constants": {k: np.asarray(consts[k]).ravel().astype(float).tolist() for k in cfg.get("pin", [])},
               "hints": cfg.get("hints", {})}, open(f"{OUT}/{name}.json", "w"), indent=1)


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n, c in CONFIG.items() if os.path.exists(os.path.join(ROOT, c["model"]))]
    for n in names:
        try: run(n, CONFIG[n])
        except Exception as e: print("FAILED", n, type(e).__name__, str(e)[:300])
