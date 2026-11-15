"""Downloads the parity_bench.py models that are not part of the survey set and writes their onnxruntime references.

usage: fetch_parity_models.py [name ...]      (default: all; run with the ORT venv's python, see bench/onnx/env.sh)

Models land in ~/models/onnx (top-level models as <name>.onnx, zoo models under <dir>/), references in
~/models/onnx/ref/<name>.json (top-level, the ReferenceTests layout) or ref/zoo/<name>.json (ZooTests layout, written by
tests/Sharplight.Import.Onnx.Tests/scripts/make_zoo_refs.py with its CONFIG). Survey models (ssd-10, bert, qwen2.5, phi-3.5,
R-CNNs) come from download_models.py and make_survey_refs.py instead.

whisper-tiny-encoder: Xenova's export also returns the attention probabilities of every layer; only last_hidden_state is
kept, as in the export measured before.

mobilesam-encoder: the samexporter export used before (opset 16) is not on the Hugging Face hub; Acly/MobileSAM's encoder
has the same interface (an HWC uint8-range image of any size) at opset 17.
"""
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/models/onnx")
HF = "https://huggingface.co/{repo}/resolve/main/{file}"

# name -> (repo, file in repo, destination relative to ROOT)
FILES = {
    "resnet50-v1-12": ("onnxmodelzoo/resnet50-v1-12", "resnet50-v1-12.onnx", "resnet50-v1-12.onnx"),
    "mobilenetv2-12": ("onnxmodelzoo/mobilenetv2-12", "mobilenetv2-12.onnx", "mobilenetv2-12.onnx"),
    "efficientnet-lite4-11": ("onnxmodelzoo/efficientnet-lite4-11", "efficientnet-lite4-11.onnx", "efficientnet-lite4-11.onnx"),
    "vit-base-patch16-224": ("Xenova/vit-base-patch16-224", "onnx/model.onnx", "vit-base-patch16-224.onnx"),
    "whisper-tiny-encoder": ("Xenova/whisper-tiny", "onnx/encoder_model.onnx", "whisper-tiny-encoder.onnx"),
    "segformer-b0": ("Xenova/segformer-b0-finetuned-ade-512-512", "onnx/model.onnx", "segformer-b0.onnx"),
    "yolov10n": ("onnx-community/yolov10n", "onnx/model.onnx", "yolov10n.onnx"),
    "convnext-tiny": ("Xenova/convnext-tiny-224", "onnx/model.onnx", "convnext-tiny/model.onnx"),
    "yolov8n": ("webml/yolov8n", "onnx/yolov8n.onnx", "yolov8n/yolov8n.onnx"),
    "mobilesam-encoder": ("Acly/MobileSAM", "mobile_sam_image_encoder.onnx", "mobilesam/mobilesam.encoder.onnx"),
    "raft-large": ("nev/raft-large-onnx", "raft_large.onnx", "raft-large/raft_large.onnx"),
}

# top-level references (ref/<name>.json): input shape overrides (-1 = batch) and generators as in make_zoo_refs.py
TOP = {
    "resnet50-v1-12": dict(shapes={"*": [-1, 3, 224, 224]}, gen={"*": "imgnorm"}),
    "mobilenetv2-12": dict(shapes={"*": [-1, 3, 224, 224]}, gen={"*": "imgnorm"}),
    "efficientnet-lite4-11": dict(gen={"*": "img"}, batches=[1]),
    "vit-base-patch16-224": dict(shapes={"*": [-1, 3, 224, 224]}, gen={"*": "imgnorm"}),
    "whisper-tiny-encoder": dict(shapes={"*": [-1, 80, 3000]}, gen={"*": ("randn", 0.5)}, fixbatch=True, keep_outputs=1),
    "segformer-b0": dict(shapes={"*": [-1, 3, 512, 512]}, gen={"*": "imgnorm"}),
    "yolov10n": dict(gen={"*": "img"}, batches=[1]),
}


def download(name):
    repo, file, dest = FILES[name]
    path = os.path.join(ROOT, dest)
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("downloading", repo, file, flush=True)
    urllib.request.urlretrieve(HF.format(repo=repo, file=file), path + ".part")
    os.replace(path + ".part", path)
    keep = TOP.get(name, {}).get("keep_outputs")
    if keep:
        import onnx
        model = onnx.load(path)
        del model.graph.output[keep:]
        onnx.save(model, path)


def main():
    names = sys.argv[1:] or list(FILES)
    for n in names:
        download(n)
    sys.path.insert(0, os.path.join(HERE, "../../tests/Sharplight.Import.Onnx.Tests/scripts"))
    import make_zoo_refs as z
    for n in names:
        if n in TOP:
            z.OUT = os.path.join(ROOT, "ref")
            z.run(n, dict(TOP[n], model=FILES[n][2]))
        else:
            z.OUT = os.path.join(ROOT, "ref", "zoo")
            z.run(n, z.CONFIG[n])


if __name__ == "__main__":
    main()
