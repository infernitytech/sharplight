"""Downloads the survey models into ~/models/onnx/<name>/ (HF hub).  usage: download_models.py [name ...]"""
import os, sys, shutil
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download, HfApi

ROOT = os.path.expanduser("~/models/onnx")
Z = "onnxmodelzoo/"
# name -> (repo, [files])  ; files land in ROOT/name/<basename>  (external-data files must be listed explicitly)
M = {
 # --- ONNX model zoo (classic, opsets 1-12) ---
 "ssd-10": (Z+"ssd-10", ["ssd-10.onnx"]),
 "ssd-12-int8": (Z+"ssd-12-int8", ["ssd-12-int8.onnx"]),
 "retinanet-9": (Z+"retinanet-9", ["retinanet-9.onnx"]),
 "fasterrcnn-12": (Z+"FasterRCNN-12", ["FasterRCNN-12.onnx"]),
 "maskrcnn-12": (Z+"MaskRCNN-12", ["MaskRCNN-12.onnx"]),
 "arcfaceresnet100-8": (Z+"arcfaceresnet100-8", ["arcfaceresnet100-8.onnx"]),
 "emotion-ferplus-8": (Z+"emotion-ferplus-8", ["emotion-ferplus-8.onnx"]),
 "super-resolution-10": (Z+"super-resolution-10", ["super-resolution-10.onnx"]),
 "candy-9": (Z+"candy-9", ["candy-9.onnx"]),
 "version-rfb-320": (Z+"version-RFB-320", ["version-RFB-320.onnx"]),
 "tinyyolov2-8": (Z+"tinyyolov2-8", ["tinyyolov2-8.onnx"]),
 "yolov3-12": (Z+"yolov3-12", ["yolov3-12.onnx"]),
 "shufflenet-v2-12": (Z+"shufflenet-v2-12", ["shufflenet-v2-12.onnx"]),
 "densenet-9": (Z+"densenet-9", ["densenet-9.onnx"]),
 "resnet18-v1-7": (Z+"resnet18-v1-7", ["resnet18-v1-7.onnx"]),
 "mnist-1": (Z+"mnist-1", ["mnist-1.onnx"]),
 "bertsquad-12": (Z+"bertsquad-12", ["bertsquad-12.onnx"]),
 "bertsquad-12-int8": (Z+"bertsquad-12-int8", ["bertsquad-12-int8.onnx"]),
 "roberta-seqcls-9": (Z+"roberta-sequence-classification-9", ["roberta-sequence-classification-9.onnx"]),
 "gpt2-lm-head-10": (Z+"gpt2-lm-head-10", ["gpt2-lm-head-10.onnx"]),
 "resnet101-duc-12": (Z+"ResNet101-DUC-12", ["ResNet101-DUC-12.onnx"]),
 "efficientnet-lite4-int8": (Z+"efficientnet-lite4-11-int8", ["efficientnet-lite4-11-int8.onnx"]),
 "bidaf-9": (Z+"bidaf-9", ["bidaf-9.onnx"]),
 # --- HF Optimum / transformers.js exports ---
 "bert-base-uncased": ("Xenova/bert-base-uncased", ["onnx/model.onnx", "onnx/model_quantized.onnx", "onnx/model_q4.onnx", "onnx/model_fp16.onnx"]),
 "roberta-base": ("Xenova/roberta-base", ["onnx/model.onnx"]),
 "deberta-v3-small": ("onnx-community/deberta-v3-small", ["onnx/model.onnx"]),
 "modernbert-base": ("onnx-community/ModernBERT-base-ONNX", ["onnx/model.onnx"]),
 "t5-small": ("Xenova/t5-small", ["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx"]),
 "distilbart-cnn": ("Xenova/distilbart-cnn-6-6", ["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx"]),
 "qwen2.5-0.5b": ("onnx-community/Qwen2.5-0.5B-Instruct", ["onnx/model_q4.onnx", "onnx/model_fp16.onnx", "onnx/model_quantized.onnx"]),
 "gemma3-270m": ("onnx-community/gemma-3-270m-it-ONNX", ["onnx/model.onnx"]),
 "lfm2-350m": ("onnx-community/LFM2-350M-ONNX", ["onnx/model.onnx"]),
 "phi35-mini-q4f16": ("onnx-community/Phi-3.5-mini-instruct-onnx-web", ["onnx/model_q4f16.onnx"]),
 "florence2-base": ("onnx-community/Florence-2-base", ["onnx/vision_encoder.onnx", "onnx/embed_tokens.onnx", "onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx"]),
 "trocr-small-printed": ("Xenova/trocr-small-printed", ["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx"]),
 "vit-gpt2-caption": ("Xenova/vit-gpt2-image-captioning", ["onnx/encoder_model.onnx", "onnx/decoder_model_merged.onnx"]),
 "dpt-hybrid-midas": ("Xenova/dpt-hybrid-midas", ["onnx/model.onnx"]),
 "depth-anything-v2-base": ("onnx-community/depth-anything-v2-base", ["onnx/model.onnx"]),
 "grounding-dino-tiny": ("onnx-community/grounding-dino-tiny-ONNX", ["onnx/model.onnx"]),
 "detr-panoptic": ("Xenova/detr-resnet-50-panoptic", ["onnx/model.onnx"]),
 "vitpose-base": ("onnx-community/vitpose-base-simple", ["onnx/model.onnx"]),
 "kokoro-82m": ("onnx-community/Kokoro-82M-ONNX", ["onnx/model.onnx"]),
 "gliner-small": ("onnx-community/gliner_small-v2.1", ["onnx/model.onnx"]),
 "modnet": ("Xenova/modnet", ["onnx/model.onnx"]),
 "mobilevitv2": ("Xenova/mobilevitv2-1.0-imagenet1k-256", ["onnx/model.onnx"]),
 "yolo26n": ("onnx-community/yolo26n-ONNX", ["onnx/model.onnx"]),
 # --- detection / pose / face / audio / OCR ---
 "yolov8n-seg": ("mobilint/YOLOv8n-seg", ["yolov8n-seg.onnx"]),
 "yolo11n-face": ("AdamCodd/YOLOv11n-face-detection", ["model.onnx"]),
 "scrfd-10g": ("immich-app/buffalo_l", ["detection/model.onnx"]),
 "arcface-w600k-r50": ("immich-app/buffalo_l", ["recognition/model.onnx"]),
 "rtmpose-m": ("AXERA-TECH/RTMPose", ["rtmpose_m_256x192.onnx"]),
 "openunmix-vocals": ("nsosu/demucs-onnx", ["umxl_vocals.onnx"]),
 "u2netp": ("BritishWerewolf/U-2-Netp", ["onnx/model.onnx"]),
 "ppocrv5-det": ("marsena/paddleocr-onnx-models", ["PP-OCRv5_server_det_infer.onnx"]),
 "uvdoc": ("marsena/paddleocr-onnx-models", ["UVDoc_infer.onnx"]),
 "zipformer-encoder": ("csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26", ["encoder-epoch-99-avg-1-chunk-16-left-64.onnx"]),
}

def get(name):
    repo, files = M[name]
    d = os.path.join(ROOT, name); os.makedirs(d, exist_ok=True)
    api = HfApi()
    for f in files:
        dst = os.path.join(d, os.path.basename(f))
        if os.path.exists(dst): continue
        try:
            p = hf_hub_download(repo, f, cache_dir=os.path.expanduser("~/.cache/hf-survey"))
            shutil.copy(os.path.realpath(p), dst)
            # external data (<file>_data / .data) next to it
            for ext in ("_data", ".data"):
                try:
                    q = hf_hub_download(repo, f + ext, cache_dir=os.path.expanduser("~/.cache/hf-survey"))
                    shutil.copy(os.path.realpath(q), dst + ext)
                except Exception: pass
            print("ok", name, f, flush=True)
        except Exception as e:
            print("FAIL", name, f, str(e)[:150], flush=True)

if __name__ == "__main__":
    names = sys.argv[1:] or list(M)
    with ThreadPoolExecutor(6) as ex: list(ex.map(get, names))
