"""Hub tiny-random checkpoints that ship only pytorch_model.bin get a model.safetensors next to it (the importers read safetensors only).
Loads with weights_only=True (no pickle code execution)."""
import glob, os, sys, torch
from safetensors.torch import save_file
for d in sorted(glob.glob(os.path.join(sys.argv[1], "tiny*"))):
    b = os.path.join(d, "pytorch_model.bin")
    if glob.glob(os.path.join(d, "*.safetensors")) or not os.path.exists(b):
        continue
    sd = torch.load(b, map_location="cpu", weights_only=True)
    seen = {}
    out = {}
    for k, v in sd.items():
        if v.data_ptr() in seen:
            v = v.clone()
        seen[v.data_ptr()] = k
        out[k] = v.contiguous()
    save_file(out, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    print("converted", os.path.basename(d), flush=True)
