"""Converts the tiny CogVLM of mk_tiny_cogvlm.py to an f32 GGUF and an f32 mmproj with llama.cpp's converter.

Usage: python convert_tiny_cogvlm.py <llama.cpp source dir> <model dir> <out.gguf> <out mmproj.gguf>

Two workarounds for llama.cpp 8ca5a80's converter: its text-model tensor filter skips every name containing "vision_", which drops the visual
experts (model.layers.N.self_attn.vision_expert_* and mlp.vision_mlp.*) that its own loader requires as blk.N.vis_*, so they are let through;
and LlamaModel needs num_key_value_heads, which CogVLM's config does not have (multi-head attention: the head count is added). The tower's name
map is widened as in convert_tiny_mmproj.py (the projector's numbered modules exceed the tiny tower's layer count).
"""
import json
import os
import re
import shutil
import sys
import tempfile

llama, model, out, out_mmproj = sys.argv[1:5]
sys.path.insert(0, os.path.join(llama, "gguf-py"))
sys.path.insert(0, llama)
import gguf  # noqa: E402
from conversion import cogvlm  # noqa: E402

original_map = gguf.get_tensor_name_map
gguf.get_tensor_name_map = lambda arch, n_blocks: original_map(arch, max(n_blocks, 8))

original_filter = cogvlm.CogVLMModel.filter_tensors
experts = re.compile(r"^model\.layers\.\d+\.(self_attn\.vision_expert_|mlp\.vision_mlp\.)")


def keep_experts(cls, item):
    return item if experts.match(item[0]) else original_filter(item)


cogvlm.CogVLMModel.filter_tensors = classmethod(keep_experts)

with open(os.path.join(llama, "convert_hf_to_gguf.py")) as f:
    code = compile(f.read(), "convert_hf_to_gguf.py", "exec")

with tempfile.TemporaryDirectory() as work:
    for name in os.listdir(model):
        if not name.startswith("reference_"):
            shutil.copy(os.path.join(model, name), work)
    path = os.path.join(work, "config.json")
    config = json.load(open(path))
    config.setdefault("num_key_value_heads", config["num_attention_heads"])
    json.dump(config, open(path, "w"), indent=2)
    preprocessor = os.path.join(work, "preprocessor_config.json")
    if not os.path.exists(preprocessor):
        # The checkpoint's remote code normalizes with the CLIP statistics; the converter reads them from here.
        json.dump({"image_mean": [0.48145466, 0.4578275, 0.40821073], "image_std": [0.26862954, 0.26130258, 0.27577711],
                   "size": {"height": config["vision_config"]["image_size"], "width": config["vision_config"]["image_size"]}}, open(preprocessor, "w"))
    for args in ([work, "--outtype", "f32", "--outfile", out], [work, "--mmproj", "--outtype", "f32", "--outfile", out_mmproj]):
        sys.argv = ["convert_hf_to_gguf.py", *args]
        try:
            exec(code, {"__name__": "__main__", "__file__": os.path.join(llama, "convert_hf_to_gguf.py")})
        except SystemExit as e:
            if e.code:
                raise
