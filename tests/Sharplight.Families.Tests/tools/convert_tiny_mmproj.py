"""Runs llama.cpp's convert_hf_to_gguf.py --mmproj on a tiny checkpoint whose vision tower has fewer layers than the projector has numbered
modules (PaddleOCR-VL's mlp_AR.linear_2 with a 2-layer tower: the converter's name map only covers block ids below the tower's layer count).
The name map is widened; the written file is what the converter would write for a deeper tower.

Usage: python convert_tiny_mmproj.py <llama.cpp source dir> <model dir> <out.gguf>

PaddleOCR-VL checkpoints also need vision_config.architectures = ["PaddleOCRVisionModel"] (the Hub's config has it; transformers drops it),
and llama.cpp reads min_pixels / max_pixels from preprocessor_config.json.
"""
import os
import sys

llama, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, os.path.join(llama, "gguf-py"))
sys.path.insert(0, llama)
import gguf  # noqa: E402

original = gguf.get_tensor_name_map
gguf.get_tensor_name_map = lambda arch, n_blocks: original(arch, max(n_blocks, 8))
sys.argv = ["convert_hf_to_gguf.py", model, "--mmproj", "--outtype", "f32", "--outfile", out]
with open(os.path.join(llama, "convert_hf_to_gguf.py")) as f:
    code = compile(f.read(), "convert_hf_to_gguf.py", "exec")
exec(code, {"__name__": "__main__", "__file__": os.path.join(llama, "convert_hf_to_gguf.py")})
