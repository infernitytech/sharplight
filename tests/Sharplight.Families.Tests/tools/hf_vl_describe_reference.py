"""Image-prompt reference of a Hugging Face vision-language checkpoint (transformers, fp32 on the CPU).

Usage: python hf_vl_describe_reference.py <model_dir> <image.png> [<new_tokens>]

Renders [{"role": "user", "content": [image, "Describe this image."]}] with the checkpoint's chat template and processor (resize,
normalization), prints the number of image tokens, the top-5 logits of the last prompt row and the greedy continuation (ids and text).
The Vision tests compare these with Sharplight's own preprocessing of the same picture.
"""
import sys

import torch
import transformers
from PIL import Image
from transformers import AutoConfig, AutoProcessor

model_dir, image_path = sys.argv[1], sys.argv[2]
new_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 32
config = AutoConfig.from_pretrained(model_dir)
model = getattr(transformers, config.architectures[0]).from_pretrained(model_dir, dtype=torch.float32).eval()
processor = AutoProcessor.from_pretrained(model_dir)
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text], images=[Image.open(image_path).convert("RGB")], return_tensors="pt")
print("prompt rows", inputs["input_ids"].shape[1], "image grid", inputs["image_grid_thw"].tolist())
with torch.no_grad():
    logits = model(**inputs).logits[0, -1]
    top = torch.topk(logits, 5)
    print("top", [(i, round(v, 4)) for i, v in zip(top.indices.tolist(), top.values.tolist())])
    out = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)[0, inputs["input_ids"].shape[1]:].tolist()
print("greedy", out)
print(repr(processor.tokenizer.decode(out)))
