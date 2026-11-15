"""Compatibility shims for remote modeling code written against transformers 4.x when it runs on transformers 5 (imported by
mk_tiny_remote_code.py and hf_greedy_reference.py before any remote module is loaded).

- `_tied_weights_keys` as a list (4.x: "these weights are tied to the input embeddings") becomes the 5.x mapping {tied weight: embedding weight}.
- The mask builders of transformers.masking_utils accept the 4.x keyword `input_embeds` (5.x: `inputs_embeds`) and ignore `cache_position`,
  which 5.x derives from the cache.
"""
import functools

import transformers.masking_utils as masking
from transformers import PreTrainedModel

_original_tied_keys = PreTrainedModel.get_expanded_tied_weights_keys


def _tied_keys(self, *args, **kwargs):
    keys = getattr(self, "_tied_weights_keys", None)
    if isinstance(keys, (list, tuple)):
        embedding = self.get_input_embeddings()
        name = next(n for n, m in self.named_modules() if m is embedding) + ".weight"
        self._tied_weights_keys = {k: name for k in keys}
    return _original_tied_keys(self, *args, **kwargs)


PreTrainedModel.get_expanded_tied_weights_keys = _tied_keys


def _accept_4x_keywords(builder):
    @functools.wraps(builder)
    def wrapped(*args, **kwargs):
        if "input_embeds" in kwargs:
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        kwargs.pop("cache_position", None)
        return builder(*args, **kwargs)
    return wrapped


for _name in ("create_causal_mask", "create_sliding_window_causal_mask", "create_chunked_causal_mask"):
    if hasattr(masking, _name):
        setattr(masking, _name, _accept_4x_keywords(getattr(masking, _name)))
