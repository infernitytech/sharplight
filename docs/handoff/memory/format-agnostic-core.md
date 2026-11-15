---
name: format-agnostic-core
description: Sharplight must stay format-agnostic; GGUF (and every other input format) lives in its own assemblies referencing only Core
metadata:
  node_type: memory
  type: feedback
  modified: 2026-11-12T20:37:34.063Z
---

GGUF is just one input format. Format support (GGUF, HF safetensors, ONNX, ...) lives in its own assemblies that reference only Sharplight.Core; nothing else (Llm, Architectures, Tokenizers, providers) may reference a format assembly — hosts opt in via `XxxImporter.Register()`.

**Why:** User stated (2026-11-12) "make sure that gguf is just one input format" and "GGUF support should be in its own set of assemblies only referencing the core engine." Also wants architecture generality ([[keep-architecture-general]]).

**How to apply:** Importers translate into the canonical ModelSource (CanonicalKeys, canonical weight names/semantics, via WeightInfo.Derived for transforms). Never read `tokenizer.ggml.*` or other format-specific keys outside the importer. When adding features (vision mmproj, MTP), keep the format knowledge in the importer assembly.

Canonical names are Sharplight's OWN vocabulary (e.g. arch `qwen3_5`, `layers.{i}.attention.q.weight`, `qwen3_5.hidden_size`), not llama.cpp's. User asked 2026-11-13: "we should have our own names and not dependent on llama.cpp naming". GGUF maps via `IGgufArchitectureMapping` (src/Sharplight.Gguf/GgufArchitectureMapping.cs), HF via its HfArchitectureMapping; the vocabulary is documented in Qwen35Adapter remarks.
