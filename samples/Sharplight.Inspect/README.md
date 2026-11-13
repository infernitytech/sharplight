# Sharplight.Inspect (`sharplight-inspect`)

Loads a model through the adapter layer (no device needed) and prints its weights/states summary, an op histogram and the
first and last lines of one compute graph.

```bash
dotnet run -c Release --project samples/Sharplight.Inspect -- ~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf          # decode graph
dotnet run -c Release --project samples/Sharplight.Inspect -- ~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf embed    # another graph
```

An unknown graph name lists the available ones.
