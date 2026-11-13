# Sharplight.GgufDump (`sharplight-ggufdump`)

Prints the header, metadata, optionally every tensor, and a per-type size histogram of a GGUF file. Pass the first
shard of a split model.

```bash
dotnet run -c Release --project samples/Sharplight.GgufDump -- ~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf
dotnet run -c Release --project samples/Sharplight.GgufDump -- ~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf --tensors
```

Long strings and arrays in the metadata are abbreviated. Exit codes: `0` success, `1` invalid command line, `2` file error.
