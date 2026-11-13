# Sharplight.OnnxRun (`sharplight-onnxrun`)

Imports an ONNX model, compiles it for the CPU or Vulkan provider, runs it with random or file input and prints import and
compile times, run timings (min, median, mean after a warm-up) and the top-5 of the first output's first batch item.

```bash
dotnet run -c Release --project samples/Sharplight.OnnxRun -- ~/models/onnx/mobilenetv2-12.onnx
dotnet run -c Release --project samples/Sharplight.OnnxRun -- ~/models/onnx/mnist-12.onnx --provider vulkan --batch 8 --runs 20

# Dynamic spatial dimensions need a full shape; a dimension <= 0 is the batch axis
dotnet run -c Release --project samples/Sharplight.OnnxRun -- model.onnx --shape pixel_values=-1x3x384x384

# Real input (raw float32) and output capture
dotnet run -c Release --project samples/Sharplight.OnnxRun -- model.onnx --input x.bin --out y.bin --dump-graph
```

Exit codes: `0` success, `1` invalid command line, `2` runtime error (provider unavailable, import failure), `3` the
`--input` file does not match the input size.
