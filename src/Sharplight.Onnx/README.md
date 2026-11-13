# Sharplight.Onnx

ONNX importer + generic graph adapter. `OnnxImporter` (format `onnx`) memory-maps the model, decodes it with a hand-rolled
protobuf reader (`ProtoReader.cs`, `OnnxParser.cs`) and exposes initializers as zero-copy weights (`raw_data` and
`.onnx.data` external data). `OnnxGraphAdapter` (architecture `onnx`) translates the graph op-by-op into one Sharplight
`ComputeGraph` named `main`.

```csharp
OnnxImporter.Register();                                   // importer + adapter
var src = OnnxImporter.Open("model.onnx", new OnnxImportOptions { InputShapes = { ["pixel_values"] = [-1, 3, 224, 224] } });
var def = ModelAdapterRegistry.Build(new ModelSourceSet { Primary = src });
// compile with CompileOptions.SymbolBounds["N"] = max batch
```

## Conventions

* The outermost dim of every input is the batch: `dim_param` / unknown (0) becomes the symbol **`N`**; a fixed outer dim stays fixed
  (`ForceSymbolicBatch` overrides). All other dims must be static (`OnnxImportOptions.InputShapes` fixes dynamic H/W, seq length, ...).
* Floats are f32 values; int32/int64 graph inputs become i32 inputs; ONNX bool = f32 0/1 mask (see Core `StandardOps.General`).
* **Constant folding**: initializer-only subgraphs, `Constant`, `Shape` (batch dim = a private sentinel), `Gather`, `Unsqueeze`, `Squeeze`,
  `Concat`, `Slice`, `Cast`, `ConstantOfShape`, `Range`, `Equal/Where/...`, arithmetic on int64 shape tensors, `Transpose`/`Reshape` of weights
  (lazy, byte-level), `QuantizeLinear/DequantizeLinear`, `NonZero`, `Tile`, `ArgMax`, ... run on the host at import time.
  Arithmetic on the symbolic batch size is rejected with a clear error; `Reshape` targets mentioning it become Sharplight's "keep outer dim" (0).
* `BatchNormalization` directly after a `Conv` with no other consumers is folded into derived conv weight/bias (`WeightInfo.Derived`).
* `Gemm`/`MatMul` with a constant B become `MatMul` with a zero-copy weight (`transB=1`) or a derived transposed weight.
* Placement groups: `onnx.000`, `onnx.001`, ... a new group every `GroupSize` (8) heavy ops (Conv/MatMul/Gemm/BatchMatMul).
* Unsupported operator types are reported together in one `NotSupportedException`.

## Operator coverage (ai.onnx, opsets 7-21)

Status: **yes** = runtime op mapped to IR; **fold** = only when all inputs are compile-time constants; **limit** = see note; **no** = unsupported.

| Op | Status | Notes |
|---|---|---|
| Abs Neg Exp Log Sqrt Reciprocal Erf Sin Cos Tan Asin Acos Atan Sinh Cosh Asinh Acosh Atanh Floor Ceil Round Sign Softsign Softplus Sigmoid Tanh Relu IsNaN IsInf Not | yes | also fold |
| HardSigmoid HardSwish Elu Celu Selu LeakyRelu ThresholdedRelu Mish Swish/Silu Gelu | yes | Gelu `approximate=tanh` supported |
| Clip | yes | min/max attr or const inputs |
| PRelu | yes | |
| Add Sub Mul Div Pow Max Min Sum Mean | yes | numpy broadcasting, variadic |
| Mod | yes | `fmod` attr |
| Equal Less Greater LessOrEqual GreaterOrEqual And Or Xor Where | yes | masks |
| Conv | yes | 1-D (via H=1) and 2-D, groups, dilation, auto_pad, BN folding; 3-D no |
| ConvTranspose | yes | 1-D/2-D, output_padding, output_shape |
| MaxPool AveragePool LpPool (p=1,2) GlobalAveragePool GlobalMaxPool | yes | ceil_mode, dilations, count_include_pad; MaxPool indices output no |
| BatchNormalization | yes | inference mode only |
| InstanceNormalization GroupNormalization LayerNormalization LpNormalization LRN MeanVarianceNormalization RMSNormalization | yes | LayerNorm over trailing axes via reshape; stash outputs no |
| Softmax LogSoftmax Hardmax | yes | any axis via Permute, opset <13 coercion semantics |
| ReduceMean/Sum/Max/Min/Prod/L1/L2/SumSquare/LogSum/LogSumExp | yes | axes attr or input; reducing the batch axis no |
| ArgMax ArgMin TopK | yes | outputs i32 |
| Gemm MatMul | yes | constant B or BatchMatMul; `transA` on batched A no |
| Einsum | limit | two operands, no ellipsis/diagonals; batch dim must stay leading |
| Flatten Reshape Squeeze Unsqueeze Transpose Identity Dropout | yes | reshapes that merge/move the batch dim: limit (error) |
| Concat Split Slice Pad Resize Upsample | yes | Slice: any non-zero step (positive = Reshape trick, negative = constant-index gather); Pad modes constant/reflect/edge; Resize nearest/linear/**cubic**, 4-D, constant scales/sizes (cubic = two constant interpolation-matrix MatMuls; non-integer *scales* are converted to sizes) |
| Gather | yes | constant indices -> Slice/Concat views; dynamic indices -> `GatherAxis` |
| GatherElements GatherND ScatterElements ScatterND Scatter CumSum Tile Expand Trilu OneHot GridSample | yes | Expand to the dynamic batch uses a zero-broadcast trick |
| DepthToSpace SpaceToDepth | yes | Reshape + Permute |
| Cast CastLike | yes | to float: identity; to int: i32; to bool: `!= 0` mask |
| Shape Size Range ConstantOfShape Constant NonZero EyeLike | fold | dynamic-shape results are unsupported |
| QuantizeLinear DequantizeLinear | yes | fold on constants; runtime = Div/Round/Add/Clip (values carried as f32) |
| DynamicQuantizeLinear | yes | whole-tensor min/max, so the batch must be fixed (`InputShapes`); scale / zero point are runtime values |
| MatMulInteger ConvInteger QLinearMatMul QLinearConv | yes | weights dequantized at import (per-tensor / per-column / per-channel zero points); activations carried as f32 ints; activation zero point may be a runtime value |
| If | limit | constant condition only (taken branch inlined) |
| LSTM GRU RNN | limit | unrolled over a static sequence length; forward/reverse/bidirectional, `layout` 0/1, default activations, no peepholes/clip/sequence_lens; initial states may be runtime inputs |
| DynamicQuantizeLinear, Loop, Scan, NonMaxSuppression, Unique, Compress, Det, ... | no | |

### com.microsoft contrib ops

Gelu, BiasGelu, FastGelu, QuickGelu, FusedConv, FusedMatMul, SkipLayerNormalization, SkipSimplifiedLayerNormalization,
SimplifiedLayerNormalization (RmsNorm), Attention (no mask/past; `unidirectional` via a constant causal mask), MultiHeadAttention (no mask/past),
BiasAdd, MatMulNBits (4/8-bit dequantized at import), QLinearAdd/Mul/Sigmoid/LeakyRelu/Softmax/AveragePool/GlobalAveragePool/Concat (dequantize, op, requantize). **RotaryEmbedding is not supported** (needs a Rope op taking cos/sin caches).

## Import options worth knowing

* `InputShapes[name]` fixes dynamic dims; giving the concrete batch (instead of `-1`) makes the batch static, which lifts the
  restrictions around the symbolic batch (reshapes that merge it, transposes that move it, arithmetic on it, global reductions such as
  `DynamicQuantizeLinear`, scalar-to-batch `Expand`). Models traced with explicit window/batch reshapes (Swin, GPT-2, wav2vec2, PaddleOCR ...) need this.
* `ConstantInputs[name] = values` pins a graph input (e.g. `sr`, `use_cache_branch`, an `orig_im_size` that fixes output shapes) to a
  compile-time constant so `If` branches and shape computations fold.
* Scalar (rank-0) graph inputs are carried as one-element vectors; `Unsqueeze`/`Squeeze` honour the logical rank.

## Source layout

* `OnnxImporter`, `OnnxGraphAdapter`, `OnnxImportOptions`: the public API; everything else is internal.
* `OnnxParser`, `ProtoReader`, `Mapped*` and `*WeightSource`: protobuf decoding and zero-copy weight sources.
* `OnnxConst`, `OnnxFold`: compile-time constants and their host-side evaluation.
* `OnnxTranslator.*.cs`: one partial class split by operator category (`Dispatch` routes nodes to `Elementwise`, `Spatial`, `Resample`,
  `Reduction`, `Linear`, `Shape`, `Indexing`, `Normalization`, `Quant`, `Recurrent`, `ControlFlow` and `Contrib`; `Helpers` holds shared
  utilities). Each operator is translated by a method named after it.

## Tests

`dotnet run --project tests/Sharplight.Onnx.Tests`: protobuf/parser unit tests, translation tests, and numeric comparisons against onnxruntime
(`.work/onnx_ref/make_synthetic.py`, `make_synthetic2.py`, `make_model_refs.py` generate the references into `~/models/onnx/ref`).
`ZooTests` covers ~45 architecturally diverse real models (detection, super-resolution, SAM, diffusion, speech, LLM decoders with KV cache,
OCR, optical flow, quantized ...): `tests/Sharplight.Onnx.Tests/scripts/make_zoo_refs.py` lists the models (downloaded to `~/models/onnx/<name>/`) and writes
references to `~/models/onnx/ref/zoo`; `make_zoo_synth.py` builds small synthetic cases for the ops they needed.
