# Sharplight.Onnx

ONNX importer + generic graph adapter. `OnnxImporter` (format `onnx`) memory-maps the model, decodes it with a hand-rolled
protobuf reader (`Protobuf.cs`, `OnnxParser.cs`) and exposes initializers as zero-copy weights (`raw_data` and
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
| Concat Split Slice Pad Resize Upsample | yes | Slice step must be 1; Pad modes constant/reflect/edge; Resize nearest/linear, 4-D, constant scales/sizes (non-integer *scales* are converted to sizes) |
| Gather | yes | constant indices -> Slice/Concat views; dynamic indices -> `GatherAxis` |
| GatherElements GatherND ScatterElements ScatterND Scatter CumSum Tile Expand Trilu OneHot GridSample | yes | Expand to the dynamic batch uses a zero-broadcast trick |
| DepthToSpace SpaceToDepth | yes | Reshape + Permute |
| Cast CastLike | yes | to float: identity; to int: i32; to bool: `!= 0` mask |
| Shape Size Range ConstantOfShape Constant NonZero EyeLike | fold | dynamic-shape results are unsupported |
| QuantizeLinear DequantizeLinear | yes | fold on constants; runtime = Div/Round/Add/Clip (values carried as f32) |
| MatMulInteger ConvInteger QLinearMatMul QLinearConv | yes | weights dequantized at import; activations carried as f32 ints |
| If | limit | constant condition only (taken branch inlined) |
| LSTM GRU RNN | limit | unrolled over a static sequence length; forward/reverse/bidirectional, `layout` 0/1, default activations, no peepholes/clip/sequence_lens; initial states must be constants |
| DynamicQuantizeLinear, Loop, Scan, NonMaxSuppression, Unique, Compress, RoiAlign, Det, ... | no | |

### com.microsoft contrib ops

Gelu, BiasGelu, FastGelu, QuickGelu, FusedConv, FusedMatMul, SkipLayerNormalization, SkipSimplifiedLayerNormalization,
SimplifiedLayerNormalization (RmsNorm), Attention (no mask/past; `unidirectional` via a constant causal mask), MultiHeadAttention (no mask/past),
BiasAdd, MatMulNBits (4/8-bit dequantized at import). **RotaryEmbedding is not supported** (needs a Rope op taking cos/sin caches).

## Tests

`dotnet run --project tests/Sharplight.Onnx.Tests`: protobuf/parser unit tests, translation tests, and numeric comparisons against onnxruntime
(`.work/onnx_ref/make_synthetic.py`, `make_synthetic2.py`, `make_model_refs.py` generate the references into `~/models/onnx/ref`).
