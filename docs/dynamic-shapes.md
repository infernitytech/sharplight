# Dynamic shapes, data-dependent sizes and control flow

This document describes how Sharplight represents shapes that are not known when a graph is built: symbolic dimension
expressions, data-dependent output sizes and runtime control flow (`If`, `Loop`, `Scan`). The guiding constraint is that
the static fast path does not change: a graph whose shapes are static (or whose only symbol is a caller-bound batch or
token count) compiles to exactly the same kernels and runs through exactly the same code as before.

## Starting point

* A `Dim` is either fixed or a named symbol (`N`, `T`, `ctx`, ...). Only the **outermost** dimension of a `Shape` may be
  symbolic, so every stride is static and kernels can be specialized at compile time.
* Symbols are **bound per run** (`RunParameters.Symbols`) and **bounded at compile time** (`CompileOptions.Bound`):
  buffers are sized for the bound, kernels read the actual value at run time (CPU: when an op runs; Vulkan: when a
  command buffer is recorded, one recording per distinct binding).
* `CompiledModel` runs segments (contiguous runs of nodes on one device) in order and moves boundary values through
  host memory. Nodes a device cannot run go to the fallback device (normally the CPU).

All three new features are built on top of these mechanisms instead of replacing them.

## 1. Symbolic dimension expressions

### Representation

`Dim.Symbol` may now hold a **canonical expression** over atomic symbols instead of a plain name:

```
N            atomic symbol
12*N         affine
N+1
4*N*T        product of symbols
ceil(N/32)   floor(e/c), ceil(e/c), min(a,b), max(a,b) (opaque sub-terms)
```

`SymbolicExpression` (Core, `Sharplight.Tensors`) parses, simplifies and prints these expressions. Expressions are kept in
a normal form: a sum of monomials with integer coefficients (`12*N + 1`), where a monomial is a product of atomic symbols
and opaque sub-terms (`floor`, `ceil`, `min`, `max`). Simplification is exact: `(12*N)/12 = N`, `N*4/2 = 2*N`,
`(N+31)/32` stays `floor((N+31)/32)`. Because the printed form is canonical, the expression *is* its name: two dims are
equal when their strings are equal, and every place that compares, hashes or dictionary-keys symbols keeps working.

The "outermost only" rule is unchanged. An expression only ever appears on the outermost dimension, so strides stay
static. `Dim * long`, `Dim + long`, `Dim.FloorDiv`, `Dim.CeilDiv`, `Dim.Min` and `Dim.Max` build expression dims.

### Binding and bounds

* **Run time**: callers still bind only atomic symbols. `Dim.Resolve` and `Shape.Resolve/Elements` evaluate expressions.
  Provider code that looks symbols up by name goes through one helper (`SymbolicExpression.Evaluate(name, bindings)`). It
  returns the bound value for an atomic name and evaluates an expression name; parsed expressions are cached by text, so
  the lookup costs one dictionary hit.
* **Compile time**: `CompileOptions.Bound(name)` evaluates an expression on the bounds of its atoms. Every operator in
  the language is monotone non-decreasing in non-negative arguments, except terms with negative coefficients. Those are
  evaluated with their atoms at 0, so the result is a safe upper bound. Linear expressions get exact bounds, which keeps
  views consistent: a buffer of bound `[12·B, 16]` is the same buffer as `[B, 12, 16]`.

### IR and translator

* `Reshape` accepts an `outer_symbol` attribute that is now any expression; shape inference checks the element count
  symbolically, so `[N, 12, 16, 64] → [12*N, 16, 64] → [N, 12, 16, 64]` round-trips without fixing the batch.
* The ONNX translator keeps shape tensors (the results of `Shape`, `Gather`, `Concat`, `Mul`, `Div`, ... on shapes) as
  compile-time constants whose elements may be expressions (`SymbolicInts` encodes an expression as a reserved int64
  code, and as a NaN payload in float shape tensors, which TF exports produce; ops that would treat such a code as a
  number refuse to fold). Arithmetic, comparisons that can be decided, `Gather`, `Slice`, `Concat`,
  `Squeeze`/`Unsqueeze`, `Cast`, `Min`/`Max`, `Where` with constant conditions, `ReduceProd` and `Equal` on shape
  tensors propagate expressions. The consumers of shape tensors then produce IR with expression dims:
  * `Reshape` targets mentioning the batch (`[-1, 16, 64]`, `[N*12, ...]`) become `Reshape` with `outer_symbol`.
  * `Range(0, N, 1)`, `ConstantOfShape([N, 1, 16])` and `Expand(x, [N, ...])` become the new `Fill` op (below)
    with an expression outer dim.
  * `Size` of a batched tensor is a symbolic scalar.
* Shape inference no longer loses information when HF exports compute `batch * heads`, `seq + past` or `ceil(len/chunk)`.

## 2. Data-dependent output sizes

### Representation

An op whose output size is only known after it runs declares a **dynamic symbol**: a fresh atomic symbol (for example
`nonzero_17`) that is the outermost dimension of its output. The graph records an **upper bound** for each dynamic symbol
in `ComputeGraph.DynamicSymbols` (name → bound `Dim`). The bound may itself be an expression of other symbols (for example
`NonZero` over `[N, 100]` has the bound `100*N`). `GraphBuilder.DynamicSymbol(hint, bound)` creates one, and the op
receives the name as an attribute (`count_symbol`).

Dynamic symbols behave exactly like caller-bound symbols for everything downstream:

* **Compile time**: `CompiledModel.Compile` evaluates each dynamic bound in graph order and adds it to the
  `CompileOptions.SymbolBounds` it passes to the providers, so buffers are sized for the bound and kernels are
  specialized as for any other symbol. Expressions over dynamic symbols (`min(nms_3,200)`) work as well.
* **Run time**: the op that owns the symbol writes the actual value into `RunParameters.Symbols` when it runs. Ops later
  in the same segment, later segments, the host transfer between segments and the caller all read it from there. After
  `Run`, `CompiledModel.OutputShape(name, parameters)` (or `Shape.Resolve(parameters.Symbols)`) gives the actual output
  shape.

### Segment and executor boundaries

The executor needs no new mechanism. A data-dependent op is a node like any other. Where its value is consumed:

* **On the CPU** the count is known as soon as the op returns, so a CPU segment can contain the producer and its
  consumers.
* **On Vulkan** the providers have no data-dependent kernels. `NonZero`, `NonMaxSuppression`, `Unique`, `TopK` with a
  dynamic `k` and friends are not supported there, so the placement planner puts them on the fallback device (CPU). The
  segment boundary this creates is also the point where the host learns the count. The next Vulkan segment then sees
  the bound symbol when it records its command buffer. Vulkan caches one recording per distinct binding of its symbols;
  the cache is capped (LRU), so a stream of different counts re-records but does not grow without limit.

This makes "sync at the data-dependent op" the rule, and it is the only correct choice without indirect dispatch. The
cost is one host round trip per data-dependent op on a GPU. In practice these ops sit in post-processing (NMS, top-k
selection) or at the front of a model (masks → indices). See "Performance" below.

### Ops

| IR op | Output | Bound | CPU | Vulkan |
|---|---|---|---|---|
| `NonZero` x | i32 `[nnz, rank]` (transposed ONNX layout) | elements of x | yes | fallback |
| `NonMaxSuppression` boxes, scores | i32 `[selected, 3]` | batches·classes·min(boxes, max_per_class) | yes | fallback |
| `Unique` x (flattened, sorted or first-occurrence order) | values/first indices/counts `[u]`, inverse `[n]` | elements of x | yes | fallback |
| `TopK` with `k_symbol` | `[k, ...]` along the symbolic axis | `k` expression | yes | fallback |
| `Fill` (start, delta, outer expr, inner dims) | `start + i·delta` along axis 0 | — (not data dependent) | yes | yes |
| `RoiAlign` x, rois `[R, 4]`, batch indices `[R]` | f32 `[R, C, oh, ow]` (R usually data dependent) | R | yes | fallback |

Consumers of a data-dependent (or any symbolic) outer axis:

* `Concat` along the outer axis has the sum expression as extent (`nms_1+nms_2+...`). The CPU kernel places each input at
  its run-time offset; Vulkan declines the node (its copies use compile-time offsets), so it runs on the fallback device.
* `Slice` of a prefix of the outer axis (`x[:1000]`, the "top-k proposals" pattern) is a view with extent `min(n,1000)`
  on both providers.
* `NonMaxSuppression` also accepts one batch and class with a symbolic box count: ONNX `[1, S, 4]` / `[1, 1, S]` are
  boxes `[S, 4]` and scores `[S]` behind leading unit dims (below).
* `NonZero(ConstantOfShape([n], c ≠ 0))`, the exporters' spelling of `arange(n)`, becomes `Fill` (no count to discover).

`Compress` (axis 0 or flattened) is translated as `NonZero` + `GatherAxis`. `NonZero` keeps the ONNX `[rank, nnz]` view
in the translator: `Transpose`, `Squeeze`, `Gather`/`Slice` of one coordinate row and `GatherND` read the IR `[nnz, rank]`
value directly.

Detection post-processing often adds a leading unit dimension (`[1, nbox, 4]`). Because a symbolic dim must be
outermost, the translator keeps such tensors as `[nbox, 4]` and records the leading unit dims (`TranslatedTensor`).
`Squeeze`/`Unsqueeze`/`Reshape` and graph outputs handle them; graph outputs keep the ONNX element layout, which is
identical.

## 3. Runtime control flow

### IR

The IR `If` op (`StandardOps.If`, category `ControlFlow`) holds its branches as ordinary `ComputeGraph`s in the
attributes `then_branch` / `else_branch`. Values the branches read from the enclosing graph become explicit node inputs
(`[cond, captured...]`) bound by name (`inputs`) to the branch graph inputs, so a branch is a self-contained graph;
constants are embedded in the branch. Both branches produce the outputs named by `outputs`. Outputs whose shapes agree keep
them; an output whose outermost dims differ gets a dynamic symbol (`output_symbols`) bounded by the larger branch.

A branch can be impossible to translate for the shapes known at import, typically because it is written for other inputs
(silero-vad's 8 kHz branch with 16 kHz frames). Such a branch is replaced by its error message (`then_error` /
`else_error`); the If imports, and taking that branch at run time throws with the message.

### Executor

The planner gives every control-flow node a segment of its own that runs on the host. `CompiledModel.Compile` turns it
into an `IfSegment`, which compiles each branch as a nested `CompiledModel` with the same placement options. Its
`Execute` reads the condition from host memory, runs the chosen branch on the captured values and keeps the outputs in
host memory, exactly like values crossing ordinary segment boundaries; the dynamic symbol of a disagreeing output is
written into `RunParameters.Symbols`. Nested Ifs work recursively.

### Translation strategy

The translator removes control flow statically whenever it can, so the runtime construct is the last resort:

1. `If` with a compile-time condition: the taken branch is inlined.
2. `Loop` / `Scan` bodies run through the translator at import, iteration by iteration. Bodies that only compute constants
   are evaluated (TF-style `arange` loops, up to `OnnxImportOptions.MaxLoopIterations`).
3. Bodies that depend on runtime tensors are unrolled up to `OnnxImportOptions.MaxUnrolledLoop` (default 64) iterations,
   so the body becomes ordinary nodes that fuse and run on the GPU. The trip count and the loop condition must be known
   at import.
4. Otherwise `If` becomes the runtime `If` node. There is no runtime `Loop` yet (see Limitations).

## 4. DynamicQuantizeLinear with a symbolic batch

`DynamicQuantizeLinear` needs min/max over the whole tensor, including the symbolic batch axis. `Reduce` now accepts the
symbolic outermost axis (output extent 1). On the CPU the reduction loops over the bound value. The Vulkan `reduce`
kernel takes the reduced element count as a push constant (`nRed`) when a reduced axis is symbolic; the static case
keeps the compile-time constant.

## Performance implications

* **Static graphs are untouched.** No new nodes, attributes, segments or symbol lookups appear unless a model uses these
  features. Plain symbol names do not take the expression path (lookups are one dictionary hit as before).
* **Expression dims** cost nothing at run time beyond evaluating a small tree when an op or a recording reads its extent.
  They remove the need to fix the batch, so one compiled model serves every batch size up to the bound.
* **Data-dependent ops** force a host synchronization on GPU devices (the segment boundary) and, on Vulkan, a command
  buffer recording per new count value (amortized by the recording cache when counts repeat). Buffers are sized for the
  upper bound, so memory follows the bound, not the actual count.
* **Runtime control flow** costs one host round trip per branch or iteration for the values crossing the boundary.
  Subgraph weights are uploaded per subgraph, which doubles device weight memory for `If` nodes whose branches share
  weights (CPU weights are memory mapped and shared). Unrolling and constant folding avoid this for the common cases.

## Limitations

* Only the outermost dimension may be symbolic (static strides). Tensors whose dynamic dim moves inward, beyond the
  leading-unit-dim case above, are rejected with a clear error, as is reducing or transposing it away.
* Runtime `Reshape` and `Expand` whose *values* (not just shapes) come from tensor data are supported only when the
  result can be expressed with symbols. A runtime `Range` limit (with constant start and delta) is supported when a
  value-range analysis proves an upper bound for it (GLiNER's `Range` up to the word count): the analysis walks the
  ONNX graph back from the limit through constants, shapes, comparisons (0/1), casts and value-preserving ops,
  `ReduceSum` over known extents, `ReduceMax` / `Max` / `Add` / `Mul`, and graph inputs with a declared bound
  (`OnnxImportOptions.InputValueBounds`, values in `[0, bound]`). The range is then the prefix of the constant
  sequence `start + i · delta` below that bound selected with `NonZero`, so its length is a dynamic symbol like
  `NonZero`'s. A declared input bound is a contract: larger values at run time truncate the range. Runtime start or
  delta, and limits bounded only by a symbolic extent, are not supported.
* `Loop` whose trip count or condition depends on runtime tensors, or that needs more than `MaxUnrolledLoop` iterations
  of a runtime body, is rejected: there is no runtime loop construct.
* A gather along a data-dependent axis that is not the outermost (Grounding DINO's `NonZero` results indexing an inner
  axis), `Unique` with an `axis` and `Compress` on an inner axis are unsupported.
* `Concat` along a symbolic outer axis and `RoiAlign` run on the CPU; on Vulkan this costs a host round trip per node.
* Packages (`ModelPackage.Save`) reject graphs with dynamic symbols or runtime `If` nodes.
* Subgraph weights are not shared between nested compiled models on accelerators.
