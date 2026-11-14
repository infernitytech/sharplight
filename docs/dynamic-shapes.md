# Dynamic shapes, data-dependent sizes and control flow

This document describes how Sharplight represents shapes that are not known when a graph is built: symbolic dimension
expressions, data-dependent output sizes, runtime control flow (`If`, `Loop`, `Scan`) and ONNX sequences and optionals. The
guiding constraint is that
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

These features are built on top of these mechanisms instead of replacing them.

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

Control-flow ops (category `ControlFlow`) hold their subgraphs as ordinary `ComputeGraph`s in attributes. Values a subgraph reads
from the enclosing graph become explicit node inputs bound by name to subgraph inputs, so every subgraph is self-contained;
constants are embedded in it.

* `If` (`StandardOps.If`): branches `then_branch` / `else_branch`, inputs `[cond, captured...]` bound to the branch inputs
  named by `inputs`. Both branches produce the outputs named by `outputs`. Outputs whose shapes agree keep them; an output whose
  outermost dims differ gets a dynamic symbol (`output_symbols`) bounded by the larger branch. A branch that cannot be
  translated for the shapes known at import (silero-vad's 8 kHz branch with 16 kHz frames) is replaced by its error message
  (`then_error` / `else_error`); taking it at run time throws with the message.
* `Loop` (`StandardOps.Loop`): body `body`, inputs `[trip, cond, carried..., scan inputs..., captured...]` bound to the body
  inputs named by `body_inputs` (whose first entry is the iteration number, an i32 `[1]`). The body returns the continue
  condition, the next carried values and one slice per scan output (`body_outputs`). Carried values keep their type and shape
  across iterations. Iteration `t` reads slice `t` of the outermost axis of each scan input; scan outputs are the per-iteration
  slices stacked along a new outermost axis of extent `scan_extent`. `scan_input_reverse` / `scan_output_reverse` walk an
  axis backwards, `trip_symbol` limits the iterations by a symbol expression (a scan over the batch axis). When the number of
  iterations is only known at run time and there are scan outputs, `scan_extent` is a dynamic symbol (`count_symbol`) the node
  binds after it ran, bounded by the trip count when that is known and by `OnnxImportOptions.RuntimeLoopScanBound` (default
  4096) otherwise.

ONNX `Scan` is a `Loop` without condition and iteration inputs: its scan axes are moved to the front (`Permute`) before the
loop and the stacked outputs moved back after it, which works whenever the moved axis is not the symbolic one.

### Executor

The planner gives every control-flow node a segment of its own that runs on the host. `CompiledModel.Compile` turns it into
an `IfSegment` or a `LoopSegment`, which compile the subgraphs once as nested `CompiledModel`s with the same placement options
(so a body placed on Vulkan runs there, with the usual CPU fallback for ops Vulkan declines). Values cross the boundary
through host memory, exactly like values crossing ordinary segment boundaries; host segments read graph constants (a constant
trip count or condition) directly from the graph.

* `IfSegment.Execute` reads the condition, runs the chosen branch on the captured values and publishes the dynamic symbol of a
  disagreeing output.
* `LoopSegment.Execute` passes the captured values to the body once, then per iteration the iteration number, the condition,
  the carried values and one slice of each scan input; it runs the body, reads the condition and the carried values back and
  appends the scan slices. A run that needs more iterations than the scan outputs were compiled for fails with a clear error.

Nesting works recursively (an `If` inside a `Loop` body capturing values of both enclosing graphs).

### Translation strategy

The translator removes control flow statically whenever it can, so the runtime constructs are the fallback:

1. `If` with a compile-time condition: the taken branch is inlined.
2. `Loop` / `Scan` bodies that only compute constants are evaluated at import, iteration by iteration (TF-style `arange`
   loops, up to `OnnxImportOptions.MaxLoopIterations`).
3. Bodies that depend on runtime tensors are unrolled up to `OnnxImportOptions.MaxUnrolledLoop` (default 64) iterations when
   the trip count and the condition are known at import, so the body becomes ordinary nodes that fuse and run on the GPU.
4. Otherwise `If` becomes the runtime `If` node and `Loop` / `Scan` the runtime `Loop` node.

The choice between 3 and 4 is made before anything is emitted: a dataflow pass over the body marks every value that depends
on a runtime tensor (enclosing runtime values, carried values whose initial value or update is runtime, iterated to a fixed
point; shapes and sizes count as known). A runtime trip count, a runtime initial or continue condition, a symbolic scan length
or more than `MaxUnrolledLoop` iterations of a runtime body select the runtime loop.

Values that subgraphs read from enclosing graphs count as reads of the node that owns the subgraph, so the translator's
ordering, liveness and consumer analysis keep them (the expanded `AffineGrid` and `LinearAttention` functions define values
that only their `If` / `Scan` bodies read).

## 4. Sequences and optionals

ONNX sequences and optionals have no IR representation: they are resolved by the translator, so providers never see them.

* **Sequences whose structure is known at import** are lists of translated tensors (`TranslatedTensor.Sequence`). Every
  sequence op becomes ordinary IR on the elements: `SequenceConstruct`, `SequenceEmpty`, `SequenceInsert` / `SequenceErase`
  at a constant position, `SequenceAt` (a constant position picks the element; a runtime position stacks same-shaped elements
  and gathers one row), `SequenceLength` (a constant), `SplitToSequence` (slices), `ConcatFromSequence` (`Concat`, with
  `Unsqueeze` for `new_axis`) and `SequenceMap` (the body runs once per element at import, like an unrolled loop; elements may
  have different shapes). Unrolled and import-time loops carry such lists like any other value, which covers the common
  "append in a loop body, `ConcatFromSequence` after" pattern whenever the loop unrolls.
* **Sequences whose length is only known at run time** come from a runtime `Loop` that appends one tensor per iteration to a
  sequence that starts empty (`SequenceInsert` without position, the only use of the carried sequence in the body). The
  appended tensor becomes a scan output, and the sequence is that stacked tensor (`TranslatedTensor.Stacked`: elements along
  the outermost axis, length = the loop's count symbol). `SequenceAt` with a non-negative position, `SequenceLength` (a
  symbolic constant), `SequenceInsert` at the end and `ConcatFromSequence` along axis 0 accept it; anything that would move the
  symbolic length inward is rejected.
* **Optionals** are their value, or an empty marker (`TranslatedTensor.None`). `Optional`, `OptionalHasElement` (a constant)
  and `OptionalGetElement` are resolved at import; whether an optional holds a value is therefore always known at import.
* `If` with a compile-time condition, `Identity`, `Loop` and `Scan` pass sequences and optionals through. A runtime `If` whose
  branches return them, or subgraphs that read a runtime sequence of the enclosing graph, are rejected.

Graph inputs and outputs: a sequence input is one graph input per element, named `name[k]`
(`OnnxGraphAdapter.SequenceElementName`); its length is given by `OnnxImportOptions.SequenceInputLengths` and the element
shapes by `InputShapes` under the element names. An optional input holds a value unless it is listed in
`OnnxImportOptions.AbsentInputs`. A sequence output is one output per element when its length is known at import, or one
output under the sequence's own name with the elements stacked along the outermost axis when it is not; an empty optional
output has no graph output. The conformance harness feeds and compares `SequenceProto` / `OptionalProto` test data this way.

## 5. DynamicQuantizeLinear with a symbolic batch

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
* **Runtime control flow** costs one host round trip per branch or iteration for the values crossing the boundary (a runtime
  `Loop` copies its carried values and scan slices through host memory every iteration, and a GPU body is one submission
  per iteration). Subgraph weights are uploaded per subgraph, which doubles device weight memory for `If` nodes whose branches
  share weights (CPU weights are memory mapped and shared). Unrolling and constant folding avoid this for the common cases.
* **Sequences and optionals** cost nothing at run time: they are resolved into ordinary IR at import.

## Limitations

* Only the outermost dimension may be symbolic (static strides). Tensors whose dynamic dim moves inward, beyond the
  leading-unit-dim case above, are rejected with a clear error, as is reducing or transposing it away.
* Runtime `Range`, `Reshape`, `Expand` whose *values* (not just shapes) come from tensor data are supported only when
  the result can be expressed with symbols; truly value-dependent extents (GLiNER's `Range` up to the word count) are
  not, since they would need a value-range analysis to bound them.
* A runtime `Loop` needs loop-carried values of constant type and shape (a value that grows per iteration only works when the
  loop unrolls), scan outputs without symbolic dims, and sequences carried through it only in the append pattern above.
* A runtime `If` cannot return sequences or optionals, and the presence of an optional is fixed at import.
* `SequenceInsert` / `SequenceErase` at a runtime position, and `SequenceMap` over a runtime-length sequence, are unsupported.
* A gather along a data-dependent axis that is not the outermost (Grounding DINO's `NonZero` results indexing an inner
  axis), `Unique` with an `axis` and `Compress` on an inner axis are unsupported.
* `Concat` along a symbolic outer axis and `RoiAlign` run on the CPU; on Vulkan this costs a host round trip per node.
* Subgraph weights are not shared between nested compiled models on accelerators.
