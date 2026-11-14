using Sharplight.Tensors;

namespace Sharplight.Graph;

/// <summary>
/// The built-in operator set. The set is intentionally small and architecture-neutral: primitives
/// (elementwise, views, matmul, reductions) plus a handful of composite ops that backends are
/// expected to implement efficiently (attention, rope, recurrent rules). New architectures can add
/// ops through <see cref="OpRegistry.Register"/> without touching the core.
/// </summary>
/// <remarks>
/// Conventions: shapes are outermost-first. The token/batch axis is the (possibly symbolic) outermost
/// axis. Reductions and normalizations operate on the innermost axis.
/// </remarks>
public static partial class StandardOps
{
    // ---- elementwise ----------------------------------------------------------------------

    /// <summary>
    /// Names of the registered one-operand elementwise ops. Each takes x (any shape) and returns f32 of the same shape (i32 for an i32
    /// input of <see cref="IntegerPreservingUnaryNames"/>);
    /// the attributes some of them read (alpha, beta, gamma) are listed next to the name. Mask-valued ops
    /// (<c>Not</c>, <c>IsNaN</c>, <c>IsInf</c>) produce and consume f32 0/1 masks.
    /// </summary>
    public static readonly string[] UnaryNames =
        ["Neg", "Exp", "Log", "Sqrt", "Rsqrt", "Sigmoid", "Silu", "Gelu", "GeluTanh", "Tanh", "Softplus", "Relu", "Abs", "Square", "Recip",
         "Erf", "HardSwish", "HardSigmoid" /* attrs alpha (0.2), beta (0.5) */,
         "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh",
         "Floor", "Ceil", "Round" /* half to even */, "Sign", "Softsign", "Mish",
         "Elu" /* alpha (1) */, "Celu" /* alpha (1) */, "Selu" /* alpha (1.67326), gamma (1.0507) */,
         "LeakyRelu" /* alpha (0.01) */, "ThresholdedRelu" /* alpha (1) */,
         // Mask-producing / mask-consuming (masks are f32 0/1; non-zero means true):
         "Not", "IsNaN", "IsInf"];
    /// <summary>
    /// Names of the registered two-operand elementwise ops. Each takes a and b with numpy-style broadcasting and returns
    /// f32 of the broadcast shape (i32 for two i32 inputs of <see cref="IntegerPreservingBinaryNames"/>); comparison and logic ops
    /// return f32 0/1 masks.
    /// </summary>
    public static readonly string[] BinaryNames =
        ["Add", "Sub", "Mul", "Div", "Max", "Min", "Pow",
         "Mod" /* attr fmod (bool): C fmod vs. Python-style sign of divisor */,
         "PRelu" /* x, slope */,
         // Comparisons and logic produce f32 masks (1 = true, 0 = false):
         "Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual", "And", "Or", "Xor"];

    /// <summary>y = x * scale + bias. Input: x (any shape). Attributes: scale (float), bias (float). Output: f32, shape of x.</summary>
    public static readonly OpSchema Scale = new("Scale", OpCategory.Elementwise, 1, 1, Same, "y = x*scale + bias");

    /// <summary>Element type conversion. Input: x. Attribute: type (string, a <see cref="DataType"/> name). Output: the given type, shape of x.</summary>
    public static readonly OpSchema Cast = new("Cast", OpCategory.Elementwise, 1, 1,
        (i, a) => [new TensorType(DataType.Get(a.Get<string>("type")), i[0].Shape)]);

    // ---- views ----------------------------------------------------------------------------
    /// <summary>
    /// Reshape (row-major). Attribute: shape (long[]; -1 = infer, 0 = keep symbolic outer dim). Optional attribute
    /// <c>outer_symbol</c> (string): the first entry of <c>shape</c> is ignored and the outer dimension is that symbol or expression.
    /// An expression over the input's symbols must preserve the element count (<c>[N, 12, 16] → [12*N, 16]</c>); a symbol unrelated to
    /// the input's (e.g. N patches → M = N/4 merged tokens) must be bound consistently by the caller.
    /// </summary>
    public static readonly OpSchema Reshape = new("Reshape", OpCategory.View, 1, 1, InferReshape);

    /// <summary>
    /// Zero-copy slice along one axis. Input: x. Attributes: axis (int, negative counts from the end), start (long), length (long).
    /// Output: x with <c>length</c> entries on that axis. On the symbolic outer axis only a prefix (start 0) is allowed; its extent is
    /// <c>min(n, length)</c>.
    /// </summary>
    public static readonly OpSchema Slice = new("Slice", OpCategory.View, 1, 1, (i, a) =>
    {
        int axis = a.Get<int>("axis");
        if (axis < 0)
        {
            axis += i[0].Rank;
        }

        long len = a.Get<long>("length");
        if (axis == 0 && i[0].Shape[0].IsSymbolic)
        {
            // A prefix of the symbolic axis: the first min(n, length) rows.
            return a.Get<long>("start") == 0
                ? [new TensorType(i[0].Type, i[0].Shape.WithDim(0, Dim.Min(i[0].Shape[0], len)))]
                : throw new NotSupportedException("Slicing the symbolic axis at an offset is not supported.");
        }

        return [new TensorType(i[0].Type, i[0].Shape.WithDim(axis, len))];
    });

    /// <summary>Axis permutation as a lazy view. Input: x. Attribute: perm (int[], output axis i is input axis perm[i]). Output: permuted shape.</summary>
    public static readonly OpSchema Permute = new("Permute", OpCategory.View, 1, 1, (i, a) =>
    {
        var perm = a.Get<int[]>("perm");
        return [new TensorType(i[0].Type, new Shape(perm.Select(p => i[0].Shape[p])))];
    });

    /// <summary>Materializes a (possibly strided) view into a dense row-major tensor. Input: x. Output: same type and shape.</summary>
    public static readonly OpSchema Contiguous = new("Contiguous", OpCategory.Data, 1, 1, Same);

    /// <summary>
    /// Concatenates 2 to 64 operands along an axis. Inputs: tensors with equal type and equal shape except on that axis.
    /// Attribute: axis (int, negative counts from the end). Output: the summed extent on that axis; along the outer axis the operands may
    /// be symbolic and the extent is the sum expression (operands are placed at their run-time offsets).
    /// </summary>
    public static readonly OpSchema Concat = new("Concat", OpCategory.Data, 2, 64, (i, a) =>
    {
        int axis = a.Get<int>("axis");
        if (axis < 0)
        {
            axis += i[0].Rank;
        }

        // Along the outer axis the inputs may be symbolic: the extent is the sum expression.
        var sum = axis == 0 ? i.Aggregate(Dim.Of(0), (s, v) => s + v.Shape[0]) : Dim.Of(i.Sum(v => v.Shape[axis].Fixed));
        return [new TensorType(i[0].Type, i[0].Shape.WithDim(axis, sum))];
    });

    /// <summary>Row gather: table [V, ...], indices i32 [N] → [N, ...] (dequantizes to f32).</summary>
    public static readonly OpSchema Gather = new("Gather", OpCategory.Data, 2, 2, (i, a) =>
    {
        var t = i[0].Shape;
        var dims = new List<Dim> { i[1].Shape[0] };
        for (int d = 1; d < t.Rank; d++)
        {
            dims.Add(t[d]);
        }
        // Rows are decoded into the activation type (f32 unless "out_type" says otherwise), whatever the table format.
        return [new TensorType(a.Get("out_type", DataType.F32), new Shape(dims))];
    });

    // ---- linear algebra -------------------------------------------------------------------
    /// <summary>
    /// MatMul attribute (bool, default false): x takes part in f32, so backends do not quantize it to int8 for integer dot
    /// products unless <see cref="Runtime.CompileOptions.AllowReducedPrecision"/> is set. Importers set it where the source
    /// format promises f32 activation math (e.g. ONNX MatMulNBits without accuracy_level 4).
    /// </summary>
    public const string F32ActivationsAttribute = "f32_activations";

    /// <summary>
    /// y = x · Wᵀ. x: [..., K] (f32), W: [N, K] (any float or quantized format) → [..., N].
    /// This is the "Linear" layout used by GGUF/PyTorch weights. Attribute: <see cref="F32ActivationsAttribute"/>.
    /// </summary>
    public static readonly OpSchema MatMul = new("MatMul", OpCategory.MatMul, 2, 2, (i, a) =>
    {
        var x = i[0].Shape;
        var w = i[1].Shape;
        if (w.Rank != 2)
        {
            throw new ArgumentException("MatMul weight must be rank 2 [N, K].");
        }

        if (x[-1] != w[1])
        {
            throw new ArgumentException($"MatMul K mismatch: x{x} w{w}.");
        }

        return [new TensorType(DataType.F32, x.WithLast(w[0]))];
    });

    // ---- reductions / normalization -------------------------------------------------------
    /// <summary>RMS normalization over the last axis; optional weight [D]. Attribute: eps.</summary>
    public static readonly OpSchema RmsNorm = new("RmsNorm", OpCategory.Reduction, 1, 2, SameF32);
    /// <summary>x / sqrt(sum(x²) + eps) over the last axis. Attribute: eps.</summary>
    public static readonly OpSchema L2Norm = new("L2Norm", OpCategory.Reduction, 1, 1, SameF32);
    /// <summary>Layer normalization over the last axis; optional weight and bias [D]. Attribute: eps.</summary>
    public static readonly OpSchema LayerNorm = new("LayerNorm", OpCategory.Reduction, 1, 3, SameF32);
    /// <summary>Softmax over the last axis. Attributes: scale (applied before exp, default 1), log (bool: LogSoftmax).</summary>
    public static readonly OpSchema Softmax = new("Softmax", OpCategory.Reduction, 1, 1, SameF32);
    /// <summary>Sum over the last axis. Input: x. Output: f32, last dimension 1.</summary>
    public static readonly OpSchema ReduceSum = new("ReduceSum", OpCategory.Reduction, 1, 1, (i, a) => [new(DataType.F32, i[0].Shape.WithLast(1))]);
    /// <summary>Maximum over the last axis. Input: x. Output: f32, last dimension 1.</summary>
    public static readonly OpSchema ReduceMax = new("ReduceMax", OpCategory.Reduction, 1, 1, (i, a) => [new(DataType.F32, i[0].Shape.WithLast(1))]);
    /// <summary>Index of the maximum along the last axis → i32 [..., 1].</summary>
    public static readonly OpSchema ArgMax = new("ArgMax", OpCategory.Reduction, 1, 1, (i, a) => [new(DataType.I32, i[0].Shape.WithLast(1))]);

    // ---- composite ------------------------------------------------------------------------
    /// <summary>
    /// Rotary position embedding on x [T, H, D] with positions i32 [T, A] (A position axes; A=1 for
    /// plain RoPE, &gt;1 for multi-axis / M-RoPE). Attributes:
    /// <c>dims</c> (rotated dims, ≤ D), <c>base</c>, <c>style</c> ("neox" pairs (i, i+dims/2), "gptj" pairs (2i, 2i+1)),
    /// <c>sections</c> (int[] per-axis frequency sections, optional), <c>interleaved</c> (bool, Qwen3-VL style),
    /// <c>freq_scale</c> (float, default 1), <c>attn_factor</c> (float, default 1: cos and sin of the rotated dimensions are multiplied by it, the YaRN magnitude; dimensions beyond <c>dims</c> stay unscaled). Optional third input: f32 [dims/2] per-frequency divisors
    /// (θᵢ = position · base^(-2i/dims) · freq_scale / factorᵢ; llama3 / yarn / longrope scaling, see <c>RopeScaling</c>).
    /// </summary>
    public static readonly OpSchema Rope = new("Rope", OpCategory.Composite, 2, 3, SameF32);

    /// <summary>
    /// Scaled dot-product attention against a KV cache.
    /// q [T, Hq, D]; kcache/vcache: State [Ctx, Hkv, D]/[Ctx, Hkv, Dv]. Query t attends cache rows
    /// [0, offset + t] (causal) where offset is the runtime scalar named by attribute <c>offset</c>
    /// (default "n_past"). Attributes: scale; optional <c>window</c> (int &gt; 0: a query only sees its last <c>window</c> keys,
    /// sliding-window attention), <c>chunk</c> (int &gt; 0: a query at position p only sees keys [p − p mod chunk, p], chunked
    /// attention) and <c>softcap</c> (float &gt; 0: scores = softcap · tanh(scores / softcap)).
    /// Attribute <c>alibi_slopes</c> (float[Hq]): ALiBi, every score gets <c>−slopes[h] · |key position − query position|</c> added before the softmax (BLOOM, MPT;
    /// the bidirectional form serves jina-bert-v2).
    /// Attributes <c>position_bias</c> (float[Hq · (2R + 1)]) and <c>position_bias_range</c> (int R): a learned relative position bias, every score of head h
    /// gets <c>position_bias[h · (2R + 1) + clamp(key position − query position, −R, R) + R]</c> added before the softmax (T5's bucketed bias, expanded per
    /// distance by the adapter; distances beyond R share the bucket of ±R).
    /// Attribute <c>sinks</c> (bool): the input after V is <c>sinks</c> (f32 [Hq]), a learned per-head logit that joins the softmax denominator but has no value (attention sinks).
    /// Attribute <c>v_dims</c> (int &gt; 0): the values are the first <c>v_dims</c> entries of each vcache row, so Dv = v_dims and
    /// vcache may be the very same state as kcache (multi-head latent attention: one latent row per token serves as key and,
    /// through its prefix, as value).
    /// Attribute <c>selection</c> (bool): sparse attention. The input after V (and after the sinks when present) is i32 [T, K]: the key positions
    /// query row t attends (as positions of its own sequence, e.g. from <see cref="IndexerTopK"/>); entries below 0 and positions the row cannot
    /// see are skipped. Not combined with <c>window</c> or <c>chunk</c>.
    /// Output [T, Hq, Dv]. Hq must be a multiple of Hkv (GQA).
    /// Multi-sequence form (see <see cref="MultiSequence"/>): the last two inputs (after the optional sinks and selection) = sequence rows [T, 4] / runs [U, 4], attribute <c>slots</c>
    /// = S; the cache holds S contiguous slot contexts of Ctx/S rows, and row t attends rows [0, position] of its slot.
    /// Paged form: attribute <c>block_size</c> = B and a last input, the block table [S, blocks per slot] i32
    /// (<see cref="Models.InputRole.KvBlockTable"/>); position p of slot s is cache row table[s, p / B] · B + p % B.
    /// </summary>
    public static readonly OpSchema CachedAttention = new("CachedAttention", OpCategory.Composite, 3, 8, (i, a) =>
    {
        var q = i[0].Shape;
        var v = i[2].Shape;
        long valueDims = a.Get("v_dims", 0L);
        if (valueDims > 0 && !v[2].IsSymbolic && valueDims > v[2].Fixed)
        {
            throw new ArgumentException($"CachedAttention v_dims {valueDims} exceeds the value cache rows {v}.");
        }

        return [new(DataType.F32, new Shape(q[0], q[1], valueDims > 0 ? Dim.Of(valueDims) : v[2]))];
    });

    /// <summary>
    /// Lightning-indexer key selection (DeepSeek sparse attention): scores the visible keys of a key cache for every query row and returns the
    /// positions of the best <c>k</c>. Score of key position j for row t: <c>Σ_h w[t, h] · relu(q[t, h] · key[j])</c>.
    /// Inputs: q [T, H, D] f32, kcache State [Ctx, 1, D] (f32 or f16), w [T, H] f32. Query t sees positions [0, offset + t] (attribute <c>offset</c>,
    /// the runtime scalar, default "n_past"). Attribute <c>k</c> (int &gt; 0).
    /// Output i32 [T, k]: the positions whose score exceeds the k-th best score in ascending order, then those equal to it in ascending order
    /// (ties go to lower positions) up to k entries; a row that sees at most k positions gets all of them in ascending order followed by −1.
    /// Multi-sequence and paged forms as <see cref="CachedAttention"/> (sequence rows [T, 4], runs [U, 4], optional block table; attributes
    /// <c>slots</c>, <c>block_size</c>): row t sees the positions [0, position] of its slot.
    /// </summary>
    public static readonly OpSchema IndexerTopK = new("IndexerTopK", OpCategory.Composite, 3, 6, (i, a) =>
    {
        long k = a.Get("k", 0L);
        if (k < 1)
        {
            throw new ArgumentException("IndexerTopK needs k >= 1.");
        }

        return [new(DataType.I32, new Shape(i[0].Shape[0], Dim.Of(k)))];
    });

    /// <summary>
    /// Bidirectional/causal attention without cache (e.g. vision encoders): q [T,Hq,D], k [S,Hkv,D], v [S,Hkv,Dv].
    /// Attributes: scale, causal (bool), optional softcap (float &gt; 0), <c>alibi_slopes</c> and <c>position_bias</c> / <c>position_bias_range</c> (see <see cref="CachedAttention"/>; with i32 spans
    /// the distances are row distances) and <c>window</c> (int &gt; 0: a query only sees keys less than <c>window</c> positions away; with <c>causal</c> that
    /// is its last <c>window</c> keys, without it the symmetric window of bidirectional sliding-window encoders such as ModernBERT and EmbeddingGemma).
    /// Optional fourth input, by type:
    /// <list type="bullet">
    /// <item>f32 [T, S]: additive mask added to the scaled scores.</item>
    /// <item>i32 [T, 2] spans (packed sequences, see <see cref="Models.InputRole.RowSpans"/>): query row t sees key rows
    /// [first, first + count) of its span; with <c>causal</c> additionally only rows ≤ t (self-attention, so q and k rows
    /// coincide). An empty span yields zeros. Lets one pass attend independently within many concatenated sequences.</item>
    /// </list>
    /// </summary>
    public static readonly OpSchema Attention = new("Attention", OpCategory.Composite, 3, 4, (i, a) =>
    {
        var q = i[0].Shape;
        var v = i[2].Shape;
        return [new(DataType.F32, new Shape(q[0], q[1], v[2]))];
    });

    /// <summary>
    /// Writes x [T, ...] into rows [offset, offset+T) of a state tensor [Ctx, ...] (converting type).
    /// Inputs: state, x. Attribute: offset (runtime scalar name, default "n_past"). No outputs; side effect.
    /// Multi-sequence form: third input = sequence rows [T, 4], attribute <c>slots</c>; row t goes to row (slot · Ctx/S + position).
    /// Paged form: attribute <c>block_size</c> and a fourth input, the block table (see <see cref="CachedAttention"/>).
    /// </summary>
    public static readonly OpSchema CacheWrite = new("CacheWrite", OpCategory.Data, 2, 4, (i, a) => []);

    /// <summary>
    /// Causal depthwise 1-D convolution with carried state (Mamba / DeltaNet short conv).
    /// x [T, C], weight [C, K], state State [K-1, C] → y [T, C]. The state is updated with the last
    /// K-1 inputs. Optional 4th input: bias [C] (added before the activation), flagged by attribute <c>bias</c> (bool) or by a non-i32 type.
    /// Attribute: activation ("none" | "silu").
    /// Multi-sequence form: trailing i32 input = sequence runs [U, 4], state [S, K-1, C]; each run continues its slot's state.
    /// Rollback form (<see cref="RecurrentRollback"/>): then the commits table and the record state.
    /// </summary>
    public static readonly OpSchema CausalConv1d = new("CausalConv1d", OpCategory.Composite, 3, 7, (i, a) => [new(DataType.F32, i[0].Shape)]);

    /// <summary>
    /// Gated delta rule recurrence (Gated DeltaNet, Kimi delta attention) and gated linear attention.
    /// q,k [T, Hk, Dk]; v [T, Hv, Dv]; g: log decay, [T, Hv] (one per head), [T, Hv, Dk] (one per key channel, Kimi delta attention) or
    /// [Hv] (time-invariant, e.g. lightning attention's fixed slopes); beta [T, Hv] or [Hv] (time-invariant); state State [Hv, Dv, Dk].
    /// Head h uses q/k head (h mod Hk). Per token, with D = diag(exp(g)) over the key channels (or the scalar exp(g)):
    /// rule "delta" (default): S = D·S; S += k ⊗ (v − Sᵀk)·beta; rule "additive": S = D·S + k ⊗ v·beta (no delta correction);
    /// then o = Sᵀq·scale. Attributes: scale, rule. Output [T, Hv, Dv].
    /// Multi-sequence form: seventh input = sequence runs [U, 4], state [S, Hv, Dv, Dk]; each run continues its slot's state.
    /// Rollback form (<see cref="RecurrentRollback"/>): then the commits table and the record state.
    /// </summary>
    public static readonly OpSchema GatedDeltaRule = new("GatedDeltaRule", OpCategory.Composite, 6, 9, (i, a) => [new(DataType.F32, i[2].Shape)]);

    /// <summary>
    /// Pools consecutive row ranges of x [T, D] (packed sequences): spans i32 [B, 2] of (first row, row count) → [B, D] f32.
    /// Attribute <c>mode</c>: "mean" (default), "sum", "max", "first", "last". An empty span yields zeros.
    /// </summary>
    public static readonly OpSchema SequencePool = new("SequencePool", OpCategory.Reduction, 2, 2, (i, a) =>
        [new(DataType.F32, new Shape(i[1].Shape[0], i[0].Shape[-1]))]);

    /// <summary>
    /// Hashed n-gram row ids (n-gram hash embeddings). Input: window f32 [T, N]: per token, column j holds the token j positions
    /// back plus one (column 0 is the token itself; 0 = no token, e.g. before the sequence start). A missing predecessor, an
    /// end-of-sequence predecessor and everything before it read as the stop token. For every order n = 2..N and each of its
    /// <c>heads_per_order</c> heads h: mixed = XOR over j &lt; n of (token_j · multipliers[j]) in uint64 arithmetic, and
    /// id = mixed mod head_sizes[h] + head_offsets[h]. Attributes: <c>multipliers</c> (long[N]), <c>head_sizes</c> and
    /// <c>head_offsets</c> (long[(N-1)·heads_per_order]), <c>heads_per_order</c> (long), <c>stop_token</c> (long).
    /// Output i32 [T, (N-1)·heads_per_order], heads of order 2 first.
    /// </summary>
    public static readonly OpSchema NGramHash = new("NGramHash", OpCategory.Data, 1, 1, (i, a) =>
        [new(DataType.I32, new Shape(i[0].Shape[0], a.Get<long[]>("head_sizes").Length))]);

    /// <summary>Copies a value between devices (inserted by the partitioner).</summary>
    public static readonly OpSchema Transfer = new("Transfer", OpCategory.Data, 1, 1, Same);

    internal static void RegisterAll()
    {
        foreach (var n in UnaryNames.Except(IntegerPreservingUnaryNames))
        {
            OpRegistry.Register(new OpSchema(n, OpCategory.Elementwise, 1, 1, SameF32));
        }

        foreach (var n in BinaryNames.Except(IntegerPreservingBinaryNames))
        {
            OpRegistry.Register(new OpSchema(n, OpCategory.Elementwise, 2, 2, Broadcast));
        }

        RegisterNumeric();

        foreach (var f in typeof(StandardOps).GetFields())
        {
            if (f.GetValue(null) is OpSchema s)
            {
                OpRegistry.Register(s);
            }
        }
    }

    // ---- inference helpers ----------------------------------------------------------------
    private static IReadOnlyList<TensorType> Same(IReadOnlyList<Value> i, Attributes a) => [new(i[0].Type, i[0].Shape)];
    private static IReadOnlyList<TensorType> SameF32(IReadOnlyList<Value> i, Attributes a) => [new(DataType.F32, i[0].Shape)];

    internal static Shape BroadcastShapes(Shape a, Shape b)
    {
        int rank = Math.Max(a.Rank, b.Rank);
        var dims = new Dim[rank];
        for (int k = 0; k < rank; k++)
        {
            int ia = a.Rank - rank + k;
            int ib = b.Rank - rank + k;
            Dim da = ia >= 0 ? a.Dims[ia] : 1;
            Dim db = ib >= 0 ? b.Dims[ib] : 1;
            if (da == db)
            {
                dims[k] = da;
            }
            else if (!da.IsSymbolic && da.Fixed == 1)
            {
                dims[k] = db;
            }
            else if (!db.IsSymbolic && db.Fixed == 1)
            {
                dims[k] = da;
            }
            else
            {
                throw new ArgumentException($"Shapes {a} and {b} are not broadcast-compatible.");
            }
        }

        return new Shape(dims);
    }

    private static IReadOnlyList<TensorType> Broadcast(IReadOnlyList<Value> i, Attributes a) =>
        [new(DataType.F32, BroadcastShapes(i[0].Shape, i[1].Shape))];

    private static Shape ReshapeToOuterSymbol(Shape src, long[] req, Dim outer)
    {
        if (src.Rank == 0 || !src[0].IsSymbolic || req.Length < 1 || req.Skip(1).Any(d => d < 0))
        {
            throw new ArgumentException("Reshape outer_symbol needs a symbolic outer input dimension and fixed inner target dimensions.");
        }

        var dims = new Dim[req.Length];
        dims[0] = outer;
        long inner = 1;
        for (int k = 1; k < req.Length; k++)
        {
            dims[k] = req[k];
            inner *= req[k];
        }

        var source = src[0].Expression * src.InnerElements;
        var target = outer.Expression * inner;
        bool related = outer.Expression.Atoms().All(s => source.Atoms().Contains(s));
        if (related && !source.Equals(target))
        {
            throw new ArgumentException($"Reshape {src} → [{outer}, {string.Join(",", req.Skip(1))}] changes the element count ({source} vs {target}).");
        }

        return new Shape(dims);
    }

    private static IReadOnlyList<TensorType> InferReshape(IReadOnlyList<Value> i, Attributes a)
    {
        var src = i[0].Shape;
        var req = a.Get<long[]>("shape");
        var dims = new Dim[req.Length];
        if (a.Get("outer_symbol", "") is { Length: > 0 } outer)
        {
            return [new(i[0].Type, ReshapeToOuterSymbol(src, req, Dim.Sym(outer)))];
        }

        long known = 1;
        int infer = -1;
        bool srcSym = src.Rank > 0 && src[0].IsSymbolic;
        for (int k = 0; k < req.Length; k++)
        {
            if (req[k] == 0 && k == 0 && srcSym)
            {
                dims[k] = src[0];
                continue;
            }

            if (req[k] == -1)
            {
                if (infer >= 0)
                {
                    throw new ArgumentException("Only one -1 allowed in reshape.");
                }

                infer = k;
                continue;
            }

            dims[k] = req[k];
            known *= req[k];
        }
        long total = srcSym ? src.InnerElements : src.Elements();
        if (srcSym && !(req.Length > 0 && req[0] == 0))
        {
            throw new ArgumentException("Reshape of a tensor with a symbolic outer dim must keep it (use 0 as first dim).");
        }

        if (infer >= 0)
        {
            dims[infer] = total / known;
        }
        else if (known != total)
        {
            throw new ArgumentException($"Reshape {src} → [{string.Join(",", req)}] changes element count.");
        }

        return [new(i[0].Type, new Shape(dims))];
    }
}
