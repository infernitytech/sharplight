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
    public static readonly string[] UnaryNames =
        ["Neg", "Exp", "Log", "Sqrt", "Rsqrt", "Sigmoid", "Silu", "Gelu", "GeluTanh", "Tanh", "Softplus", "Relu", "Abs", "Square", "Recip",
         "Erf", "HardSwish", "HardSigmoid" /* attrs alpha (0.2), beta (0.5) */,
         "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh",
         "Floor", "Ceil", "Round" /* half to even */, "Sign", "Softsign", "Mish",
         "Elu" /* alpha (1) */, "Celu" /* alpha (1) */, "Selu" /* alpha (1.67326), gamma (1.0507) */,
         "LeakyRelu" /* alpha (0.01) */, "ThresholdedRelu" /* alpha (1) */,
         // Mask-producing / mask-consuming (masks are f32 0/1; non-zero means true):
         "Not", "IsNaN", "IsInf"];
    public static readonly string[] BinaryNames =
        ["Add", "Sub", "Mul", "Div", "Max", "Min", "Pow",
         "Mod" /* attr fmod (bool): C fmod vs. Python-style sign of divisor */,
         "PRelu" /* x, slope */,
         // Comparisons and logic produce f32 masks (1 = true, 0 = false):
         "Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual", "And", "Or", "Xor"];

    /// <summary>y = x * scale + bias. Attributes: scale (float), bias (float).</summary>
    public static readonly OpSchema Scale = new("Scale", OpCategory.Elementwise, 1, 1, Same, "y = x*scale + bias");

    /// <summary>Element type conversion. Attribute: type (string data type name).</summary>
    public static readonly OpSchema Cast = new("Cast", OpCategory.Elementwise, 1, 1,
        (i, a) => [new TensorType(DataType.Get(a.Get<string>("type")), i[0].Shape)]);

    // ---- views ----------------------------------------------------------------------------
    /// <summary>
    /// Reshape (row-major). Attribute: shape (long[]; -1 = infer, 0 = keep symbolic outer dim). Optional attribute
    /// <c>outer_symbol</c> (string): the first entry of <c>shape</c> is then a new symbolic outer dimension of that name
    /// (e.g. N patches → M = N/4 merged tokens); the caller binds both symbols consistently at run time.
    /// </summary>
    public static readonly OpSchema Reshape = new("Reshape", OpCategory.View, 1, 1, InferReshape);

    /// <summary>Strided slice along one axis. Attributes: axis, start, length.</summary>
    public static readonly OpSchema Slice = new("Slice", OpCategory.View, 1, 1, (i, a) =>
    {
        int axis = a.Get<int>("axis");
        if (axis < 0) axis += i[0].Rank;
        if (axis == 0 && i[0].Shape[0].IsSymbolic) throw new NotSupportedException("Slicing the symbolic axis is not supported.");
        long len = a.Get<long>("length");
        return [new TensorType(i[0].Type, i[0].Shape.WithDim(axis, len))];
    });

    /// <summary>Axis permutation (lazy view). Attribute: perm (int[]).</summary>
    public static readonly OpSchema Permute = new("Permute", OpCategory.View, 1, 1, (i, a) =>
    {
        var perm = a.Get<int[]>("perm");
        return [new TensorType(i[0].Type, new Shape(perm.Select(p => i[0].Shape[p])))];
    });

    /// <summary>Materializes a (possibly strided) view into a dense tensor.</summary>
    public static readonly OpSchema Contiguous = new("Contiguous", OpCategory.Data, 1, 1, Same);

    /// <summary>Concatenate along an axis. Attribute: axis.</summary>
    public static readonly OpSchema Concat = new("Concat", OpCategory.Data, 2, 64, (i, a) =>
    {
        int axis = a.Get<int>("axis");
        if (axis < 0) axis += i[0].Rank;
        long sum = i.Sum(v => v.Shape[axis].Fixed);
        return [new TensorType(i[0].Type, i[0].Shape.WithDim(axis, sum))];
    });

    /// <summary>Row gather: table [V, ...], indices i32 [N] → [N, ...] (dequantizes to f32).</summary>
    public static readonly OpSchema Gather = new("Gather", OpCategory.Data, 2, 2, (i, a) =>
    {
        var t = i[0].Shape;
        var dims = new List<Dim> { i[1].Shape[0] };
        for (int d = 1; d < t.Rank; d++) dims.Add(t[d]);
        // Rows are decoded into the activation type (f32 unless "out_type" says otherwise), whatever the table format.
        return [new TensorType(a.Get("out_type", DataType.F32), new Shape(dims))];
    });

    // ---- linear algebra -------------------------------------------------------------------
    /// <summary>
    /// y = x · Wᵀ. x: [..., K] (f32), W: [N, K] (any float or quantized format) → [..., N].
    /// This is the "Linear" layout used by GGUF/PyTorch weights.
    /// </summary>
    public static readonly OpSchema MatMul = new("MatMul", OpCategory.MatMul, 2, 2, (i, a) =>
    {
        var x = i[0].Shape; var w = i[1].Shape;
        if (w.Rank != 2) throw new ArgumentException("MatMul weight must be rank 2 [N, K].");
        if (x[-1] != w[1]) throw new ArgumentException($"MatMul K mismatch: x{x} w{w}.");
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
    /// <summary>Sum over the last axis (keeps a dimension of size 1).</summary>
    public static readonly OpSchema ReduceSum = new("ReduceSum", OpCategory.Reduction, 1, 1, (i, a) => [new(DataType.F32, i[0].Shape.WithLast(1))]);
    public static readonly OpSchema ReduceMax = new("ReduceMax", OpCategory.Reduction, 1, 1, (i, a) => [new(DataType.F32, i[0].Shape.WithLast(1))]);
    /// <summary>Index of the maximum along the last axis → i32 [..., 1].</summary>
    public static readonly OpSchema ArgMax = new("ArgMax", OpCategory.Reduction, 1, 1, (i, a) => [new(DataType.I32, i[0].Shape.WithLast(1))]);

    // ---- composite ------------------------------------------------------------------------
    /// <summary>
    /// Rotary position embedding on x [T, H, D] with positions i32 [T, A] (A position axes; A=1 for
    /// plain RoPE, &gt;1 for multi-axis / M-RoPE). Attributes:
    /// <c>dims</c> (rotated dims, ≤ D), <c>base</c>, <c>style</c> ("neox" pairs (i, i+dims/2), "gptj" pairs (2i, 2i+1)),
    /// <c>sections</c> (int[] per-axis frequency sections, optional), <c>interleaved</c> (bool, Qwen3-VL style),
    /// <c>freq_scale</c> (float, default 1). Optional third input: f32 [dims/2] per-frequency divisors
    /// (θᵢ = position · base^(-2i/dims) · freq_scale / factorᵢ; llama3 / yarn / longrope scaling, see <c>RopeScaling</c>).
    /// </summary>
    public static readonly OpSchema Rope = new("Rope", OpCategory.Composite, 2, 3, SameF32);

    /// <summary>
    /// Scaled dot-product attention against a KV cache.
    /// q [T, Hq, D]; kcache/vcache: State [Ctx, Hkv, D]/[Ctx, Hkv, Dv]. Query t attends cache rows
    /// [0, offset + t] (causal) where offset is the runtime scalar named by attribute <c>offset</c>
    /// (default "n_past"). Attributes: scale; optional <c>window</c> (int &gt; 0: a query only sees its last <c>window</c> keys,
    /// sliding-window attention) and <c>softcap</c> (float &gt; 0: scores = softcap · tanh(scores / softcap)).
    /// Output [T, Hq, Dv]. Hq must be a multiple of Hkv (GQA).
    /// Multi-sequence form (see <see cref="MultiSequence"/>): fourth / fifth inputs = sequence rows [T, 4] / runs [U, 4], attribute <c>slots</c>
    /// = S; the cache holds S contiguous slot contexts of Ctx/S rows, and row t attends rows [0, position] of its slot.
    /// </summary>
    public static readonly OpSchema CachedAttention = new("CachedAttention", OpCategory.Composite, 3, 5, (i, a) =>
    {
        var q = i[0].Shape; var v = i[2].Shape;
        return [new(DataType.F32, new Shape(q[0], q[1], v[2]))];
    });

    /// <summary>
    /// Bidirectional/causal attention without cache (e.g. vision encoders): q [T,Hq,D], k [S,Hkv,D], v [S,Hkv,Dv].
    /// Attributes: scale, causal (bool).
    /// </summary>
    public static readonly OpSchema Attention = new("Attention", OpCategory.Composite, 3, 4, (i, a) =>
    {
        var q = i[0].Shape; var v = i[2].Shape;
        return [new(DataType.F32, new Shape(q[0], q[1], v[2]))];
    });

    /// <summary>
    /// Writes x [T, ...] into rows [offset, offset+T) of a state tensor [Ctx, ...] (converting type).
    /// Inputs: state, x. Attribute: offset (runtime scalar name, default "n_past"). No outputs; side effect.
    /// Multi-sequence form: third input = sequence rows [T, 4], attribute <c>slots</c>; row t goes to row (slot · Ctx/S + position).
    /// </summary>
    public static readonly OpSchema CacheWrite = new("CacheWrite", OpCategory.Data, 2, 3, (i, a) => []);

    /// <summary>
    /// Causal depthwise 1-D convolution with carried state (Mamba / DeltaNet short conv).
    /// x [T, C], weight [C, K], state State [K-1, C] → y [T, C]. The state is updated with the last
    /// K-1 inputs. Attribute: activation ("none" | "silu").
    /// Multi-sequence form: fourth input = sequence runs [U, 4], state [S, K-1, C]; each run continues its slot's state.
    /// </summary>
    public static readonly OpSchema CausalConv1d = new("CausalConv1d", OpCategory.Composite, 3, 4, (i, a) => [new(DataType.F32, i[0].Shape)]);

    /// <summary>
    /// Gated delta rule recurrence (Gated DeltaNet linear attention).
    /// q,k [T, Hk, Dk]; v [T, Hv, Dv]; g [T, Hv] (log decay); beta [T, Hv]; state State [Hv, Dv, Dk].
    /// Head h uses q/k head (h mod Hk). Per token: S = exp(g)·S; S += k ⊗ (v − Sᵀk)·beta; o = Sᵀq·scale.
    /// Attribute: scale. Output [T, Hv, Dv].
    /// Multi-sequence form: seventh input = sequence runs [U, 4], state [S, Hv, Dv, Dk]; each run continues its slot's state.
    /// </summary>
    public static readonly OpSchema GatedDeltaRule = new("GatedDeltaRule", OpCategory.Composite, 6, 7, (i, a) => [new(DataType.F32, i[2].Shape)]);

    /// <summary>Copies a value between devices (inserted by the partitioner).</summary>
    public static readonly OpSchema Transfer = new("Transfer", OpCategory.Data, 1, 1, Same);

    internal static void RegisterAll()
    {
        foreach (var n in UnaryNames) OpRegistry.Register(new OpSchema(n, OpCategory.Elementwise, 1, 1, SameF32));
        foreach (var n in BinaryNames) OpRegistry.Register(new OpSchema(n, OpCategory.Elementwise, 2, 2, Broadcast));
        foreach (var f in typeof(StandardOps).GetFields())
            if (f.GetValue(null) is OpSchema s) OpRegistry.Register(s);
    }

    // ---- inference helpers ----------------------------------------------------------------
    static IReadOnlyList<TensorType> Same(IReadOnlyList<Value> i, Attributes a) => [new(i[0].Type, i[0].Shape)];
    static IReadOnlyList<TensorType> SameF32(IReadOnlyList<Value> i, Attributes a) => [new(DataType.F32, i[0].Shape)];

    public static Shape BroadcastShapes(Shape a, Shape b)
    {
        int rank = Math.Max(a.Rank, b.Rank);
        var dims = new Dim[rank];
        for (int k = 0; k < rank; k++)
        {
            int ia = a.Rank - rank + k, ib = b.Rank - rank + k;
            Dim da = ia >= 0 ? a.Dims[ia] : 1;
            Dim db = ib >= 0 ? b.Dims[ib] : 1;
            if (da == db) dims[k] = da;
            else if (!da.IsSymbolic && da.Fixed == 1) dims[k] = db;
            else if (!db.IsSymbolic && db.Fixed == 1) dims[k] = da;
            else throw new ArgumentException($"Shapes {a} and {b} are not broadcast-compatible.");
        }
        return new Shape(dims);
    }

    static IReadOnlyList<TensorType> Broadcast(IReadOnlyList<Value> i, Attributes a) =>
        [new(DataType.F32, BroadcastShapes(i[0].Shape, i[1].Shape))];

    static IReadOnlyList<TensorType> InferReshape(IReadOnlyList<Value> i, Attributes a)
    {
        var src = i[0].Shape;
        var req = a.Get<long[]>("shape");
        var dims = new Dim[req.Length];
        if (a.Get("outer_symbol", "") is { Length: > 0 } outer)
        {
            if (!src[0].IsSymbolic || req.Length < 1 || req[0] != 0) throw new ArgumentException("Reshape outer_symbol needs a symbolic outer dimension and shape[0] = 0.");
            dims[0] = Dim.Sym(outer);
            for (int k = 1; k < req.Length; k++) dims[k] = req[k];
            return [new(i[0].Type, new Shape(dims))];
        }
        long known = 1; int infer = -1;
        bool srcSym = src.Rank > 0 && src[0].IsSymbolic;
        for (int k = 0; k < req.Length; k++)
        {
            if (req[k] == 0 && k == 0 && srcSym) { dims[k] = src[0]; continue; }
            if (req[k] == -1) { if (infer >= 0) throw new ArgumentException("Only one -1 allowed in reshape."); infer = k; continue; }
            dims[k] = req[k];
            known *= req[k];
        }
        long total = srcSym ? src.InnerElements : src.Elements();
        if (srcSym && !(req.Length > 0 && req[0] == 0))
            throw new ArgumentException("Reshape of a tensor with a symbolic outer dim must keep it (use 0 as first dim).");
        if (infer >= 0) dims[infer] = total / known;
        else if (known != total) throw new ArgumentException($"Reshape {src} → [{string.Join(",", req)}] changes element count.");
        return [new(i[0].Type, new Shape(dims))];
    }
}
