using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

/// <summary>An ONNX value during translation: either known at import time or a dynamic IR value.</summary>
internal sealed class Tn
{
    public Value? V;
    public OnnxConst? C;
    public Value? CachedValue;
    public bool IsConst => C is not null;
    public static Tn Dyn(Value v) => new() { V = v };
    public static Tn Const(OnnxConst c) => new() { C = c };
}

/// <summary>Translates a parsed ONNX graph into a Sharplight <see cref="ComputeGraph"/>.</summary>
internal sealed partial class OnnxTranslator
{
    public const string BatchSymbol = "N";

    static readonly HashSet<string> s_dynamicOps =
    [
        "Conv", "BatchNormalization", "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool", "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin",
        "ReduceL2", "ReduceSumSquare", "Gemm", "MatMul", "Relu", "Sigmoid", "Tanh", "Exp", "Log", "Sqrt", "Neg", "Abs", "Erf", "Reciprocal", "HardSigmoid",
        "HardSwish", "Clip", "LeakyRelu", "Gelu", "Softplus", "Mish", "PRelu", "Add", "Sub", "Mul", "Div", "Pow", "Max", "Min", "Sum", "Mean", "Flatten",
        "Reshape", "Squeeze", "Unsqueeze", "Transpose", "Concat", "Slice", "Softmax", "LogSoftmax", "LayerNormalization", "InstanceNormalization", "Pad",
        "Resize", "Upsample", "Identity", "Dropout", "Split", "Gather", "Expand", "Shape", "Size", "Cast",
        "Elu", "Celu", "Selu", "ThresholdedRelu", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh", "Asinh", "Acosh", "Atanh", "Floor", "Ceil", "Round",
        "Sign", "Softsign", "Not", "IsNaN", "IsInf", "Swish", "Silu", "Mod", "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual", "And", "Or", "Xor", "Where",
        "ReduceL1", "ReduceProd", "ReduceLogSum", "ReduceLogSumExp", "ArgMax", "ArgMin", "TopK", "Hardmax", "GroupNormalization", "LpNormalization", "LRN",
        "MeanVarianceNormalization", "ConvTranspose", "LpPool", "Einsum", "Tile", "GatherElements", "GatherND", "ScatterElements", "ScatterND", "Scatter", "CumSum",
        "DepthToSpace", "SpaceToDepth", "Trilu", "OneHot", "GridSample", "CastLike", "QuantizeLinear", "DequantizeLinear", "MatMulInteger", "ConvInteger",
        "QLinearMatMul", "QLinearConv", "If", "LSTM", "GRU", "RNN", "RMSNormalization", "NonZero", "EyeLike", "DynamicQuantizeLinear",
    ];

    static readonly HashSet<string> s_contribOps =
    [
        "Gelu", "BiasGelu", "FastGelu", "QuickGelu", "FusedConv", "FusedMatMul", "SkipLayerNormalization", "SkipSimplifiedLayerNormalization",
        "SimplifiedLayerNormalization", "Attention", "MultiHeadAttention", "BiasAdd", "MatMulNBits",
    ];

    public static bool IsSupported(string opType) => s_dynamicOps.Contains(opType) || OnnxFold.Ops.Contains(opType);

    readonly OnnxModel _model;
    readonly OnnxGraph _graph;
    readonly OnnxImportOptions _opts;
    readonly Dictionary<string, WeightInfo> _weights = new(StringComparer.Ordinal);
    readonly Dictionary<string, Tn> _env = new(StringComparer.Ordinal);
    readonly Dictionary<(OnnxConst, string), Value> _constValues = new();
    readonly HashSet<string> _usedNames = new(StringComparer.Ordinal);
    readonly Dictionary<string, List<OnnxNode>> _consumers = new(StringComparer.Ordinal);
    readonly HashSet<OnnxNode> _skip = [];
    readonly HashSet<string> _graphOutputs;
    GraphBuilder _b = null!;
    IDisposable? _groupScope;
    int _groupIndex, _heavyInGroup;
    Value? _batchZero;
    Value? _firstInput;
    int _anon;

    public OnnxTranslator(OnnxModel model)
    {
        _model = model;
        _graph = model.Graph;
        _opts = model.Options;
        _graphOutputs = _graph.Outputs.Select(o => o.Name).ToHashSet(StringComparer.Ordinal);
    }

    public IReadOnlyDictionary<string, WeightInfo> Weights => _weights;
    public ComputeGraph Graph { get; private set; } = null!;
    public long Opset => _model.Opset();

    public ComputeGraph Translate(string graphName)
    {
        _b = new GraphBuilder(graphName, n => _weights.GetValueOrDefault(n));

        var initNames = new HashSet<string>(StringComparer.Ordinal);
        foreach (var t in _graph.Initializers)
        {
            if (t.Source is null) continue;
            initNames.Add(t.Name);
            _env[t.Name] = Tn.Const(OnnxConst.FromTensor(t));
        }

        var order = Order(_graph.Nodes, initNames);
        var live = Liveness(order);
        CheckSupport(live);
        foreach (var n in live)
            foreach (var i in n.Inputs) if (i != "") { _usedNames.Add(i); if (!_consumers.TryGetValue(i, out var l)) _consumers[i] = l = []; l.Add(n); }

        NextGroup();
        foreach (var vi in _graph.Inputs)
        {
            if (initNames.Contains(vi.Name)) continue;
            var shape = InputShape(vi);
            bool isInt = vi.ElemType is OnnxType.Int64 or OnnxType.Int32;
            if (!OnnxType.IsFloat(vi.ElemType) && !isInt && vi.ElemType != OnnxType.Bool)
                throw new NotSupportedException($"Graph input '{vi.Name}' has type {OnnxType.Name(vi.ElemType)}; only float, int32/int64 and bool inputs are supported.");
            bool image = !isInt && shape.Rank == 4 && shape[1].Fixed is 1 or 3 or 4;
            var v = _b.Input(vi.Name, image ? InputRole.Media : isInt ? InputRole.TokenIds : InputRole.Other, isInt ? DataType.I32 : DataType.F32, shape);
            if (!isInt) _firstInput ??= v;
            _env[vi.Name] = Tn.Dyn(v);
        }
        if (_graph.Inputs.All(i => initNames.Contains(i.Name))) throw new NotSupportedException("The ONNX graph has no dynamic inputs.");

        foreach (var n in live)
        {
            if (_skip.Contains(n)) continue;
            try { HandleNode(n); }
            catch (Exception e) when (e is not NotSupportedException && e is not InvalidDataException)
            {
                throw new InvalidDataException($"Translating {n.OpType} node '{n.Name}' ({string.Join(", ", n.Outputs)}): {e.Message}", e);
            }
            catch (NotSupportedException e) when (!e.Message.Contains("node '"))
            {
                throw new NotSupportedException($"{n.OpType} node '{n.Name}' ({string.Join(", ", n.Outputs)}): {e.Message}", e);
            }
        }

        foreach (var o in _graph.Outputs)
        {
            if (!_env.TryGetValue(o.Name, out var t)) throw new InvalidDataException($"Graph output '{o.Name}' is never produced.");
            if (t.C is not null) throw new NotSupportedException($"Graph output '{o.Name}' is a compile-time constant; outputs must depend on an input.");
            var ov = t.V!;
            if (ov.Kind != ValueKind.Intermediate) ov = _b.Scale(ov, 1f, 0f, o.Name);   // outputs must be produced by a node
            _b.Output(o.Name, ov);
        }
        _groupScope?.Dispose();
        return Graph = _b.Build();
    }

    // ---- inputs ------------------------------------------------------------------------------------
    Shape InputShape(OnnxValueInfo vi)
    {
        long[]? ov = _opts.InputShapes.GetValueOrDefault(vi.Name);
        int rank = ov?.Length ?? vi.Shape?.Count ?? throw new NotSupportedException($"Graph input '{vi.Name}' has no shape; supply one through OnnxImportOptions.InputShapes.");
        if (rank == 0) throw new NotSupportedException($"Graph input '{vi.Name}' is a scalar.");
        var dims = new Dim[rank];
        for (int i = 0; i < rank; i++)
        {
            long? fixedV = ov is not null ? (ov[i] > 0 ? ov[i] : null) : vi.Shape![i].IsKnown ? vi.Shape[i].Value : null;
            if (i == 0 && (_opts.ForceSymbolicBatch || fixedV is null)) dims[i] = Dim.Sym(BatchSymbol);
            else if (fixedV is { } f) dims[i] = f;
            else throw new NotSupportedException($"Graph input '{vi.Name}' has a dynamic dimension {i} ('{vi.Shape![i]}'); only the outermost (batch) dimension may be symbolic. Fix it via OnnxImportOptions.InputShapes.");
        }
        return new Shape(dims);
    }

    // ---- scheduling / analysis -----------------------------------------------------------------------
    static List<OnnxNode> Order(IReadOnlyList<OnnxNode> nodes, HashSet<string> initNames)
    {
        // ONNX requires topological order, but be tolerant: Kahn's algorithm keeping the original order stable.
        var produced = new HashSet<string>(initNames, StringComparer.Ordinal);
        var producers = new HashSet<string>(StringComparer.Ordinal);
        foreach (var n in nodes) foreach (var o in n.Outputs) producers.Add(o);
        var result = new List<OnnxNode>(nodes.Count);
        var pending = new List<OnnxNode>(nodes);
        bool ok = true;
        foreach (var n in nodes)
        {
            if (n.Inputs.Any(i => i != "" && producers.Contains(i) && !produced.Contains(i))) { ok = false; break; }
            foreach (var o in n.Outputs) produced.Add(o);
        }
        if (ok) return pending;
        produced = new HashSet<string>(initNames, StringComparer.Ordinal);
        while (pending.Count > 0)
        {
            int before = pending.Count;
            for (int k = 0; k < pending.Count; k++)
            {
                var n = pending[k];
                if (n.Inputs.Any(i => i != "" && producers.Contains(i) && !produced.Contains(i))) continue;
                result.Add(n); foreach (var o in n.Outputs) produced.Add(o);
                pending.RemoveAt(k--);
            }
            if (pending.Count == before) throw new InvalidDataException("The ONNX graph contains a cycle or an undefined value.");
        }
        return result;
    }

    List<OnnxNode> Liveness(List<OnnxNode> order)
    {
        var producer = new Dictionary<string, OnnxNode>(StringComparer.Ordinal);
        foreach (var n in order) foreach (var o in n.Outputs) if (o != "") producer[o] = n;
        var live = new HashSet<OnnxNode>();
        var stack = new Stack<string>(_graph.Outputs.Select(o => o.Name));
        var seen = new HashSet<string>(StringComparer.Ordinal);
        while (stack.Count > 0)
        {
            var name = stack.Pop();
            if (!seen.Add(name) || !producer.TryGetValue(name, out var n)) continue;
            if (!live.Add(n)) continue;
            foreach (var i in n.Inputs) if (i != "") stack.Push(i);
        }
        return order.Where(live.Contains).ToList();
    }

    void CheckSupport(List<OnnxNode> live)
    {
        var bad = live.Where(n => !IsSupportedNode(n)).Select(n => n.Domain is "" or "ai.onnx" ? n.OpType : $"{n.Domain}.{n.OpType}")
            .Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        if (bad.Count > 0)
            throw new NotSupportedException($"The ONNX model uses {bad.Count} unsupported operator type(s): {string.Join(", ", bad)}.");
    }

    static bool IsSupportedNode(OnnxNode n)
        => n.Domain is "" or "ai.onnx" ? IsSupported(n.OpType) : n.Domain == "com.microsoft" && (s_contribOps.Contains(n.OpType) || s_quantContrib.Contains(n.OpType));

    // ---- group handling --------------------------------------------------------------------------------
    void NextGroup()
    {
        _groupScope?.Dispose();
        _groupScope = _b.Group($"onnx.{_groupIndex++:D3}");
        _heavyInGroup = 0;
    }

    void Heavy()
    {
        if (++_heavyInGroup >= Math.Max(1, _opts.GroupSize)) NextGroup();
    }

    // ---- constants -> IR ---------------------------------------------------------------------------------
    internal const int MaxInlineConstElements = 4096;

    Value AsValue(Tn t, string? hint = null)
    {
        if (t.V is not null)
        {
            if (t.V.Type == DataType.F32) return t.V;
            return t.CachedValue ??= _b.Emit1("Cast", [t.V], Attributes.From(("type", "f32")));
        }
        return t.CachedValue ??= ConstValue(t.C!, hint);
    }

    string UniqueWeightName(string baseName)
    {
        if (baseName == "") baseName = $"const.{_anon++}";
        if (!_weights.ContainsKey(baseName)) return baseName;
        int i = 1;
        while (_weights.ContainsKey($"{baseName}#{i}")) i++;
        return $"{baseName}#{i}";
    }

    /// <summary>f32 IR value for a constant: a small graph Constant, or a (possibly zero-copy) Weight.</summary>
    Value ConstValue(OnnxConst c, string? hint)
    {
        var dims = c.Rank == 0 ? [1L] : c.Dims;
        if (_constValues.TryGetValue((c, "f32"), out var cached)) return cached;
        var shape = new Shape(dims.Select(d => (Dim)d).ToArray());
        Value v;
        string name = c.Name != "" ? c.Name : hint ?? $"const.{_anon}";
        if (c.Count <= MaxInlineConstElements)
            v = _b.Constant(name, DataType.F32, shape, c.F32Bytes());
        else
        {
            var wname = UniqueWeightName(name);
            _weights[wname] = c.ElemType == OnnxType.Float && c.HasBackingSource
                ? new WeightInfo { Name = wname, Type = DataType.F32, Shape = shape, Source = c.Source! }
                : WeightInfo.Derived(wname, DataType.F32, shape, c.F32Bytes);
            v = _b.Weight(wname);
        }
        _constValues[(c, "f32")] = v;
        return v;
    }

    /// <summary>
    /// Weight value for matmul/conv: keeps f32/f16/bf16 storage (zero-copy when backed by the mapped file),
    /// otherwise converts to f32.
    /// </summary>
    Value WeightValue(OnnxConst c, bool keepHalf, string? nameHint = null)
    {
        var shape = new Shape(c.Dims.Select(d => (Dim)d).ToArray());
        bool half = keepHalf && c.ElemType is OnnxType.Float16 or OnnxType.BFloat16;
        var key = (c, half ? "w:half" : "w:f32");
        if (_constValues.TryGetValue(key, out var cached)) return cached;
        var wname = UniqueWeightName(c.Name != "" ? c.Name : nameHint ?? "");
        WeightInfo w;
        if ((c.ElemType == OnnxType.Float || half) && c.HasBackingSource)
            w = new WeightInfo { Name = wname, Type = OnnxImporter.ToDataType(c.ElemType)!, Shape = shape, Source = c.Source! };
        else if (c.ElemType == OnnxType.Float || half)
            w = WeightInfo.Derived(wname, OnnxImporter.ToDataType(c.ElemType)!, shape, c.Bytes);
        else
            w = WeightInfo.Derived(wname, DataType.F32, shape, c.F32Bytes);
        _weights[wname] = w;
        var v = _b.Weight(wname);
        _constValues[key] = v;
        return v;
    }

    // ---- node dispatch -------------------------------------------------------------------------------------
    Tn Get(string name)
        => _env.TryGetValue(name, out var t) ? t : throw new InvalidDataException($"ONNX value '{name}' is used before it is defined.");

    void HandleNode(OnnxNode n)
    {
        var ins = new Tn?[n.Inputs.Count];
        for (int i = 0; i < ins.Length; i++) ins[i] = n.Inputs[i] == "" ? null : Get(n.Inputs[i]);

        bool allConst = ins.All(t => t is null || t.IsConst);
        if (allConst && OnnxFold.Ops.Contains(n.OpType) && !(n.OpType == "Expand" && ins[1]!.C!.Ints().Any(v => v == OnnxConst.SymBatch)))
        {
            var res = OnnxFold.Eval(n, ins.Select(t => t?.C).ToList(), Opset);
            for (int i = 0; i < n.Outputs.Count && i < res.Count; i++)
                if (n.Outputs[i] != "")
                {
                    var c = res[i];
                    _env[n.Outputs[i]] = Tn.Const(c.Name == "" ? c.Reshaped(c.Dims, n.Outputs[i]) : c);
                }
            return;
        }
        Dynamic(n, ins);
    }

    void SetDyn(OnnxNode n, int idx, Value v) => _env[n.Outputs[idx]] = Tn.Dyn(v);

    void CheckUnusedOutputs(OnnxNode n, int from)
    {
        for (int i = from; i < n.Outputs.Count; i++)
            if (n.Outputs[i] != "" && (_usedNames.Contains(n.Outputs[i]) || _graphOutputs.Contains(n.Outputs[i])))
                throw new NotSupportedException($"output #{i} ('{n.Outputs[i]}') is used but not supported.");
    }

    static string Nm(OnnxNode n, int i = 0) => n.Outputs.Count > i && n.Outputs[i] != "" ? n.Outputs[i] : $"{n.OpType.ToLowerInvariant()}_{n.Name}";

    static bool Sym(Value v) => v.Rank > 0 && v.Shape[0].IsSymbolic;
}
