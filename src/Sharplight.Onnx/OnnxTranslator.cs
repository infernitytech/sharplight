using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

/// <summary>Translates a parsed ONNX graph into a Sharplight <see cref="ComputeGraph"/>.</summary>
internal sealed partial class OnnxTranslator
{
    /// <summary>Name of the symbolic outermost (batch) dimension.</summary>
    public const string BatchSymbol = "N";

    private readonly OnnxModel _model;
    private readonly OnnxGraph _graph;
    private readonly OnnxImportOptions _opts;
    private readonly Dictionary<string, WeightInfo> _weights = new(StringComparer.Ordinal);
    private readonly Dictionary<string, TranslatedTensor> _env = new(StringComparer.Ordinal);
    private Dictionary<(OnnxConst, string), Value> _constValues = new();
    private readonly HashSet<string> _usedNames = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<OnnxNode>> _consumers = new(StringComparer.Ordinal);
    private readonly HashSet<OnnxNode> _skip = [];
    private readonly HashSet<string> _graphOutputs;
    private GraphBuilder _b = null!;
    private IDisposable? _groupScope;
    private int _groupIndex;
    private int _heavyInGroup;
    private int _anon;

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

    /// <summary>Translates the graph; call once per translator instance.</summary>
    public ComputeGraph Translate(string graphName)
    {
        _b = new GraphBuilder(graphName, n => _weights.GetValueOrDefault(n));

        var initializerNames = RegisterInitializers();
        var order = Order(_graph.Nodes, initializerNames);
        var live = Liveness(order);
        CheckSupport(live);
        IndexConsumers(live);

        NextGroup();
        foreach (var input in _graph.Inputs)
        {
            if (!initializerNames.Contains(input.Name))
            {
                DeclareInput(input);
            }
        }

        if (_graph.Inputs.All(i => initializerNames.Contains(i.Name) || _opts.ConstantInputs.ContainsKey(i.Name)))
        {
            throw new NotSupportedException("The ONNX graph has no dynamic inputs.");
        }

        foreach (var node in live)
        {
            if (!_skip.Contains(node))
            {
                TranslateNode(node);
            }
        }

        foreach (var output in _graph.Outputs)
        {
            DeclareOutput(output);
        }

        _groupScope?.Dispose();
        return Graph = _b.Build();
    }

    /// <summary>Makes every initializer with data available as a constant; returns the names of all initializers.</summary>
    private HashSet<string> RegisterInitializers()
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var tensor in _graph.Initializers)
        {
            if (tensor.Source is null)
            {
                continue;
            }

            names.Add(tensor.Name);
            _env[tensor.Name] = TranslatedTensor.Const(OnnxConst.FromTensor(tensor));
        }

        return names;
    }

    /// <summary>Records which live nodes read each value (used to fuse Conv + BatchNormalization and to detect unsupported optional outputs).</summary>
    private void IndexConsumers(List<OnnxNode> live)
    {
        foreach (var node in live)
        {
            foreach (var input in node.Inputs)
            {
                if (input == "")
                {
                    continue;
                }

                _usedNames.Add(input);
                if (!_consumers.TryGetValue(input, out var consumers))
                {
                    consumers = [];
                    _consumers[input] = consumers;
                }

                consumers.Add(node);
            }
        }
    }

    private void DeclareInput(OnnxValueInfo info)
    {
        if (_opts.ConstantInputs.TryGetValue(info.Name, out var pinned))
        {
            long[] dims = _opts.InputShapes.GetValueOrDefault(info.Name) ?? info.Shape!.Select(d => d.Value ?? 1).ToArray();
            var constant = OnnxType.IsFloat(info.ElemType)
                ? OnnxConst.FromFloats(info.ElemType, dims, pinned, info.Name)
                : OnnxConst.FromInts(info.ElemType, dims, pinned.Select(v => (long)v).ToArray(), info.Name);
            _env[info.Name] = TranslatedTensor.Const(constant);
            return;
        }

        var shape = InputShape(info);
        // Integer inputs of every width travel as i32 and float inputs of every precision as f32 (see docs/onnx-support.md, "Types").
        bool isInt = OnnxType.IsInteger(info.ElemType);
        if (!OnnxType.IsFloat(info.ElemType) && !isInt && info.ElemType != OnnxType.Bool)
        {
            throw new NotSupportedException($"Graph input '{info.Name}' has type {OnnxType.Name(info.ElemType)}; only float, integer and bool inputs are supported.");
        }

        bool image = !isInt && shape.Rank == 4 && shape[1].Fixed is 1 or 3 or 4;
        var role = image ? InputRole.Media : isInt ? InputRole.TokenIds : InputRole.Other;
        var value = _b.Input(info.Name, role, isInt ? DataType.I32 : DataType.F32, shape);
        var tensor = TranslatedTensor.Dyn(value);
        _env[info.Name] = tensor;
        if ((_opts.InputShapes.GetValueOrDefault(info.Name)?.Length ?? info.Shape?.Count) == 0)
        {
            tensor.Scalar = true;
        }
    }

    /// <summary>Translates one node, adding the node identity to errors that do not carry it yet.</summary>
    private void TranslateNode(OnnxNode node)
    {
        try
        {
            HandleNode(node);
        }
        catch (Exception e) when (e is not NotSupportedException && e is not InvalidDataException)
        {
            throw new InvalidDataException($"Translating {node.OpType} node '{node.Name}' ({string.Join(", ", node.Outputs)}): {e.Message}", e);
        }
        catch (NotSupportedException e) when (!e.Message.Contains("node '"))
        {
            throw new NotSupportedException($"{node.OpType} node '{node.Name}' ({string.Join(", ", node.Outputs)}): {e.Message}", e);
        }
    }

    private void DeclareOutput(OnnxValueInfo output)
    {
        if (!_env.TryGetValue(output.Name, out var tensor))
        {
            throw new InvalidDataException($"Graph output '{output.Name}' is never produced.");
        }

        if (tensor.C is not null)
        {
            throw new NotSupportedException($"Graph output '{output.Name}' is a compile-time constant; outputs must depend on an input.");
        }

        var value = OutputCopy(tensor.V!, output.Name);

        _b.Output(output.Name, value);
    }

    // ---- inputs ------------------------------------------------------------------------------------
    private Shape InputShape(OnnxValueInfo vi)
    {
        long[]? ov = _opts.InputShapes.GetValueOrDefault(vi.Name);
        int rank = ov?.Length ?? vi.Shape?.Count ?? throw new NotSupportedException($"Graph input '{vi.Name}' has no shape; supply one through OnnxImportOptions.InputShapes.");
        if (rank == 0)
        {
            return new Shape([(Dim)1L]);   // scalar inputs are carried as one-element vectors (numpy-style broadcasting makes them equivalent)
        }

        var dims = new Dim[rank];
        for (int i = 0; i < rank; i++)
        {
            long? fixedV = ov is not null ? (ov[i] > 0 ? ov[i] : null) : vi.Shape![i].IsKnown ? vi.Shape[i].Value : null;
            if (i == 0 && (_opts.ForceSymbolicBatch || fixedV is null))
            {
                dims[i] = Dim.Sym(BatchSymbol);
            }
            else if (fixedV is { } f)
            {
                dims[i] = f;
            }
            else
            {
                throw new NotSupportedException($"Graph input '{vi.Name}' has a dynamic dimension {i} ('{vi.Shape![i]}'); only the outermost (batch) dimension may be symbolic. Fix it via OnnxImportOptions.InputShapes.");
            }
        }
        return new Shape(dims);
    }

    // ---- scheduling / analysis -----------------------------------------------------------------------
    private static List<OnnxNode> Order(IReadOnlyList<OnnxNode> nodes, HashSet<string> initNames)
    {
        // ONNX requires topological order, but be tolerant: Kahn's algorithm keeping the original order stable.
        var produced = new HashSet<string>(initNames, StringComparer.Ordinal);
        var producers = new HashSet<string>(StringComparer.Ordinal);
        foreach (var n in nodes)
        {
            foreach (var o in n.Outputs)
            {
                producers.Add(o);
            }
        }

        var result = new List<OnnxNode>(nodes.Count);
        var pending = new List<OnnxNode>(nodes);
        bool ok = true;
        foreach (var n in nodes)
        {
            if (n.Inputs.Any(i => i != "" && producers.Contains(i) && !produced.Contains(i)))
            {
                ok = false;
                break;
            }
            foreach (var o in n.Outputs)
            {
                produced.Add(o);
            }
        }
        if (ok)
        {
            return pending;
        }

        produced = new HashSet<string>(initNames, StringComparer.Ordinal);
        while (pending.Count > 0)
        {
            int before = pending.Count;
            for (int k = 0; k < pending.Count; k++)
            {
                var n = pending[k];
                if (n.Inputs.Any(i => i != "" && producers.Contains(i) && !produced.Contains(i)))
                {
                    continue;
                }

                result.Add(n);
                foreach (var o in n.Outputs)
                {
                    produced.Add(o);
                }

                pending.RemoveAt(k--);
            }
            if (pending.Count == before)
            {
                throw new InvalidDataException("The ONNX graph contains a cycle or an undefined value.");
            }
        }
        return result;
    }

    private List<OnnxNode> Liveness(List<OnnxNode> order)
    {
        var producer = new Dictionary<string, OnnxNode>(StringComparer.Ordinal);
        foreach (var n in order)
        {
            foreach (var o in n.Outputs)
            {
                if (o != "")
                {
                    producer[o] = n;
                }
            }
        }

        var live = new HashSet<OnnxNode>();
        var stack = new Stack<string>(_graph.Outputs.Select(o => o.Name));
        var seen = new HashSet<string>(StringComparer.Ordinal);
        while (stack.Count > 0)
        {
            var name = stack.Pop();
            if (!seen.Add(name) || !producer.TryGetValue(name, out var n))
            {
                continue;
            }

            if (!live.Add(n))
            {
                continue;
            }

            foreach (var i in n.Inputs)
            {
                if (i != "")
                {
                    stack.Push(i);
                }
            }
        }
        return order.Where(live.Contains).ToList();
    }

    private void CheckSupport(List<OnnxNode> live)
    {
        var bad = live.Where(n => !IsSupportedNode(n)).Select(n => n.Domain is "" or "ai.onnx" ? n.OpType : $"{n.Domain}.{n.OpType}")
            .Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        if (bad.Count > 0)
        {
            throw new NotSupportedException($"The ONNX model uses {bad.Count} unsupported operator type(s): {string.Join(", ", bad)}.");
        }
    }

    private static bool IsSupportedNode(OnnxNode n)
        => n.Domain is "" or "ai.onnx" ? IsSupported(n.OpType) : n.Domain == "com.microsoft" && (s_contribOps.Contains(n.OpType) || s_quantContrib.Contains(n.OpType));

    // ---- group handling --------------------------------------------------------------------------------
    private void NextGroup()
    {
        _groupScope?.Dispose();
        _groupScope = _b.Group($"onnx.{_groupIndex++:D3}");
        _heavyInGroup = 0;
    }

    private void Heavy()
    {
        if (++_heavyInGroup >= Math.Max(1, _opts.GroupSize))
        {
            NextGroup();
        }
    }

    // ---- constants -> IR ---------------------------------------------------------------------------------
    internal const int MaxInlineConstElements = 4096;

    private Value AsValue(TranslatedTensor t, string? hint = null)
    {
        if (t.V is not null)
        {
            if (t.V.Type == DataType.F32)
            {
                return t.V;
            }

            return t.CachedValue ??= _b.Emit1("Cast", [t.V], Attributes.From(("type", "f32")));
        }
        return t.CachedValue ??= ConstValue(t.C!, hint);
    }

    private string UniqueWeightName(string baseName)
    {
        if (baseName == "")
        {
            baseName = $"const.{_anon++}";
        }

        if (!_weights.ContainsKey(baseName))
        {
            return baseName;
        }

        int i = 1;
        while (_weights.ContainsKey($"{baseName}#{i}"))
        {
            i++;
        }

        return $"{baseName}#{i}";
    }

    /// <summary>f32 IR value for a constant: a small graph Constant, or a (possibly zero-copy) Weight.</summary>
    private Value ConstValue(OnnxConst c, string? hint)
    {
        var dims = c.Rank == 0 ? [1L] : c.Dims;
        if (_constValues.TryGetValue((c, "f32"), out var cached))
        {
            return cached;
        }

        if (c.HasSymbols)
        {
            return _constValues[(c, "f32")] = SymbolicData(c, DataType.F32, hint);
        }

        var shape = new Shape(dims.Select(d => (Dim)d).ToArray());
        Value v;
        string name = c.Name != "" ? c.Name : hint ?? $"const.{_anon}";
        if (c.Count <= MaxInlineConstElements)
        {
            v = _b.Constant(name, DataType.F32, shape, c.F32Bytes());
        }
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
    private Value WeightValue(OnnxConst c, bool keepHalf, string? nameHint = null)
    {
        var shape = new Shape(c.Dims.Select(d => (Dim)d).ToArray());
        bool half = keepHalf && c.ElemType is OnnxType.Float16 or OnnxType.BFloat16;
        var key = (c, half ? "w:half" : "w:f32");
        if (_constValues.TryGetValue(key, out var cached))
        {
            return cached;
        }

        var wname = UniqueWeightName(c.Name != "" ? c.Name : nameHint ?? "");
        WeightInfo w;
        if ((c.ElemType == OnnxType.Float || half) && c.HasBackingSource)
        {
            w = new WeightInfo { Name = wname, Type = OnnxImporter.ToDataType(c.ElemType)!, Shape = shape, Source = c.Source! };
        }
        else if (c.ElemType == OnnxType.Float || half)
        {
            w = WeightInfo.Derived(wname, OnnxImporter.ToDataType(c.ElemType)!, shape, c.Bytes);
        }
        else
        {
            w = WeightInfo.Derived(wname, DataType.F32, shape, c.F32Bytes);
        }

        _weights[wname] = w;
        var v = _b.Weight(wname);
        _constValues[key] = v;
        return v;
    }

    // ---- node dispatch -------------------------------------------------------------------------------------
    private TranslatedTensor Get(string name)
        => _env.TryGetValue(name, out var t) ? t : throw new InvalidDataException($"ONNX value '{name}' is used before it is defined.");

    private void HandleNode(OnnxNode n)
    {
        var ins = new TranslatedTensor?[n.Inputs.Count];
        for (int i = 0; i < ins.Length; i++)
        {
            ins[i] = n.Inputs[i] == "" ? null : Get(n.Inputs[i]);
        }

        bool allConst = ins.All(t => t is null || t.IsConst);
        if (allConst && !BuildsSymbolicExtent(n, ins) && OnnxFold.ShouldFold(n, ins.Select(t => t?.C).ToList()))
        {
            var res = OnnxFold.Eval(n, ins.Select(t => t?.C).ToList());
            for (int i = 0; i < n.Outputs.Count && i < res.Count; i++)
            {
                if (n.Outputs[i] != "")
                {
                    var c = res[i];
                    _env[n.Outputs[i]] = TranslatedTensor.Const(c.Name == "" ? c.Reshaped(c.Dims, n.Outputs[i]) : c);
                }
            }

            return;
        }
        if (ins.Any(t => t is { TransposedNonZero: true }))
        {
            NonZeroConsumer(n, ins);
            return;
        }

        int leadingUnits = ins.Max(t => t?.LeadingUnits ?? 0);
        bool indicesOnly = n.OpType == "Gather" && ins[0]!.LeadingUnits == 0 && n.Int("axis", 0) == 0;   // [1, count] indices into axis 0
        if (leadingUnits > 0 && !indicesOnly && !s_leadingUnitAware.Contains(n.OpType) && !s_unary.ContainsKey(n.OpType) && !s_binary.ContainsKey(n.OpType))
        {
            throw new NotSupportedException($"{n.OpType} of a tensor with unit dims in front of its symbolic dim");
        }

        Dynamic(n, ins);
        if (leadingUnits > 0 && !s_leadingUnitSetters.Contains(n.OpType))
        {
            foreach (var o in n.Outputs)
            {
                if (o != "" && _env.TryGetValue(o, out var r) && r.V is { } v && Sym(v))
                {
                    r.LeadingUnits = leadingUnits;
                }
            }
        }
        if (ins.Any(t => t is { Scalar: true }) && s_scalarPreserving.Contains(n.OpType) && ins.All(t => t is null || t.Scalar || (t.IsConst && t.C!.Count == 1)))
        {
            foreach (var o in n.Outputs)
            {
                if (o != "" && _env.TryGetValue(o, out var r) && !r.IsConst)
                {
                    r.Scalar = true;
                }
            }
        }
    }

    // Ops that accept tensors with leading unit dims (see TranslatedTensor.LeadingUnits) besides unary and binary elementwise ops;
    // the setters compute the leading units of their outputs themselves.
    private static readonly HashSet<string> s_leadingUnitAware = ["Identity", "Cast", "Unsqueeze", "Squeeze", "Shape", "Clip", "NonMaxSuppression"];
    private static readonly HashSet<string> s_leadingUnitSetters = ["Unsqueeze", "Squeeze", "Shape", "NonMaxSuppression"];

    private static readonly HashSet<string> s_scalarPreserving =
        ["Cast", "Add", "Sub", "Mul", "Div", "Neg", "Abs", "Sqrt", "Exp", "Log", "Sin", "Cos", "Tanh", "Sigmoid", "Relu", "Identity", "Pow", "Floor", "Ceil", "Round"];

    private void SetDyn(OnnxNode n, int idx, Value v) => _env[n.Outputs[idx]] = TranslatedTensor.Dyn(v);

    private void CheckUnusedOutputs(OnnxNode n, int from)
    {
        for (int i = from; i < n.Outputs.Count; i++)
        {
            if (n.Outputs[i] != "" && (_usedNames.Contains(n.Outputs[i]) || _graphOutputs.Contains(n.Outputs[i])))
            {
                throw new NotSupportedException($"output #{i} ('{n.Outputs[i]}') is used but not supported.");
            }
        }
    }

    private static string Nm(OnnxNode n, int i = 0) => n.Outputs.Count > i && n.Outputs[i] != "" ? n.Outputs[i] : $"{n.OpType.ToLowerInvariant()}_{n.Name}";

    private static bool Sym(Value v) => v.Rank > 0 && v.Shape[0].IsSymbolic;
}
