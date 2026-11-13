using System.Globalization;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;


/// <summary>
/// Lowers a graph segment into a backend-neutral <see cref="KernelProgram"/>:
/// resolves views to strided tensor references, fuses elementwise chains into generated kernels,
/// specializes library kernels for the concrete shapes/formats, plans activation memory with
/// liveness-based reuse, and records read/write ranges for barrier placement.
/// </summary>
public sealed partial class GraphLowerer
{
    private readonly GraphSegment _seg;
    private readonly SlangCompiler _compiler;
    private readonly LoweringOptions _opt;
    private readonly List<BufferDecl> _buffers = [];
    private readonly List<KernelLaunch> _launches = [];
    private readonly Dictionary<Value, TensorRef> _refs = [];
    private readonly Dictionary<WeightInfo, BufferDecl> _weightBuffers = [];
    private readonly Dictionary<StateDefinition, BufferDecl> _stateBuffers = [];
    private readonly Dictionary<Value, TensorRef> _inputRefs = [];
    private readonly Dictionary<Value, TensorRef> _outputRefs = [];
    private readonly HashSet<Value> _segmentOutputs;
    private readonly Dictionary<string, int> _scalarSlots = new(StringComparer.Ordinal);
    private readonly BufferDecl _arena;
    private readonly BufferDecl _runtime;
    private readonly ActivationAllocator _alloc = new();
    private readonly Dictionary<Value, Node> _absorbedNorms = [];
    private readonly Dictionary<Value, Node> _absorbedRope = [];   // rope output → rope node computed inside its CacheWrite  // value → norm node computed inside its consumer
    private readonly Dictionary<Value, Value> _root = [];        // view → materialized root value
    private readonly Dictionary<Value, int> _lastUse = [];       // root → last unit index using it

    private GraphLowerer(GraphSegment seg, SlangCompiler compiler, LoweringOptions opt)
    {
        _seg = seg;
        _compiler = compiler;
        _opt = opt;
        _segmentOutputs = [.. seg.Outputs];
        _runtime = NewBuffer(BufferKind.Runtime, "runtime", 256);
        _arena = NewBuffer(BufferKind.Activations, "activations", 0);
        var scalars = seg.Graph.RuntimeScalars.OrderBy(s => s == WellKnown.PastScalar ? "" : s, StringComparer.Ordinal).ToList();
        for (int i = 0; i < scalars.Count; i++)
        {
            _scalarSlots[scalars[i]] = i;
        }
    }

    public static KernelProgram Lower(GraphSegment segment, SlangCompiler compiler, LoweringOptions options)
        => new GraphLowerer(segment, compiler, options).Run();

    // ---- op support --------------------------------------------------------------------------
    private static readonly HashSet<string> s_elementwise = [.. StandardOps.UnaryNames, .. StandardOps.BinaryNames, "Scale", "Cast", "Clip", "Where"];
    private static readonly HashSet<string> s_supported =
        [.. s_elementwise, "Reshape", "Slice", "Permute", "Contiguous", "Concat", "Gather", "MatMul", "RmsNorm", "L2Norm",
         "Rope", "CacheWrite", "CachedAttention", "Attention", "CausalConv1d", "GatedDeltaRule", "SelectiveScan", "MatMulId", "SequencePool", .. SpatialOpNames, .. GeneralOpNames];

    public static OpSupport Supports(Node node)
    {
        if (!s_supported.Contains(node.Op.Name))
        {
            return OpSupport.No($"no kernel for {node.Op.Name}");
        }

        if (s_spatialOps.Contains(node.Op.Name))
        {
            return SupportsSpatial(node);
        }

        if (s_generalOps.Contains(node.Op.Name))
        {
            return SupportsGeneral(node);
        }

        if (node.Op.Name == "SelectiveScan")
        {
            return SupportsSelectiveScan(node);
        }

        if (node.Op.Name == "MatMulId")
        {
            return SupportsMoe(node);
        }

        if (node.Op.Name == "CachedAttention" && node.Attributes.Get("sinks", false) && node.Inputs[3].Type != DataType.F32)
        {
            return OpSupport.No("attention sinks must be f32");
        }

        if (node.Op.Name == "Attention")
        {
            var ins = node.Inputs;
            bool spans = ins.Length == 4 && ins[3].Type == DataType.I32 && ins[3].Rank == 2 && ins[3].Shape[1] == 2;
            bool causal = node.Attributes.Get("causal", false);
            if (ins.Length != 3 && !spans)
            {
                return OpSupport.No("Attention: an f32 mask input is not implemented (i32 spans are)");
            }

            if (ins[1].Shape[1] != ins[2].Shape[1] || ins[0].Shape[2].Fixed % 8 != 0 || ins[1].Shape[2].Fixed % 8 != 0 || ins[2].Shape[2].Fixed % 8 != 0)
            {
                return OpSupport.No("Attention: k/v head counts must match and head sizes be multiples of 8");
            }

            if (ins[0].Shape[1].Fixed % ins[1].Shape[1].Fixed != 0)
            {
                return OpSupport.No("Attention: query heads must be a multiple of key heads");
            }

            if (!spans && (causal || ins[0].Shape[0] != ins[1].Shape[0]))
            {
                return OpSupport.No("Attention: causal or cross attention needs i32 spans");
            }

            if (spans && causal && ins[0].Shape[0] != ins[1].Shape[0])
            {
                return OpSupport.No("Attention: causal attention must be self-attention");
            }
        }
        if (node.Op.Name == "SequencePool" && (node.Inputs[1].Type != DataType.I32 || node.Inputs[0].Type.IsQuantized
            || node.Attributes.Get("mode", "mean") is not ("mean" or "sum" or "max" or "first" or "last")))
        {
            return OpSupport.No("SequencePool: i32 spans and mode mean | sum | max | first | last");
        }

        if (node.Op.Name == "MatMul" && node.Inputs[1].Type.IsQuantized && node.Inputs[1].Shape[-1] is { IsSymbolic: false } kd && kd.Fixed % 8 != 0)
        {
            return OpSupport.No($"quantized MatMul K={kd.Fixed} is not a multiple of 8");
        }

        if (node.Op.Name == "Rope" && node.Inputs.Length > 2 && (node.Inputs[2].Type != DataType.F32 || node.Inputs[2].Shape.Rank != 1))
        {
            return OpSupport.No("Rope frequency factors must be a rank-1 f32 tensor");
        }

        foreach (var v in node.Inputs)
        {
            if (v.Type.IsQuantized && node.Op.Name is not ("MatMul" or "Gather"))
            {
                return OpSupport.No($"{node.Op.Name} cannot read quantized {v.Type}");
            }

            if (v.Type.IsQuantized && WeightFormat(v.Type) is null)
            {
                return OpSupport.No($"weight format {v.Type} not supported");
            }
        }
        return OpSupport.Yes;
    }

    private static int? WeightFormat(DataType t) =>
        t == DataType.F32 ? 0 : t == DataType.F16 ? 1 : t == DataType.BF16 ? 2 : t == DataType.Q8_0 ? 8 : t == ShaderTypes.Q8_0R ? 108 :
        t == DataType.Q4_0 ? 2001 : t == DataType.Q4_K ? 12 : t == DataType.Q5_K ? 13 : t == DataType.Q6_K ? 14 :
        t == DataType.Q4_1 ? 3 : t == DataType.Q5_0 ? 6 : t == DataType.Q5_1 ? 7 : t == DataType.IQ4_NL ? 20 : t == DataType.IQ4_XS ? 23 : t == DataType.Q1_0 ? 41 : t == DataType.Q2_0 ? 42 : t == DataType.Q2_0_G128 ? 142 : t == DataType.Tern1_G128 ? 143 : null;

    // ---- main --------------------------------------------------------------------------------
    private KernelProgram Run()
    {
        var units = Schedule(FormUnits());
        ComputeLiveness(units);

        // Memory of dead values is only recycled once a later dependency level starts: reusing it within the
        // same level would add a write-after-read hazard (and a barrier) between otherwise independent launches.
        var deferred = new List<(Value Root, int Level)>();
        for (int u = 0; u < units.Count; u++)
        {
            var unit = units[u];
            deferred.RemoveAll(d => { if (d.Level >= unit.Level) { return false; } _alloc.Free(d.Root); return true; });
            if (unit.Dual is { } d)
            {
                LowerDualMatMul(d.A, d.B, unit.Fused!, unit.FusedOutputs![0]);
            }
            else if (unit.Fused is not null)
            {
                LowerElementwise(unit.Fused, unit.FusedOutputs!, unit.NormTail, unit.NormHead);
            }
            else
            {
                LowerNode(unit.Node!);
            }

            foreach (var (root, last) in _lastUse)
            {
                if (last == u)
                {
                    deferred.Add((root, unit.Level));
                }
            }
        }

        // Segment outputs that are views (or inputs) need a dense copy in an output buffer.
        foreach (var o in _seg.Outputs)
        {
            if (!_outputRefs.ContainsKey(o))
            {
                var src = Ref(o);
                var dst = Dense(NewBuffer(BufferKind.Output, "out:" + o.Name, 0), o);
                dst.Buffer.Bytes = dst.MaxBytes;
                EmitCopy(src, dst, $"copy {o.Name}");
                _outputRefs[o] = dst;
            }
        }

        _arena.Bytes = Math.Max(256, _alloc.HighWater);
        return new KernelProgram
        {
            Buffers = _buffers,
            Launches = _launches,
            ResetLaunches = BuildResetLaunches(),
            InputRefs = _inputRefs,
            OutputRefs = _outputRefs,
            RuntimeBuffer = _runtime,
            ScalarSlots = _scalarSlots,
            Symbols = [.. _seg.Nodes.SelectMany(n => n.Outputs.Concat(n.Inputs)).Select(v => v.Shape.Symbol).Where(s => s is not null && s != WellKnown.ContextSymbol).Distinct()!],
        };
    }

    private Value RootOf(Value v) => _root.TryGetValue(v, out var r) ? r : v;

    // ---- buffers & refs ----------------------------------------------------------------------
    private BufferDecl NewBuffer(BufferKind kind, string name, long bytes, WeightInfo? w = null, StateDefinition? s = null, byte[]? init = null)
    {
        var b = new BufferDecl { Id = _buffers.Count, Kind = kind, Name = name, Bytes = bytes, Weight = w, State = s, InitialData = init };
        _buffers.Add(b);
        return b;
    }

    private long Bound(Dim d) => d.IsSymbolic ? _opt.Compile.Bound(d.Symbol!) : d.Fixed;

    private long[] BoundShape(Shape s) => [.. s.Dims.Select(Bound)];

    private TensorRef Dense(BufferDecl buf, Value v, long byteOffset = 0)
    {
        var shape = BoundShape(v.Shape);
        return new TensorRef(buf, byteOffset, v.Type, shape, TensorRef.DenseStrides(shape), v.Shape.Symbol);
    }

    /// <summary>Returns the tensor reference of a value, materializing leaves (weights, states, inputs) on demand.</summary>

    private TensorRef Ref(Value v)
    {
        if (_refs.TryGetValue(v, out var r))
        {
            return r;
        }

        switch (v.Kind)
        {
            case ValueKind.Weight:
                var w = v.Weight!;
                if (!_weightBuffers.TryGetValue(w, out var wb))
                {
                    _weightBuffers[w] = wb = NewBuffer(BufferKind.Weight, w.Name, w.ByteSize, w: w);
                    // Device-friendly layout: aligned int8 quants for wide loads (same byte size).
                    if (!DiagnosticSwitches.NoWeightRepack && w.Type == DataType.Q8_0 && w.Shape.Rank >= 1 && !w.Shape[-1].IsSymbolic && w.Shape[-1].Fixed % 128 == 0)
                    {
                        wb.Repack = WeightRepack.Q8_0Split;
                        wb.RepackRowElements = w.Shape[-1].Fixed;
                    }
                }
                r = Dense(wb, v);
                if (wb.Repack == WeightRepack.Q8_0Split)
                {
                    r = r with { Type = ShaderTypes.Q8_0R };
                }

                break;
            case ValueKind.State:
                var s = v.State!;
                if (!_stateBuffers.TryGetValue(s, out var sb))
                {
                    _stateBuffers[s] = sb = NewBuffer(BufferKind.State, s.Name, PlacementPlanner.StateBytes(s, _opt.Compile), s: s);
                }

                r = Dense(sb, v);
                break;
            case ValueKind.Constant:
                r = Dense(NewBuffer(BufferKind.Weight, "const:" + v.Name, v.ConstantData!.Length, init: v.ConstantData), v);
                break;
            default:
                // Graph input, or an activation produced by a previous segment.
                var ib = NewBuffer(BufferKind.Input, "in:" + v.Name, 0);
                r = Dense(ib, v);
                ib.Bytes = Math.Max(16, r.MaxBytes);
                _inputRefs[v] = r;
                break;
        }
        _refs[v] = r;
        return r;
    }

    /// <summary>Allocates a dense result tensor for a node output (activation arena or output buffer).</summary>
    private TensorRef Produce(Value v)
    {
        if (_refs.TryGetValue(v, out var existing))
        {
            return existing; // fused launch + its unfused fallback launches
        }

        TensorRef r;
        if (_segmentOutputs.Contains(v))
        {
            var ob = NewBuffer(BufferKind.Output, "out:" + v.Name, 0);
            r = Dense(ob, v);
            ob.Bytes = Math.Max(16, r.MaxBytes);
            _outputRefs[v] = r;
        }
        else
        {
            var shape = BoundShape(v.Shape);
            long bytes = v.Type.ByteSize(shape.Aggregate(1L, (a, b) => a * b));
            long off = _alloc.Allocate(v, Math.Max(bytes, 16));
            r = new TensorRef(_arena, off, v.Type, shape, TensorRef.DenseStrides(shape), v.Shape.Symbol);
        }
        _refs[v] = r;
        return r;
    }

    private static PointerArg Pointer(TensorRef r) => new(r.Buffer, r.ByteOffset);

    private static BufferRange Range(TensorRef r) => new(r.Buffer, r.ByteOffset, r.ByteOffset + r.MaxBytes);

    private static string Num(double v) => v.ToString("R", CultureInfo.InvariantCulture);
    internal static string F(float v)
    {
        if (float.IsPositiveInfinity(v))
        {
            return "(1.0/0.0)";
        }

        var s = v.ToString("R", CultureInfo.InvariantCulture);
        if (!s.Contains('.') && !s.Contains('E') && !s.Contains('e'))
        {
            s += ".0";
        }

        return s;
    }

    private KernelArg TokenArg(TensorRef r, long multiplier = 1) =>
        r.Symbol is null ? new UIntArg((uint)(r.Shape[0] * multiplier)) : new SymbolArg(r.Symbol, multiplier);

    internal static long Rows(Bindings b, TensorRef r) => r.Symbol is null ? r.Shape[0] : b[r.Symbol];

    private Dictionary<string, string> Macros(params (string, object)[] items)
    {
        var d = new Dictionary<string, string>(StringComparer.Ordinal) { ["SG"] = _opt.SubgroupSize.ToString(CultureInfo.InvariantCulture) };
        foreach (var (k, v) in items)
        {
            d[k] = v switch
            {
                bool b => b ? "1" : "0",
                float f => F(f),
                double x => Num(x),
                _ => Convert.ToString(v, CultureInfo.InvariantCulture)!,
            };
        }

        return d;
    }

    private CompiledKernel Lib(string file, string entry, Dictionary<string, string> macros)
        => _compiler.CompileLibrary(file, entry, _opt.Target, macros, _opt.Cache);

    /// <summary>Extra condition for launches emitted while set (conditional fallback launches).</summary>
    private Func<Bindings, bool>? _launchCondition;

    private void Emit(string name, IEnumerable<Node> nodes, IReadOnlyList<KernelVariant> variants, IReadOnlyList<KernelArg> args,
        IEnumerable<TensorRef> reads, IEnumerable<TensorRef> writes)
    {
        if (_launchCondition is { } cond)
        {
            variants = [.. variants.Select(v => v with { When = b => cond(b) && v.When(b) })];
        }

        _launches.Add(new KernelLaunch
        {
            Name = name,
            Variants = variants,
            Args = args,
            Reads = [.. reads.Select(Range)],
            Writes = [.. writes.Select(Range)],
            Nodes = [.. nodes],
        });
    }

    private static KernelVariant Always(CompiledKernel k, Func<Bindings, Grid> grid) => new(_ => true, k, grid);

    // ---- reset -------------------------------------------------------------------------------
    private List<KernelLaunch> BuildResetLaunches()
    {
        var list = new List<KernelLaunch>();
        if (_stateBuffers.Count == 0)
        {
            return list;
        }

        var k = Lib("data.slang", "fill", Macros());
        foreach (var b in _stateBuffers.Values)
        {
            uint count = (uint)(b.Bytes / 4);
            uint groups = Math.Min(65535u, Grid.Div(count, 256));
            list.Add(new KernelLaunch
            {
                Name = "reset " + b.Name,
                Variants = [Always(k, _ => new Grid(groups))],
                Args = [new PointerArg(b, 0), new UIntArg(count), new UIntArg(0), new UIntArg(groups * 256)],
                Reads = [],
                Writes = [new BufferRange(b, 0, b.Bytes)],
            });
        }
        return list;
    }
}
