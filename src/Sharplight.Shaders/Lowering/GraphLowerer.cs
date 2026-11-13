using System.Globalization;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed class LoweringOptions
{
    public ShaderTarget Target { get; init; } = ShaderTarget.SpirV;
    public int SubgroupSize { get; init; } = 32;
    /// <summary>Shared (workgroup) memory per workgroup in bytes.</summary>
    public long MaxSharedMemory { get; init; } = 32768;
    /// <summary>
    /// Effective device features (<see cref="DeviceFeatures"/>) the generated code may use. Every
    /// feature-specific kernel has a portable fallback used when its feature is absent.
    /// </summary>
    public IReadOnlySet<string> Features { get; init; } = new HashSet<string>();
    public bool Has(string feature) => Features.Contains(feature);
    public required CompileOptions Compile { get; init; }
    /// <summary>Model-level compilation cache for this device (see <see cref="CompileOptions.CacheFor"/>).</summary>
    public Runtime.ICompilationCache? Cache { get; init; }
}

/// <summary>
/// Lowers a graph segment into a backend-neutral <see cref="KernelProgram"/>:
/// resolves views to strided tensor references, fuses elementwise chains into generated kernels,
/// specializes library kernels for the concrete shapes/formats, plans activation memory with
/// liveness-based reuse, and records read/write ranges for barrier placement.
/// </summary>
public sealed partial class GraphLowerer
{
    readonly GraphSegment _seg;
    readonly SlangCompiler _compiler;
    readonly LoweringOptions _opt;
    readonly List<BufferDecl> _buffers = [];
    readonly List<KernelLaunch> _launches = [];
    readonly Dictionary<Value, TensorRef> _refs = [];
    readonly Dictionary<WeightInfo, BufferDecl> _weightBuffers = [];
    readonly Dictionary<StateDefinition, BufferDecl> _stateBuffers = [];
    readonly Dictionary<Value, TensorRef> _inputRefs = [];
    readonly Dictionary<Value, TensorRef> _outputRefs = [];
    readonly HashSet<Value> _segmentOutputs;
    readonly Dictionary<string, int> _scalarSlots = new(StringComparer.Ordinal);
    readonly BufferDecl _arena;
    readonly BufferDecl _runtime;
    readonly ActivationAllocator _alloc = new();
    readonly Dictionary<Value, Node> _absorbedNorms = [];
    readonly Dictionary<Value, Node> _absorbedRope = [];   // rope output → rope node computed inside its CacheWrite  // value → norm node computed inside its consumer
    readonly Dictionary<Value, Value> _root = [];        // view → materialized root value
    readonly Dictionary<Value, int> _lastUse = [];       // root → last unit index using it

    GraphLowerer(GraphSegment seg, SlangCompiler compiler, LoweringOptions opt)
    {
        _seg = seg;
        _compiler = compiler;
        _opt = opt;
        _segmentOutputs = [.. seg.Outputs];
        _runtime = NewBuffer(BufferKind.Runtime, "runtime", 256);
        _arena = NewBuffer(BufferKind.Activations, "activations", 0);
        var scalars = seg.Graph.RuntimeScalars.OrderBy(s => s == WellKnown.PastScalar ? "" : s, StringComparer.Ordinal).ToList();
        for (int i = 0; i < scalars.Count; i++) _scalarSlots[scalars[i]] = i;
    }

    public static KernelProgram Lower(GraphSegment segment, SlangCompiler compiler, LoweringOptions options)
        => new GraphLowerer(segment, compiler, options).Run();

    // ---- op support --------------------------------------------------------------------------
    static readonly HashSet<string> s_elementwise = [.. StandardOps.UnaryNames, .. StandardOps.BinaryNames, "Scale", "Cast", "Clip", "Where"];
    static readonly HashSet<string> s_supported =
        [.. s_elementwise, "Reshape", "Slice", "Permute", "Contiguous", "Concat", "Gather", "MatMul", "RmsNorm", "L2Norm",
         "Rope", "CacheWrite", "CachedAttention", "Attention", "CausalConv1d", "GatedDeltaRule", "SelectiveScan", "MatMulId", .. SpatialOpNames, .. GeneralOpNames];

    public static OpSupport Supports(Node node)
    {
        if (!s_supported.Contains(node.Op.Name)) return OpSupport.No($"no kernel for {node.Op.Name}");
        if (s_spatialOps.Contains(node.Op.Name)) return SupportsSpatial(node);
        if (s_generalOps.Contains(node.Op.Name)) return SupportsGeneral(node);
        if (node.Op.Name == "SelectiveScan") return SupportsSelectiveScan(node);
        if (node.Op.Name == "MatMulId") return SupportsMoe(node);
        if (node.Op.Name == "CachedAttention" && node.Attributes.Get("sinks", false) && node.Inputs[3].Type != DataType.F32) return OpSupport.No("attention sinks must be f32");
        if (node.Op.Name == "Attention" && (node.Inputs.Length != 3 || node.Attributes.Get("causal", false) || node.Inputs[1].Shape[1] != node.Inputs[2].Shape[1]
            || node.Inputs[0].Shape[0] != node.Inputs[1].Shape[0] || node.Inputs[0].Shape[2].Fixed % 8 != 0 || node.Inputs[1].Shape[2].Fixed % 8 != 0))
            return OpSupport.No("Attention: only bidirectional self-attention without mask over equally long q/k/v with head size % 8 == 0 is implemented");
        if (node.Op.Name == "MatMul" && node.Inputs[1].Type.IsQuantized && node.Inputs[1].Shape[-1] is { IsSymbolic: false } kd && kd.Fixed % 8 != 0)
            return OpSupport.No($"quantized MatMul K={kd.Fixed} is not a multiple of 8");
        if (node.Op.Name == "Rope" && node.Inputs.Length > 2 && (node.Inputs[2].Type != DataType.F32 || node.Inputs[2].Shape.Rank != 1))
            return OpSupport.No("Rope frequency factors must be a rank-1 f32 tensor");
        foreach (var v in node.Inputs)
        {
            if (v.Type.IsQuantized && node.Op.Name is not ("MatMul" or "Gather"))
                return OpSupport.No($"{node.Op.Name} cannot read quantized {v.Type}");
            if (v.Type.IsQuantized && WeightFormat(v.Type) is null)
                return OpSupport.No($"weight format {v.Type} not supported");
        }
        return OpSupport.Yes;
    }

    static int? WeightFormat(DataType t) =>
        t == DataType.F32 ? 0 : t == DataType.F16 ? 1 : t == DataType.BF16 ? 2 : t == DataType.Q8_0 ? 8 : t == ShaderTypes.Q8_0R ? 108 :
        t == DataType.Q4_0 ? 2001 : t == DataType.Q4_K ? 12 : t == DataType.Q5_K ? 13 : t == DataType.Q6_K ? 14 :
        t == DataType.Q1_0 ? 41 : t == DataType.Q2_0 ? 42 : t == DataType.Q2_0_G128 ? 142 : t == DataType.Tern1_G128 ? 143 : null;

    // ---- main --------------------------------------------------------------------------------
    KernelProgram Run()
    {
        var units = Schedule(FormUnits());
        ComputeLiveness(units);

        // Memory of dead values is only recycled once a later dependency level starts: reusing it within the
        // same level would add a write-after-read hazard (and a barrier) between otherwise independent launches.
        var deferred = new List<(Value Root, int Level)>();
        for (int u = 0; u < units.Count; u++)
        {
            var unit = units[u];
            deferred.RemoveAll(d => { if (d.Level >= unit.Level) return false; _alloc.Free(d.Root); return true; });
            if (unit.Dual is { } d) LowerDualMatMul(d.A, d.B, unit.Fused!, unit.FusedOutputs![0]);
            else if (unit.Fused is not null) LowerElementwise(unit.Fused, unit.FusedOutputs!, unit.NormTail, unit.NormHead);
            else LowerNode(unit.Node!);
            foreach (var (root, last) in _lastUse)
                if (last == u) deferred.Add((root, unit.Level));
        }

        // Segment outputs that are views (or inputs) need a dense copy in an output buffer.
        foreach (var o in _seg.Outputs)
            if (!_outputRefs.ContainsKey(o))
            {
                var src = Ref(o);
                var dst = Dense(NewBuffer(BufferKind.Output, "out:" + o.Name, 0), o);
                dst.Buffer.Bytes = dst.MaxBytes;
                EmitCopy(src, dst, $"copy {o.Name}");
                _outputRefs[o] = dst;
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

    // ---- units & fusion ----------------------------------------------------------------------
    sealed class Unit
    {
        public Node? Node;
        public List<Node>? Fused;
        public List<Value>? FusedOutputs;
        /// <summary>Row normalization (RmsNorm/L2Norm) of a group output, computed in the same kernel.</summary>
        public Node? NormTail;
        /// <summary>Row normalization whose output feeds this group, computed in the same kernel (e.g. gated RmsNorm).</summary>
        public Node? NormHead;
        /// <summary>Two matmuls (sharing x) whose outputs only feed this group: lowered as one dual matvec for small batches.</summary>
        public (Node A, Node B)? Dual;
        /// <summary>Nodes computed inside this unit's kernel (e.g. L2Norm of delta-rule q/k).</summary>
        public List<Node>? Absorbed;
        /// <summary>Dependency level (ASAP): launches of one level are mutually independent.</summary>
        public int Level;
        public bool IsView => Node is { Op.Category: OpCategory.View };
        public IEnumerable<Node> Nodes => Fused is null ? [.. Absorbed ?? [], Node!]
            : [.. Dual is { } d ? new[] { d.A, d.B } : [], .. NormHead is null ? [] : new[] { NormHead }, .. Fused, .. NormTail is null ? [] : new[] { NormTail }];
    }

    static bool IsFusable(Node n) =>
        s_elementwise.Contains(n.Op.Name) && n.Outputs[0].Type == DataType.F32 &&
        n.Inputs.All(i => !i.Type.IsQuantized && (i.Type == DataType.F32 || i.Type == DataType.F16 || i.Type == DataType.BF16));

    List<Unit> FormUnits()
    {
        var units = new List<Unit>();
        List<Node>? open = null;
        HashSet<Value>? openValues = null;

        void Close()
        {
            if (open is null) return;
            units.Add(new Unit { Fused = open });
            open = null;
            openValues = null;
        }

        foreach (var n in _seg.Nodes)
        {
            if (n.Op.Category == OpCategory.View)
            {
                if (openValues is not null && openValues.Contains(n.Inputs[0])) Close();
                units.Add(new Unit { Node = n });
                continue;
            }
            if (IsFusable(n))
            {
                bool joins = open is not null && n.Outputs[0].Shape.Equals(open[0].Outputs[0].Shape) && n.Inputs.Any(openValues!.Contains);
                if (!joins) { Close(); open = []; openValues = []; }
                open!.Add(n);
                openValues!.Add(n.Outputs[0]);
                continue;
            }
            Close();
            units.Add(new Unit { Node = n });
        }
        Close();

        // An elementwise group directly followed by a row norm of one of its outputs becomes one kernel
        // (e.g. residual Add + RmsNorm): saves a dispatch and a barrier.
        for (int i = 0; i + 1 < units.Count; i++)
        {
            if (units[i].Fused is not { } g || units[i + 1].Node is not { } nn || nn.Op.Name is not ("RmsNorm" or "L2Norm")) continue;
            var x = nn.Inputs[0];
            if (!g.Any(f => f.Output == x) || !x.Shape.Equals(g[0].Output.Shape) || x.Shape.Rank < 1 || x.Shape[-1].IsSymbolic) continue;
            if (nn.Inputs.Length > 1 && (nn.Inputs[1].Type != DataType.F32 || nn.Inputs[1].Shape.Rank != 1)) continue;
            units[i].NormTail = nn;
            units.RemoveAt(i + 1);
        }

        // A row norm whose output is consumed only by the next elementwise group (views of other values may sit in
        // between) is computed in that group's kernel.
        {
            var uses = new Dictionary<Value, List<Node>>();
            foreach (var n in _seg.Nodes) foreach (var i in n.Inputs) (uses.TryGetValue(i, out var l) ? l : uses[i] = []).Add(n);
            for (int i = 0; i < units.Count; i++)
            {
                if (units[i].Node is not { Op.Name: "RmsNorm" or "L2Norm" } nn) continue;
                int j = i + 1;
                while (j < units.Count && units[j].IsView && !units[j].Node!.Inputs.Contains(nn.Output)) j++;
                if (j >= units.Count || units[j].Fused is not { } g || units[j].NormTail is not null) continue;
                var y = nn.Output;
                if (_segmentOutputs.Contains(y) || !y.Shape.Equals(g[0].Output.Shape) || y.Shape.Rank < 1 || y.Shape[-1].IsSymbolic) continue;
                if (!uses.TryGetValue(y, out var us) || us.Any(c => !g.Contains(c))) continue;
                if (nn.Inputs.Length > 1 && (nn.Inputs[1].Type != DataType.F32 || nn.Inputs[1].Shape.Rank != 1)) continue;
                units[j].NormHead = nn;
                units.RemoveAt(i);
                i--;
            }
        }

        // Gate/up style matmul pairs feeding one elementwise group.
        if (!s_noDual)
        {
            var uses = new Dictionary<Value, List<Node>>();
            foreach (var n in _seg.Nodes) foreach (var i in n.Inputs) (uses.TryGetValue(i, out var l) ? l : uses[i] = []).Add(n);
            var unitOf = new Dictionary<Node, Unit>();
            foreach (var u in units) if (u.Node is not null) unitOf[u.Node] = u;
            var removed = new HashSet<Unit>();
            foreach (var u in units)
            {
                if (u.Fused is not { } g || u.NormHead is not null || u.NormTail is not null) continue;
                var inGroup = g.ToHashSet();
                var ext = g.SelectMany(n => n.Inputs).Where(v => !inGroup.Contains(v.Producer!)).Distinct().ToList();
                if (ext.Count != 2 || ext[0].Producer is not { Op.Name: "MatMul" } a || ext[1].Producer is not { Op.Name: "MatMul" } b) continue;
                if (a.Inputs[0] != b.Inputs[0] || a.Inputs[1].Type != b.Inputs[1].Type || !a.Inputs[1].Shape.Equals(b.Inputs[1].Shape)) continue;
                if (a.Inputs[1].Kind != ValueKind.Weight || b.Inputs[1].Kind != ValueKind.Weight || a.Inputs[1].Shape.Rank != 2) continue;
                if (a.Inputs[1].Shape[1] is not { IsSymbolic: false } kd || kd.Fixed % 8 != 0 || WeightFormat(a.Inputs[1].Type) is null) continue;
                if (!unitOf.TryGetValue(a, out var ua) || !unitOf.TryGetValue(b, out var ub)) continue;
                if (ext.Any(v => _segmentOutputs.Contains(v) || uses[v].Any(c => !inGroup.Contains(c)))) continue;
                var outs = g.Select(n => n.Output).Where(v => _segmentOutputs.Contains(v) || (uses.TryGetValue(v, out var cs) && cs.Any(c => !inGroup.Contains(c)))).ToList();
                if (outs.Count != 1 || !outs[0].Shape.Equals(a.Output.Shape) || g[0].Output.Shape.Equals(a.Output.Shape) is false) continue;
                u.Dual = (a, b);
                removed.Add(ua); removed.Add(ub);
            }
            units.RemoveAll(removed.Contains);
        }

        // L2-normalized q/k consumed only by a delta rule are normalized inside the delta-rule kernel.
        var users = new Dictionary<Value, int>();
        foreach (var n in _seg.Nodes) foreach (var i in n.Inputs) users[i] = users.GetValueOrDefault(i) + 1;
        var absorbed = new HashSet<Node>();
        foreach (var u in units)
        {
            if (u.Node is not { Op.Name: "GatedDeltaRule" } dn) continue;
            for (int idx = 0; idx < 2; idx++)
            {
                var v = dn.Inputs[idx];
                if (v.Producer is not { Op.Name: "L2Norm" } ln || users.GetValueOrDefault(v) != 1 || _segmentOutputs.Contains(v) || ln.Inputs.Length != 1) continue;
                (u.Absorbed ??= []).Add(ln);
                absorbed.Add(ln);
                _absorbedNorms[v] = ln;
            }
        }
        // Rope of an RmsNorm-ed row computes the norm itself; a rope whose only consumer is a cache write stores into the cache.
        var unitOfNode = units.Where(u => u.Node is not null).ToDictionary(u => u.Node!);
        foreach (var u in units)
        {
            if (u.Node is { Op.Name: "Rope" } r && r.Inputs[0] is var rv && rv.Producer is { Op.Name: "RmsNorm" } rn
                && users.GetValueOrDefault(rv) == 1 && !_segmentOutputs.Contains(rv) && rn.Inputs.Length == 2
                && rn.Inputs[1].Type == DataType.F32 && rn.Inputs[1].Shape.Rank == 1 && rv.Shape.Rank == 3 && unitOfNode.ContainsKey(rn))
            {
                (u.Absorbed ??= []).Add(rn);
                absorbed.Add(rn);
                _absorbedNorms[rv] = rn;
            }
        }
        foreach (var u in units)
        {
            if (u.Node is { Op.Name: "CacheWrite" } cw && cw.Inputs[1] is var cv && cv.Producer is { Op.Name: "Rope" } r
                && users.GetValueOrDefault(cv) == 1 && !_segmentOutputs.Contains(cv) && unitOfNode.TryGetValue(r, out var ru)
                && cw.Inputs[0].Shape.Rank == 3 && cv.Shape.Rank == 3 && cw.Inputs[0].Shape[1].Equals(cv.Shape[1]) && cw.Inputs[0].Shape[2].Equals(cv.Shape[2]))
            {
                u.Absorbed = [.. ru.Absorbed ?? [], r, .. u.Absorbed ?? []];
                absorbed.Add(r);
                _absorbedRope[cv] = r;
            }
        }
        units.RemoveAll(u => u.Node is not null && absorbed.Contains(u.Node));

        // Views were emitted as separate units in between; a fused group must be emitted before any
        // later unit consumes its outputs, which holds because units keep graph order.
        // Determine materialized outputs of each fused group.
        var consumers = new Dictionary<Value, List<Node>>();
        foreach (var n in _seg.Nodes)
            foreach (var i in n.Inputs)
                (consumers.TryGetValue(i, out var l) ? l : consumers[i] = []).Add(n);
        foreach (var u in units.Where(u => u.Fused is not null))
        {
            var inGroup = u.Nodes.ToHashSet();
            u.FusedOutputs = [.. u.Fused!.Select(n => n.Outputs[0]).Where(v =>
                _segmentOutputs.Contains(v) || (consumers.TryGetValue(v, out var cs) && cs.Any(c => !inGroup.Contains(c))))];
        }
        return units;
    }

    static readonly bool s_noDual = Environment.GetEnvironmentVariable("SHARPLIGHT_NO_DUAL") == "1";
    static readonly bool s_noSchedule = Environment.GetEnvironmentVariable("SHARPLIGHT_NO_SCHEDULE") == "1";

    /// <summary>
    /// Orders units by dependency level (stable within a level) so independent launches are recorded next to
    /// each other and share barriers. Units touching the same state keep their relative order.
    /// </summary>
    static List<Unit> Schedule(List<Unit> units)
    {
        var producer = new Dictionary<Value, int>();
        for (int i = 0; i < units.Count; i++)
            foreach (var n in units[i].Nodes)
                foreach (var o in n.Outputs) producer[o] = i;
        var lastState = new Dictionary<object, int>();
        var finish = new int[units.Count];
        for (int i = 0; i < units.Count; i++)
        {
            int lv = 0;
            var states = new List<object>();
            foreach (var n in units[i].Nodes)
                foreach (var v in n.Inputs.Concat(n.Writes))
                {
                    if (producer.TryGetValue(v, out int p) && p != i) lv = Math.Max(lv, finish[p]);
                    if (v.Kind == ValueKind.State)
                    {
                        if (lastState.TryGetValue(v.State!, out int q) && q != i) lv = Math.Max(lv, finish[q]);
                        states.Add(v.State!);
                    }
                }
            foreach (var st in states) lastState[st] = i;
            units[i].Level = lv;
            finish[i] = lv + (units[i].IsView ? 0 : 1);
        }
        if (s_noSchedule) return units;
        return [.. units.Select((u, i) => (u, i)).OrderBy(x => x.u.Level).ThenBy(x => x.i).Select(x => x.u)];
    }

    void ComputeLiveness(List<Unit> units)
    {
        // Root of each value: views map to the root of their input; everything else is its own root.
        foreach (var u in units)
            foreach (var n in u.Nodes)
                foreach (var o in n.Outputs)
                    _root[o] = n.Op.Category == OpCategory.View ? RootOf(n.Inputs[0]) : o;

        for (int u = 0; u < units.Count; u++)
            foreach (var n in units[u].Nodes)
            {
                foreach (var o in n.Outputs) _lastUse.TryAdd(RootOf(o), u);
                foreach (var i in n.Inputs) _lastUse[RootOf(i)] = u;
            }
        foreach (var o in _seg.Outputs) _lastUse[RootOf(o)] = int.MaxValue;
    }

    Value RootOf(Value v) => _root.TryGetValue(v, out var r) ? r : v;

    // ---- buffers & refs ----------------------------------------------------------------------
    BufferDecl NewBuffer(BufferKind kind, string name, long bytes, WeightInfo? w = null, StateDefinition? s = null, byte[]? init = null)
    {
        var b = new BufferDecl { Id = _buffers.Count, Kind = kind, Name = name, Bytes = bytes, Weight = w, State = s, InitialData = init };
        _buffers.Add(b);
        return b;
    }

    long Bound(Dim d) => d.IsSymbolic ? _opt.Compile.Bound(d.Symbol!) : d.Fixed;

    long[] BoundShape(Shape s) => [.. s.Dims.Select(Bound)];

    TensorRef Dense(BufferDecl buf, Value v, long byteOffset = 0)
    {
        var shape = BoundShape(v.Shape);
        return new TensorRef(buf, byteOffset, v.Type, shape, TensorRef.DenseStrides(shape), v.Shape.Symbol);
    }

    /// <summary>Returns the tensor reference of a value, materializing leaves (weights, states, inputs) on demand.</summary>
    static readonly bool s_repack = Environment.GetEnvironmentVariable("SHARPLIGHT_NO_REPACK") != "1";

    TensorRef Ref(Value v)
    {
        if (_refs.TryGetValue(v, out var r)) return r;
        switch (v.Kind)
        {
            case ValueKind.Weight:
                var w = v.Weight!;
                if (!_weightBuffers.TryGetValue(w, out var wb))
                {
                    _weightBuffers[w] = wb = NewBuffer(BufferKind.Weight, w.Name, w.ByteSize, w: w);
                    // Device-friendly layout: aligned int8 quants for wide loads (same byte size).
                    if (s_repack && w.Type == DataType.Q8_0 && w.Shape.Rank >= 1 && !w.Shape[-1].IsSymbolic && w.Shape[-1].Fixed % 128 == 0)
                    {
                        wb.Repack = WeightRepack.Q8_0Split;
                        wb.RepackRowElements = w.Shape[-1].Fixed;
                    }
                }
                r = Dense(wb, v);
                if (wb.Repack == WeightRepack.Q8_0Split) r = r with { Type = ShaderTypes.Q8_0R };
                break;
            case ValueKind.State:
                var s = v.State!;
                if (!_stateBuffers.TryGetValue(s, out var sb))
                    _stateBuffers[s] = sb = NewBuffer(BufferKind.State, s.Name, PlacementPlanner.StateBytes(s, _opt.Compile), s: s);
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
    TensorRef Produce(Value v)
    {
        if (_refs.TryGetValue(v, out var existing)) return existing; // fused launch + its unfused fallback launches
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

    static BufferRange Range(TensorRef r) => new(r.Buffer, r.ByteOffset, r.ByteOffset + r.MaxBytes);

    static string Num(double v) => v.ToString("R", CultureInfo.InvariantCulture);
    static string F(float v)
    {
        if (float.IsPositiveInfinity(v)) return "(1.0/0.0)";
        var s = v.ToString("R", CultureInfo.InvariantCulture);
        if (!s.Contains('.') && !s.Contains('E') && !s.Contains('e')) s += ".0";
        return s;
    }

    KernelArg TokenArg(TensorRef r, long multiplier = 1) =>
        r.Symbol is null ? new UIntArg((uint)(r.Shape[0] * multiplier)) : new SymbolArg(r.Symbol, multiplier);

    static long Rows(Bindings b, TensorRef r) => r.Symbol is null ? r.Shape[0] : b[r.Symbol];

    Dictionary<string, string> Macros(params (string, object)[] items)
    {
        var d = new Dictionary<string, string>(StringComparer.Ordinal) { ["SG"] = _opt.SubgroupSize.ToString(CultureInfo.InvariantCulture) };
        foreach (var (k, v) in items)
            d[k] = v switch
            {
                bool b => b ? "1" : "0",
                float f => F(f),
                double x => Num(x),
                _ => Convert.ToString(v, CultureInfo.InvariantCulture)!,
            };
        return d;
    }

    CompiledKernel Lib(string file, string entry, Dictionary<string, string> macros)
        => _compiler.CompileLibrary(file, entry, _opt.Target, macros, _opt.Cache);

    /// <summary>Extra condition for launches emitted while set (conditional fallback launches).</summary>
    Func<Bindings, bool>? _launchCondition;

    void Emit(string name, IEnumerable<Node> nodes, IReadOnlyList<KernelVariant> variants, IReadOnlyList<KernelArg> args,
        IEnumerable<TensorRef> reads, IEnumerable<TensorRef> writes)
    {
        if (_launchCondition is { } cond)
            variants = [.. variants.Select(v => v with { When = b => cond(b) && v.When(b) })];
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

    static KernelVariant Always(CompiledKernel k, Func<Bindings, Grid> grid) => new(_ => true, k, grid);
}

/// <summary>First-fit allocator for the activation arena (offsets aligned to 256 bytes).</summary>
sealed class ActivationAllocator
{
    readonly SortedList<long, long> _free = [];   // offset → size
    readonly Dictionary<object, (long Off, long Size)> _live = [];
    long _top, _high;
    public long HighWater => _high;

    public long Allocate(object v, long bytes)
    {
        bytes = (bytes + 255) & ~255L;
        foreach (var (off, size) in _free)
        {
            if (size < bytes) continue;
            _free.Remove(off);
            if (size > bytes) _free.Add(off + bytes, size - bytes);
            _live[v] = (off, bytes);
            return off;
        }
        long o = _top;
        _top += bytes;
        _high = Math.Max(_high, _top);
        _live[v] = (o, bytes);
        return o;
    }

    public void Free(object v)
    {
        if (!_live.Remove(v, out var a)) return;
        _free.Add(a.Off, a.Size);
        // coalesce neighbours
        bool merged = true;
        while (merged)
        {
            merged = false;
            for (int i = 0; i + 1 < _free.Count; i++)
            {
                if (_free.Keys[i] + _free.Values[i] == _free.Keys[i + 1])
                {
                    long off = _free.Keys[i], size = _free.Values[i] + _free.Values[i + 1];
                    _free.RemoveAt(i + 1);
                    _free[off] = size;
                    merged = true;
                    break;
                }
            }
        }
        // shrink the top
        if (_free.Count > 0 && _free.Keys[^1] + _free.Values[^1] == _top)
        {
            _top = _free.Keys[^1];
            _free.RemoveAt(_free.Count - 1);
        }
    }
}
