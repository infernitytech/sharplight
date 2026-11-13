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

    /// <summary>Norm output → norm node computed inside its consumer's kernel (delta rule, rope).</summary>
    private readonly Dictionary<Value, Node> _absorbedNorms = [];

    /// <summary>Outputs of row gathers read in place by their fused consumer group (see <c>FoldRowGathers</c>).</summary>
    private readonly Dictionary<Value, Node> _foldedGathers = [];

    /// <summary>Rope output → rope node computed inside its CacheWrite.</summary>
    private readonly Dictionary<Value, Node> _absorbedRope = [];

    /// <summary>View → materialized root value.</summary>
    private readonly Dictionary<Value, Value> _root = [];

    /// <summary>Root → index of the last unit using it.</summary>
    private readonly Dictionary<Value, int> _lastUse = [];

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

    /// <summary>Lowers a graph segment into kernel launches with their buffers and hazard information.</summary>
    /// <param name="segment">The segment to lower; every node must satisfy <see cref="Supports"/>.</param>
    /// <param name="compiler">Compiler for the library and generated kernels (thread-safe).</param>
    /// <param name="options">Target, device limits and features the generated code may rely on.</param>
    /// <returns>The backend-neutral program; the lowerer is single-use and not thread-safe, but separate calls are independent.</returns>
    /// <exception cref="NotSupportedException">A node has no lowering for its shapes or types.</exception>
    /// <exception cref="KernelCompilationException">A kernel failed to compile.</exception>
    public static KernelProgram Lower(GraphSegment segment, SlangCompiler compiler, LoweringOptions options)
        => new GraphLowerer(segment, compiler, options).Run();

    // ---- main --------------------------------------------------------------------------------
    private KernelProgram Run()
    {
        var units = Schedule(FormUnits());
        ComputeLiveness(units);
        LowerUnits(units);
        CopyNonMaterializedOutputs();

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
            Symbols = CollectSymbols(),
        };
    }

    private void LowerUnits(List<LoweringUnit> units)
    {
        // Memory of dead values is only recycled once a later dependency level starts: reusing it within the
        // same level would add a write-after-read hazard (and a barrier) between otherwise independent launches.
        var deferred = new List<(Value Root, int Level)>();
        for (int index = 0; index < units.Count; index++)
        {
            var unit = units[index];
            FreeDeadValues(deferred, unit.Level);
            LowerUnit(unit);

            foreach (var (root, lastUse) in _lastUse)
            {
                if (lastUse == index)
                {
                    deferred.Add((root, unit.Level));
                }
            }
        }
    }

    /// <summary>Releases the memory of values whose last use was in an earlier dependency level than <paramref name="level"/>.</summary>
    private void FreeDeadValues(List<(Value Root, int Level)> deferred, int level)
    {
        deferred.RemoveAll(entry =>
        {
            if (entry.Level >= level)
            {
                return false;
            }

            _alloc.Free(entry.Root);
            ReleaseRootScratch(entry.Root);
            return true;
        });
    }

    private void LowerUnit(LoweringUnit unit)
    {
        if (unit.Dual is { } dual)
        {
            LowerDualMatMul(dual.A, dual.B, unit.Fused!, unit.FusedOutputs![0]);
            FreeTemps();
        }
        else if (unit.Fused is not null)
        {
            LowerElementwise(unit.Fused, unit.FusedOutputs!, unit.NormTail, unit.NormHead);
        }
        else
        {
            LowerNode(unit.Node!);
        }
    }

    /// <summary>Segment outputs that are views (or inputs) need a dense copy in an output buffer.</summary>
    private void CopyNonMaterializedOutputs()
    {
        foreach (var output in _seg.Outputs)
        {
            if (_outputRefs.ContainsKey(output))
            {
                continue;
            }

            var src = Ref(output);
            var dst = Dense(NewBuffer(BufferKind.Output, "out:" + output.Name, 0), output);
            dst.Buffer.Bytes = dst.MaxBytes;
            EmitCopy(src, dst, $"copy {output.Name}");
            _outputRefs[output] = dst;
        }
    }

    /// <summary>Symbolic dimensions (other than the context length) the segment's tensors depend on.</summary>
    private List<string> CollectSymbols()
        => [.. _seg.Nodes.SelectMany(n => n.Outputs.Concat(n.Inputs)).Select(v => v.Shape.Symbol).OfType<string>().Where(s => s != WellKnown.ContextSymbol).Distinct()];

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

    /// <summary>
    /// True for expert tensors in MXFP4 ([E, N, K], N % 16 == 0) whose consumers (the MatMulId kernels) read the aligned
    /// <see cref="WeightRepack.Mxfp4Split"/> layout.
    /// </summary>
    private static bool UseMxfp4Repack(WeightInfo w)
        => !DiagnosticSwitches.NoWeightRepack && DiagnosticSwitches.Mxfp4Repack && w.Type == DataType.MXFP4 && w.Shape.Rank == 3
           && w.Shape.Dims.All(d => !d.IsSymbolic) && w.Shape[1].Fixed % WeightRepacker.Mxfp4GroupRows == 0 && w.Shape[2].Fixed % 32 == 0;

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
                    else if (UseMxfp4Repack(w))
                    {
                        wb.Repack = WeightRepack.Mxfp4Split;
                        wb.RepackRowElements = w.Shape[-1].Fixed;
                    }
                }
                r = Dense(wb, v);
                if (wb.Repack == WeightRepack.Q8_0Split)
                {
                    r = r with { Type = ShaderTypes.Q8_0R };
                }
                else if (wb.Repack == WeightRepack.Mxfp4Split)
                {
                    r = r with { Type = ShaderTypes.MXFP4R };
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
            Args = WithGridArgument(args),
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
