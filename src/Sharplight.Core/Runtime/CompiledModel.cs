using System.Runtime.InteropServices;
using Sharplight.Graph;

namespace Sharplight.Runtime;

/// <summary>
/// A graph compiled onto one or more devices. Executes its segments in order and moves boundary
/// activations between devices through host memory (activations are small compared to weights).
/// </summary>
public sealed class CompiledModel : IDisposable
{
    private readonly ICompiledSegment[] _segments;
    private static readonly HashSet<Value> s_noSkips = [];
    private readonly Dictionary<Value, byte[]> _hostValues = [];
    private readonly Dictionary<Value, string[]> _stringInputs = [];
    private RunParameters? _lastRun;

    /// <summary>The graph that was compiled.</summary>
    public ComputeGraph Graph { get; }
    /// <summary>How the graph was distributed over devices.</summary>
    public PlacementPlan Plan { get; }
    /// <summary>The options the graph was compiled with.</summary>
    public CompileOptions Options { get; }
    /// <summary>The compiled segments in execution order.</summary>
    public IReadOnlyList<ICompiledSegment> Segments => _segments;

    private CompiledModel(ComputeGraph graph, PlacementPlan plan, CompileOptions options, ICompiledSegment[] segments)
    {
        Graph = graph;
        Plan = plan;
        Options = options;
        _segments = segments;
    }

    /// <summary>
    /// Places the graph and compiles every segment on its device.
    /// </summary>
    /// <param name="graph">The graph to compile.</param>
    /// <param name="placement">Devices, fallback, memory reserve and compile options.</param>
    /// <returns>The compiled model; the caller disposes it.</returns>
    /// <exception cref="ArgumentException">No device was given.</exception>
    /// <exception cref="InsufficientMemoryException">A placement group fits on no device and no fallback device is configured.</exception>
    /// <exception cref="NotSupportedException">A node is supported neither by its device nor by the fallback device.</exception>
    public static CompiledModel Compile(ComputeGraph graph, PlacementOptions placement)
    {
        placement = WithDynamicBounds(graph, placement);
        var ct = placement.Compile.CancellationToken;
        ct.ThrowIfCancellationRequested();
        var placing = new LoadProgressCounter(placement.Compile.Progress, LoadStage.Placing, 1, graph.Name);
        var plan = PlacementPlanner.Plan(graph, placement);
        placing.Complete();
        // Segments report through one combined count per stage.
        var progress = placement.Compile.Progress is { } target ? new SegmentProgress(target) : null;
        var segmentOptions = progress is null ? placement.Compile : placement.Compile.WithProgress(progress);
        var compiled = new List<ICompiledSegment>();
        try
        {
            foreach (var seg in plan.Segments)
            {
                ct.ThrowIfCancellationRequested();
                compiled.Add(seg.Nodes switch
                {
                    [{ Op.Name: "Loop", Op.Category: OpCategory.ControlFlow }] => new LoopSegment(seg, placement),
                    [{ Op.Category: OpCategory.ControlFlow }] => new IfSegment(seg, placement),
                    _ => plan.Slots[seg.DeviceSlot].Device.Compile(seg, segmentOptions),
                });
                progress?.NextSegment();
            }
        }
        catch
        {
            foreach (var c in compiled)
            {
                c.Dispose();
            }

            throw;
        }
        return new CompiledModel(graph, plan, placement.Compile, [.. compiled]);
    }

    /// <summary>
    /// Adds the upper bounds of the graph's data-dependent symbols (<see cref="ComputeGraph.DynamicSymbols"/>) to the symbol bounds, so providers size
    /// buffers and specialize kernels for them like for any caller-bound symbol. Bounds are evaluated in creation order, so a bound may refer
    /// to earlier dynamic symbols.
    /// </summary>
    private static PlacementOptions WithDynamicBounds(ComputeGraph graph, PlacementOptions placement)
    {
        if (graph.DynamicSymbols.Count == 0)
        {
            return placement;
        }

        var compile = placement.Compile;
        foreach (var (name, bound) in graph.DynamicSymbols)
        {
            long value = bound.IsSymbolic ? compile.Bound(bound.Symbol!) : bound.Fixed;
            compile = compile.WithSymbolBounds([new(name, Math.Max(1, value))]);
        }

        return new PlacementOptions
        {
            Devices = placement.Devices,
            Fallback = placement.Fallback,
            ReservePerDevice = placement.ReservePerDevice,
            Compile = compile,
        };
    }

    /// <summary>
    /// The actual shape of a graph output after <see cref="Run"/>: symbolic dimensions are evaluated with the caller's bindings and the values
    /// data-dependent ops wrote into <paramref name="parameters"/>.
    /// </summary>
    /// <param name="name">Output name.</param>
    /// <param name="parameters">The parameters passed to the last <see cref="Run"/>.</param>
    public long[] OutputShape(string name, RunParameters parameters) => Graph.Outputs[name].Shape.Resolve(parameters.Symbols);

    /// <summary>Device memory held on accelerators (segments placed on non-CPU devices): what counts against GPU memory budgets.</summary>
    public long AcceleratorMemoryBytes
        => _segments.Where(s => Plan.Slots[s.Segment.DeviceSlot].Device.Info.Kind != DeviceKind.Cpu).Sum(s => s.DeviceMemoryBytes);

    /// <summary>Supplies a graph input for the next <see cref="Run"/> (dense logical layout).</summary>
    public void SetInput(Value input, ReadOnlySpan<byte> data)
    {
        foreach (var s in _segments)
        {
            if (s.Segment.Inputs.Contains(input))
            {
                s.SetInput(input, data);
            }
        }
    }

    /// <summary>Typed overload of <see cref="SetInput(Value, ReadOnlySpan{byte})"/>.</summary>
    public void SetInput<T>(Value input, ReadOnlySpan<T> data) where T : unmanaged
        => SetInput(input, MemoryMarshal.AsBytes(data));

    /// <summary>
    /// Supplies a string graph input (<see cref="Tensors.DataType.Str"/>) for the next <see cref="Run"/>, in row-major order. The strings are
    /// added to <see cref="RunParameters.Strings"/> when the run starts.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="input"/> is not a string value.</exception>
    public void SetInput(Value input, ReadOnlySpan<string> strings)
    {
        if (input.Type != Tensors.DataType.Str)
        {
            throw new ArgumentException($"Input '{input.Name}' is {input.Type}, not a string tensor.", nameof(input));
        }

        _stringInputs[input] = strings.ToArray();
    }

    /// <summary>Reads a string graph output (<see cref="Tensors.DataType.Str"/>) after <see cref="Run"/>, in row-major order.</summary>
    /// <exception cref="InvalidOperationException">Called before <see cref="Run"/>.</exception>
    public void GetOutput(string name, Span<string> destination)
    {
        var run = _lastRun ?? throw new InvalidOperationException("GetOutput called before Run.");
        var handles = new int[destination.Length];
        GetOutput(name, MemoryMarshal.AsBytes(handles.AsSpan()));
        for (int i = 0; i < handles.Length; i++)
        {
            destination[i] = run.Strings[handles[i]];
        }
    }

    /// <summary>
    /// Executes all segments in order, forwarding boundary activations through host memory. Not thread-safe: one run at a time per instance.
    /// </summary>
    /// <param name="parameters">
    /// Bindings for symbolic dimensions and runtime scalars; must cover every caller-bound symbol and scalar the graph uses. Data-dependent ops
    /// add the values of their symbols (<see cref="ComputeGraph.DynamicSymbols"/>) to <see cref="RunParameters.Symbols"/>.
    /// </param>
    public void Run(RunParameters parameters)
    {
        _lastRun = parameters;
        parameters.Strings.Clear();
        foreach (var (input, strings) in _stringInputs)
        {
            var handles = new int[strings.Length];
            for (int i = 0; i < strings.Length; i++)
            {
                handles[i] = parameters.Strings.Add(strings[i]);
            }

            SetInput(input, MemoryMarshal.AsBytes(handles.AsSpan()));
        }

        for (int i = 0; i < _segments.Length; i++)
        {
            var seg = _segments[i];
            // Forward activations produced by earlier segments.
            foreach (var v in seg.Segment.Inputs)
            {
                if (v.Kind == ValueKind.Intermediate && _hostValues.TryGetValue(v, out var bytes))
                {
                    seg.SetInput(v, bytes.AsSpan(0, (int)ByteCount(v, parameters)));
                }
            }

            var skip = SkippedDownloads(i, parameters);
            if (skip.Count == 0)
            {
                seg.Execute(parameters);
            }
            else
            {
                seg.Execute(parameters, skip);
            }

            // Pull values needed by later segments.
            if (i < _segments.Length - 1)
            {
                foreach (var o in seg.Segment.Outputs)
                {
                    if (!IsConsumedAfter(i, o))
                    {
                        continue;
                    }

                    var n = ByteCount(o, parameters);
                    if (!_hostValues.TryGetValue(o, out var buf) || buf.Length < n)
                    {
                        _hostValues[o] = buf = new byte[n];
                    }

                    seg.GetOutput(o, buf.AsSpan(0, (int)n));
                }
            }
        }
    }

    private bool IsConsumedAfter(int segment, Value v) => _segments.Skip(segment + 1).Any(s => s.Segment.Inputs.Contains(v));

    /// <summary>Graph outputs of segment <paramref name="segment"/> listed in <see cref="RunParameters.SkipOutputs"/> that no later segment needs.</summary>
    private HashSet<Value> SkippedDownloads(int segment, RunParameters parameters)
    {
        if (parameters.SkipOutputs.Count == 0)
        {
            return s_noSkips;
        }

        var skip = new HashSet<Value>();
        foreach (var name in parameters.SkipOutputs)
        {
            if (Graph.Outputs.TryGetValue(name, out var v) && _segments[segment].Segment.Outputs.Contains(v) && !IsConsumedAfter(segment, v))
            {
                skip.Add(v);
            }
        }

        return skip;
    }

    /// <summary>Reads a graph output after <see cref="Run"/>.</summary>
    public void GetOutput(string name, Span<byte> destination)
    {
        var v = Graph.Outputs[name];
        var seg = _segments.Last(s => s.Segment.Outputs.Contains(v));
        seg.GetOutput(v, destination);
    }

    /// <summary>
    /// Reads <c>destination.Length</c> bytes of a graph output starting at <paramref name="byteOffset"/> (dense logical layout)
    /// after <see cref="Run"/>. Several threads may read disjoint ranges at once when the producing segment supports partial
    /// reads (<see cref="IPartialOutputSegment"/>); otherwise the output is read whole and sliced.
    /// </summary>
    /// <exception cref="InvalidOperationException">Called before <see cref="Run"/>.</exception>
    public void GetOutput(string name, long byteOffset, Span<byte> destination)
    {
        var v = Graph.Outputs[name];
        var seg = _segments.Last(s => s.Segment.Outputs.Contains(v));
        if (seg is IPartialOutputSegment partial)
        {
            partial.GetOutput(v, byteOffset, destination);
            return;
        }

        var run = _lastRun ?? throw new InvalidOperationException("GetOutput called before Run.");
        var whole = new byte[ByteCount(v, run)];
        seg.GetOutput(v, whole);
        whole.AsSpan(checked((int)byteOffset), destination.Length).CopyTo(destination);
    }

    /// <summary>Typed overload of <see cref="GetOutput(string, Span{byte})"/>.</summary>
    public void GetOutput<T>(string name, Span<T> destination) where T : unmanaged
        => GetOutput(name, MemoryMarshal.AsBytes(destination));

    /// <summary>Clears the persistent state (KV caches, recurrent state) of every segment, starting a new sequence.</summary>
    public void ResetState()
    {
        foreach (var s in _segments)
        {
            s.ResetState();
        }
    }

    internal static long ByteCount(Value v, RunParameters p) => v.Type.ByteSize(v.Shape.Elements(p.Symbols));

    /// <summary>Releases the compiled segments and their device memory.</summary>
    public void Dispose()
    {
        foreach (var s in _segments)
        {
            s.Dispose();
        }
    }
}
