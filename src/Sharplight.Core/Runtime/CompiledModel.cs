using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Runtime;

/// <summary>
/// A graph compiled onto one or more devices. Executes its segments in order and moves boundary
/// activations between devices through host memory (activations are small compared to weights).
/// </summary>
public sealed class CompiledModel : IDisposable
{
    readonly ICompiledSegment[] _segments;
    readonly Dictionary<Value, byte[]> _hostValues = [];

    public ComputeGraph Graph { get; }
    public PlacementPlan Plan { get; }
    public CompileOptions Options { get; }
    public IReadOnlyList<ICompiledSegment> Segments => _segments;

    CompiledModel(ComputeGraph graph, PlacementPlan plan, CompileOptions options, ICompiledSegment[] segments)
    {
        Graph = graph;
        Plan = plan;
        Options = options;
        _segments = segments;
    }

    public static CompiledModel Compile(ComputeGraph graph, PlacementOptions placement)
    {
        var plan = PlacementPlanner.Plan(graph, placement);
        var compiled = new List<ICompiledSegment>();
        try
        {
            foreach (var seg in plan.Segments)
                compiled.Add(plan.Slots[seg.DeviceSlot].Device.Compile(seg, placement.Compile));
        }
        catch
        {
            foreach (var c in compiled) c.Dispose();
            throw;
        }
        return new CompiledModel(graph, plan, placement.Compile, [.. compiled]);
    }

    /// <summary>Device memory held on accelerators (segments placed on non-CPU devices): what counts against GPU memory budgets.</summary>
    public long AcceleratorMemoryBytes
        => _segments.Where(s => Plan.Slots[s.Segment.DeviceSlot].Device.Info.Kind != DeviceKind.Cpu).Sum(s => s.DeviceMemoryBytes);

    /// <summary>Supplies a graph input for the next <see cref="Run"/> (dense logical layout).</summary>
    public void SetInput(Value input, ReadOnlySpan<byte> data)
    {
        foreach (var s in _segments)
            if (s.Segment.Inputs.Contains(input))
                s.SetInput(input, data);
    }

    public void SetInput<T>(Value input, ReadOnlySpan<T> data) where T : unmanaged
        => SetInput(input, MemoryMarshal.AsBytes(data));

    public void Run(RunParameters parameters)
    {
        for (int i = 0; i < _segments.Length; i++)
        {
            var seg = _segments[i];
            // Forward activations produced by earlier segments.
            foreach (var v in seg.Segment.Inputs)
                if (v.Kind == ValueKind.Intermediate && _hostValues.TryGetValue(v, out var bytes))
                    seg.SetInput(v, bytes.AsSpan(0, (int)ByteCount(v, parameters)));

            seg.Execute(parameters);

            // Pull values needed by later segments.
            if (i < _segments.Length - 1)
                foreach (var o in seg.Segment.Outputs)
                {
                    if (!_segments.Skip(i + 1).Any(s => s.Segment.Inputs.Contains(o))) continue;
                    var n = ByteCount(o, parameters);
                    if (!_hostValues.TryGetValue(o, out var buf) || buf.Length < n)
                        _hostValues[o] = buf = new byte[n];
                    seg.GetOutput(o, buf.AsSpan(0, (int)n));
                }
        }
    }

    /// <summary>Reads a graph output after <see cref="Run"/>.</summary>
    public void GetOutput(string name, Span<byte> destination)
    {
        var v = Graph.Outputs[name];
        var seg = _segments.Last(s => s.Segment.Outputs.Contains(v));
        seg.GetOutput(v, destination);
    }

    public void GetOutput<T>(string name, Span<T> destination) where T : unmanaged
        => GetOutput(name, MemoryMarshal.AsBytes(destination));

    public void ResetState()
    {
        foreach (var s in _segments) s.ResetState();
    }

    public static long ByteCount(Value v, RunParameters p) => v.Type.ByteSize(v.Shape.Elements(p.Symbols));

    public void Dispose()
    {
        foreach (var s in _segments) s.Dispose();
    }
}

/// <summary>Registry of available execution providers (Vulkan, CUDA, CPU, ...).</summary>
public static class ProviderRegistry
{
    static readonly List<Func<IExecutionProvider>> s_factories = [];
    static readonly Lock s_lock = new();

    public static void Register(Func<IExecutionProvider> factory)
    {
        lock (s_lock) s_factories.Add(factory);
    }

    /// <summary>Instantiates all registered providers and reports their availability.</summary>
    public static IReadOnlyList<(IExecutionProvider Provider, ProviderAvailability Availability)> Discover()
    {
        List<Func<IExecutionProvider>> f;
        lock (s_lock) f = [.. s_factories];
        var result = new List<(IExecutionProvider, ProviderAvailability)>();
        foreach (var factory in f)
        {
            IExecutionProvider p;
            try { p = factory(); }
            catch (Exception e) { result.Add((new UnavailableProvider(factory.Method.DeclaringType?.Name ?? "?", e.Message), ProviderAvailability.Unavailable(e.Message))); continue; }
            ProviderAvailability a;
            try { a = p.CheckAvailability(); }
            catch (Exception e) { a = ProviderAvailability.Unavailable(e.Message); }
            result.Add((p, a));
        }
        return result;
    }

    sealed class UnavailableProvider(string name, string reason) : IExecutionProvider
    {
        public string Name => name;
        public ProviderAvailability CheckAvailability() => ProviderAvailability.Unavailable(reason);
        public IReadOnlyList<DeviceInfo> GetDevices() => [];
        public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null) => throw new NotSupportedException(reason);
        public void Dispose() { }
    }
}
