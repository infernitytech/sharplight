using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>Weight bytes in managed memory; optionally exposes a pinned pointer to exercise the zero-copy path.</summary>
internal sealed class MemSource : IWeightSource, IDisposable
{
    private readonly byte[] _data; private readonly GCHandle _h; private readonly bool _zeroCopy;
    public MemSource(byte[] data, bool zeroCopy = true) { _data = data; _zeroCopy = zeroCopy; if (zeroCopy)
        {
            _h = GCHandle.Alloc(data, GCHandleType.Pinned);
        }
    }
    public long ByteLength => _data.Length;
    public void Read(long offset, Span<byte> destination) => _data.AsSpan((int)offset, destination.Length).CopyTo(destination);
    public nint TryGetPointer() => _zeroCopy ? _h.AddrOfPinnedObject() : 0;
    public void Dispose() { if (_zeroCopy && _h.IsAllocated)
        {
            _h.Free();
        }
    }
}

/// <summary>Builder wrapper that owns weights.</summary>
internal sealed class TG
{
    public readonly Dictionary<string, WeightInfo> Weights = [];
    public readonly GraphBuilder B;
    public static readonly Dim T = Dim.Sym("T");
    public TG(string name = "t") { B = new GraphBuilder(name, n => Weights.GetValueOrDefault(n)); }

    public Value W(string name, DataType type, Shape shape, byte[] data, bool zeroCopy = true)
    {
        Weights[name] = new WeightInfo { Name = name, Type = type, Shape = shape, Source = new MemSource(data, zeroCopy) };
        return B.Weight(name);
    }

    public Value WF(string name, float[] data, params long[] dims) =>
        W(name, DataType.F32, new Shape(dims.Select(d => (Dim)d)), MemoryMarshal.AsBytes(data.AsSpan()).ToArray());

    public Value In(string name, DataType type, params Dim[] dims) => B.Input(name, InputRole.Other, type, new Shape(dims));
    public Value InF(string name, params Dim[] dims) => In(name, DataType.F32, dims);

    public Value State(string name, DataType type, StateKind kind, params Dim[] dims) =>
        B.State(new StateDefinition { Name = name, Type = type, Shape = new Shape(dims), Kind = kind });
}

internal static class Kit
{
    public static CpuExecutionProvider Provider = new();
    public static IComputeDevice Device = Provider.OpenDevice(Provider.GetDevices()[0]);

    public static CompiledModel Compile(ComputeGraph g, long maxTokens = 64, long ctx = 64)
    {
        var opts = new CompileOptions { MaxTokens = maxTokens, ContextLength = ctx };
        return CompiledModel.Compile(g, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = Device }],
            Compile = opts,
            ReservePerDevice = 0,
        });
    }

    public static void Set(CompiledModel m, Value v, float[] data) => m.SetInput(v, MemoryMarshal.AsBytes(data.AsSpan()));
    public static void Set(CompiledModel m, Value v, int[] data) => m.SetInput(v, MemoryMarshal.AsBytes(data.AsSpan()));

    public static RunParameters P(long t, long past = 0, params (string, long)[] extra)
    {
        var p = new RunParameters();
        p.Symbols["T"] = t; p.Symbols["R"] = t;
        p.Scalars["n_past"] = past;
        foreach (var (k, v) in extra)
        {
            p.Symbols[k] = v;
        }

        return p;
    }

    public static float[] Out(CompiledModel m, string name, RunParameters p)
    {
        var v = m.Graph.Outputs[name];
        var r = new float[v.Shape.Elements(p.Symbols)];
        m.GetOutput<float>(name, r);
        return r;
    }

    public static int[] OutI(CompiledModel m, string name, RunParameters p)
    {
        var v = m.Graph.Outputs[name];
        var r = new int[v.Shape.Elements(p.Symbols)];
        m.GetOutput<int>(name, r);
        return r;
    }

    public static float[] Rand(int n, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        }

        return a;
    }

    public static void AssertClose(float[] expected, float[] actual, double atol = 1e-5, double rtol = 1e-4, string what = "")
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double e = expected[i], a = actual[i];
            if (double.IsNaN(e) != double.IsNaN(a) || Math.Abs(e - a) > atol + rtol * Math.Abs(e))
            {
                Assert.Fail($"{what} mismatch at {i}: expected {e:G9}, actual {a:G9} (atol {atol}, rtol {rtol})");
            }
        }
    }

    public static float[] Bytes2F(byte[] b) => MemoryMarshal.Cast<byte, float>(b.AsSpan()).ToArray();
}
