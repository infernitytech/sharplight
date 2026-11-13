using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Cpu.Tests;

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
        p.Symbols["T"] = t;
        p.Symbols["R"] = t;
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
            double e = expected[i];
            double a = actual[i];
            if (double.IsNaN(e) != double.IsNaN(a) || Math.Abs(e - a) > atol + rtol * Math.Abs(e))
            {
                Assert.Fail($"{what} mismatch at {i}: expected {e:G9}, actual {a:G9} (atol {atol}, rtol {rtol})");
            }
        }
    }

    public static float[] Bytes2F(byte[] b) => MemoryMarshal.Cast<byte, float>(b.AsSpan()).ToArray();
}
