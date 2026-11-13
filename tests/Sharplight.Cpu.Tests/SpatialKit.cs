using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

internal static class SpatialKit
{
    public static Dim[] D(params long[] d) => d.Select(v => (Dim)v).ToArray();
    public static Dim[] DSym(params long[] tail) => [TestGraph.T, .. tail.Select(v => (Dim)v)];

    public static byte[] Half(float[] w, out float[] rounded)
    {
        var b = new byte[w.Length * 2];
        rounded = new float[w.Length];
        for (int i = 0; i < w.Length; i++)
        {
            var h = (Half)w[i];
            BitConverter.TryWriteBytes(b.AsSpan(i * 2), h);
            rounded[i] = (float)h;
        }
        return b;
    }

    /// <summary>Registers a weight in the requested storage type; returns the value and the f32 values the kernel will effectively see.</summary>
    public static Value Weight(TestGraph tg, string name, float[] w, long[] dims, string type, out float[] effective)
    {
        var shape = new Shape(dims.Select(d => (Dim)d));
        switch (type)
        {
            case "f32":
                effective = w;
                return tg.WF(name, w, dims);
            case "f16":
                {
                    var b = Half(w, out effective);
                    return tg.W(name, DataType.F16, shape, b);
                }
            case "q8_0":
                {
                    var q = Quantizers.Q8_0(w);
                    effective = Quantizers.DeqQ8_0(q);
                    return tg.W(name, DataType.Q8_0, shape, q);
                }
            default:
                throw new ArgumentException(type);
        }
    }

}
