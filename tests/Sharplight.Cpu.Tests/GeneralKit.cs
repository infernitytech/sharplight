using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

internal static class GeneralKit
{
    public static long[] Strides(long[] d)
    {
        var s = new long[d.Length];
        long e = 1;
        for (int i = d.Length - 1; i >= 0; i--)
        {
            s[i] = e;
            e *= d[i];
        }
        return s;
    }
    public static long[] Unravel(long l, long[] d)
    {
        var ix = new long[d.Length];
        for (int i = d.Length - 1; i >= 0; i--)
        {
            ix[i] = l % d[i];
            l /= d[i];
        }
        return ix;
    }
    public static long P(long[] d) => d.Aggregate(1L, (a, b) => a * b);

    public static Value SetUp(TestGraph tg, string name, DataType t, params long[] dims) => tg.In(name, t, dims.Select(d => (Dim)d).ToArray());

    public static void SetI64(CompiledModel m, Value v, long[] d) => m.SetInput(v, MemoryMarshal.AsBytes(d.AsSpan()));
    public static long[] OutL(CompiledModel m, string name, RunParameters p)
    {
        var v = m.Graph.Outputs[name];
        var r = new long[v.Shape.Elements(p.Symbols)];
        m.GetOutput<long>(name, r);
        return r;
    }

    public static Attributes A(params (string, object)[] kv) => Attributes.From(kv);
}
