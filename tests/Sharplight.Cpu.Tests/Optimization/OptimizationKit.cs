using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests.Optimization;

/// <summary>Helpers for the graph optimization tests: constants, op counts and bit-exact comparison of two graphs on the CPU.</summary>
internal static class OptimizationKit
{
    public static Value ConstF(GraphBuilder b, string name, float[] data, params long[] dims)
        => b.Constant(name, DataType.F32, new Shape(dims.Select(d => (Dim)d)), MemoryMarshal.AsBytes(data.AsSpan()).ToArray());

    public static Value ConstI(GraphBuilder b, string name, int[] data, params long[] dims)
        => b.Constant(name, DataType.I32, new Shape(dims.Select(d => (Dim)d)), MemoryMarshal.AsBytes(data.AsSpan()).ToArray());

    public static int Count(ComputeGraph g, string op) => g.Nodes.Count(n => n.Op.Name == op);

    /// <summary>
    /// Compiles both graphs on the CPU, runs each step on both with the same inputs and asserts that every output of
    /// <paramref name="original"/> is bit-identical in <paramref name="optimized"/> after every step.
    /// </summary>
    public static void AssertSameResults(ComputeGraph original, ComputeGraph optimized, IReadOnlyDictionary<Value, byte[]> inputs, params RunParameters[] steps)
    {
        Assert.Equal(original.Outputs.Keys.Order(), optimized.Outputs.Keys.Order());
        using var a = Kit.Compile(original);
        using var b = Kit.Compile(optimized);
        foreach (var p in steps.Length == 0 ? [Kit.P(1)] : steps)
        {
            foreach (var (v, data) in inputs)
            {
                a.SetInput(v, data);
                b.SetInput(v, data);
            }

            a.Run(p);
            b.Run(p);
            foreach (var (name, v) in original.Outputs)
            {
                long bytes = v.Type.ByteSize(v.Shape.Elements(p.Symbols));
                var expected = new byte[bytes];
                var actual = new byte[bytes];
                a.GetOutput(name, expected);
                b.GetOutput(name, actual);
                Assert.True(expected.AsSpan().SequenceEqual(actual), $"output '{name}' differs after optimization");
            }
        }
    }

    public static byte[] Bytes(float[] data) => MemoryMarshal.AsBytes(data.AsSpan()).ToArray();

    public static byte[] Bytes(int[] data) => MemoryMarshal.AsBytes(data.AsSpan()).ToArray();
}
