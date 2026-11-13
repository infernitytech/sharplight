using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Cpu.Tests;

/// <summary>Builder wrapper that owns weights.</summary>
internal sealed class TestGraph
{
    public readonly Dictionary<string, WeightInfo> Weights = [];
    public readonly GraphBuilder B;
    public static readonly Dim T = Dim.Sym("T");
    public TestGraph(string name = "t")
    {
        B = new GraphBuilder(name, n => Weights.GetValueOrDefault(n));
    }

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
